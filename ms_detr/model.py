# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
import torch.nn.functional as F
from torch import nn
import math
import copy
from ms_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx

from ms_detr.matcher import build_matcher
from ms_detr.transformer import build_transformer, build_transformer_single
from ms_detr.position_encoding import build_position_encoding
from ms_detr.misc import accuracy
from ms_detr.dn_components import prepare_for_cdn, dn_post_process, compute_dn_loss
import numpy as np

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

class MS_DETR(nn.Module):

    def __init__(self, transformer, transformer_sf, transformer_clip, position_embed, 
                 txt_position_embed, txt_dim, vid_dim, slowfast_dim, clip_dim, num_queries, input_dropout, 
                 aux_loss=False, interm_neg=False, use_sf=False, use_dn=False,
                 contrastive_align_loss=False, contrastive_hdim=64, use_tgt=False, use_dn_neg=False,
                 max_v_l=75, span_loss_type="l1", use_txt_pos=False, n_input_proj=2, aud_dim=0):
        
        super().__init__()
        self.num_queries = num_queries
        self.transformer = transformer
        hidden_dim = transformer.d_model
        self.position_embed = position_embed
        self.txt_position_embed = txt_position_embed
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        span_pred_dim = 2 if span_loss_type == "l1" else max_v_l * 2
        _span_embed = MLP(hidden_dim, hidden_dim, span_pred_dim, 3)
        _class_embed = nn.Linear(hidden_dim, 2)  # 0: foreground, 1: background,

        prior_prob = 0.01
        self.num_classes = 1
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        _class_embed.bias.data = torch.ones(self.num_classes + 1) * bias_value
        nn.init.constant_(_span_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_span_embed.layers[-1].bias.data, 0)

        span_embed_layerlist = [_span_embed for i in range(transformer.num_decoder_layers)]
        class_embed_layerlist = [_class_embed for i in range(transformer.num_decoder_layers)]
        self.span_embed = nn.ModuleList(span_embed_layerlist)
        self.class_embed = nn.ModuleList(class_embed_layerlist)
        
        self.use_txt_pos = use_txt_pos
        self.n_input_proj = n_input_proj

        relu_args = [True] * 3
        relu_args[n_input_proj-1] = False
        self.input_txt_proj = nn.Sequential(*[
            LinearLayer(txt_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])
        self.input_vid_proj = nn.Sequential(*[
            LinearLayer(vid_dim + aud_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
            LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
        ][:n_input_proj])

        self.saliency_proj1 = nn.Linear(hidden_dim, hidden_dim)
        self.saliency_proj2 = nn.Linear(hidden_dim, hidden_dim)
        self.aux_loss = aux_loss

        self.hidden_dim = hidden_dim
        self.global_rep_token = torch.nn.Parameter(torch.randn(hidden_dim))
        self.global_rep_pos = torch.nn.Parameter(torch.randn(hidden_dim))

        if use_tgt:
            self.interm_neg = interm_neg
            self.transformer.num_queries = num_queries
            self.transformer.enc_out_span_embed = copy.deepcopy(_span_embed)
            self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)
            self.tgt_embed = nn.Embedding(num_queries, hidden_dim)
        else:
            self.query_embed = nn.Embedding(num_queries, 2)

        if use_dn or use_dn_neg:
            self.label_enc = nn.Embedding(self.num_classes + 1, hidden_dim)

        if use_sf:
            self.transformer_sf = transformer_sf
            self.transformer_clip = transformer_clip
            self.input_slowfast_vid_proj = nn.Sequential(*[
                LinearLayer(slowfast_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])
            self.input_clip_vid_proj = nn.Sequential(*[
                LinearLayer(clip_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])
            self.concat_vid_proj = nn.Sequential(*[
                LinearLayer(hidden_dim*2, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[0]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[1]),
                LinearLayer(hidden_dim, hidden_dim, layer_norm=True, dropout=input_dropout, relu=relu_args[2])
            ][:n_input_proj])

        self.contrastive_align_loss = contrastive_align_loss
        if contrastive_align_loss:
            self.contrastive_align_projection_query = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_txt = nn.Linear(hidden_dim, contrastive_hdim)
            self.contrastive_align_projection_vid = nn.Linear(hidden_dim, contrastive_hdim)
        
    def forward(self, src_txt, src_txt_mask, src_txt_neg, src_txt_mask_neg, src_vid, src_vid_mask, opt, src_aud=None, src_aud_mask=None, dn_args=None, mode='train'):
        if opt.use_sf:
            return self.forward_sf_clip(src_txt, src_txt_mask, src_txt_neg, src_txt_mask_neg, src_vid, src_vid_mask, opt, src_aud, src_aud_mask, dn_args, mode)
        else:
            return self.forward_normal(src_txt, src_txt_mask, src_txt_neg, src_txt_mask_neg, src_vid, src_vid_mask, opt, src_aud, src_aud_mask, dn_args, mode)
    
    def forward_normal(self, src_txt, src_txt_mask, src_txt_neg, src_txt_mask_neg, src_vid, src_vid_mask, opt, src_aud, src_aud_mask, dn_args, mode):
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
            
        src_vid = self.input_vid_proj(src_vid)
        src_txt = self.input_txt_proj(src_txt)
        src = torch.cat([src_vid, src_txt], dim=1)  # (bsz, L_vid+L_txt, d)
        mask = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()  # (bsz, L_vid+L_txt)

        pos_vid = self.position_embed(src_vid, src_vid_mask)  # (bsz, L_vid, d)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)  # (bsz, L_txt, d)

        pos = torch.cat([pos_vid, pos_txt], dim=1)

        mask_ = torch.tensor([[True]]).to(mask.device).repeat(mask.shape[0], 1)
        mask = torch.cat([mask_, mask], dim=1)
        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src.shape[0], 1, 1)
        src = torch.cat([src_, src], dim=1)
        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos.shape[0], 1, 1)
        pos = torch.cat([pos_, pos], dim=1)

        video_length = src_vid.shape[1]

        if opt.use_dn:
            input_query_label, input_query_bbox, attn_mask, mask_dict = \
                prepare_for_cdn(dn_args, src.size(0), mode, self.num_queries, self.num_classes,
                            self.hidden_dim, self.label_enc, opt)
        else:
            input_query_label, input_query_bbox, attn_mask, mask_dict = None, None, None, None

        if opt.use_tgt:
            hs, reference, memory, memory_global, hs_enc, ref_enc, init_span_proposal = \
                self.transformer(src, ~mask, self.tgt_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos, video_length=video_length)
        else:
            hs, reference, memory, memory_global = self.transformer(src, ~mask, self.query_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos, video_length=video_length)
        
        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_span_embed, layer_hs) in enumerate(zip(reference, self.span_embed, hs)):
            layer_delta_unsig = layer_span_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            if self.span_loss_type == "l1":
                outputs_coord = layer_outputs_unsig.sigmoid()
            else:
                outputs_coord = layer_outputs_unsig
            outputs_coord_list.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coord_list)
        if opt.use_dn:
            outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)
        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        # for encoder output
        if opt.use_tgt and opt.use_interm:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_spans': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_spans': init_span_proposal}

            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
 
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))

        ### Neg Pairs ###
        if mode == 'train' and opt.use_neg_captions:
            src_txt_neg = self.input_txt_proj(src_txt_neg)
            pos_txt_neg = self.txt_position_embed(src_txt_neg) if self.use_txt_pos else torch.zeros_like(src_txt_neg)  # (bsz, L_txt, d)
            pos_neg = torch.cat([pos_vid, pos_txt_neg], dim=1)
            pos_neg = torch.cat([pos_, pos_neg], dim=1)
        else:
            src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
            src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
            pos_neg = pos.clone()
        src_neg = torch.cat([src_vid, src_txt_neg], dim=1)
        mask_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        mask_neg = torch.cat([mask_, mask_neg], dim=1)
        src_neg = torch.cat([src_, src_neg], dim=1)

        if opt.use_tgt:
            hs_neg, reference_neg, memory_neg, memory_global_neg, hs_enc_neg, ref_enc_neg, init_span_proposal_neg = \
                self.transformer(src_neg, ~mask_neg, self.tgt_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_neg, video_length=video_length)
        else:
            hs_neg, reference_neg, memory_neg, memory_global_neg = \
                self.transformer(src_neg, ~mask_neg, self.query_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_neg, video_length=video_length)
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]


        if opt.use_decoder_neg:
            outputs_class_neg = torch.stack([layer_cls_embed(layer_hs_neg) for
                                        layer_cls_embed, layer_hs_neg in zip(self.class_embed, hs_neg)])
            outputs_coord_list_neg = []
            for dec_lid, (layer_ref_sig_neg, layer_span_embed, layer_hs_neg) in enumerate(zip(reference_neg, self.span_embed, hs_neg)):
                layer_delta_unsig_neg = layer_span_embed(layer_hs_neg)
                layer_outputs_unsig_neg = layer_delta_unsig_neg  + inverse_sigmoid(layer_ref_sig_neg)
                if self.span_loss_type == "l1":
                    outputs_coord_neg = layer_outputs_unsig_neg.sigmoid()
                else:
                    outputs_coord_neg = layer_outputs_unsig_neg
                outputs_coord_list_neg.append(outputs_coord_neg)
            outputs_coord_neg = torch.stack(outputs_coord_list_neg) 
            
            outputs_class_neg, outputs_coord_neg = dn_post_process(outputs_class_neg, outputs_coord_neg, mask_dict)
            out['pred_logits_neg'] = outputs_class_neg[-1]
            out['pred_spans_neg'] = outputs_coord_neg[-1]

        # for encoder output
        if opt.use_tgt and self.interm_neg:
            # prepare intermediate outputs
            interm_coord_neg = ref_enc_neg[-1]
            interm_class_neg = self.transformer.enc_out_class_embed(hs_enc_neg[-1])
            out['interm_outputs_neg'] = {'pred_logits': interm_class_neg, 'pred_spans': interm_coord_neg}
            out['interm_outputs_for_matching_pre_neg'] = {'pred_logits': interm_class_neg, 'pred_spans': init_span_proposal_neg}

            # prepare enc outputs
            if hs_enc_neg.shape[0] > 1:
                enc_outputs_coord_neg = []
                enc_outputs_class_neg = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc_neg, layer_ref_enc_neg) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc_neg[:-1], ref_enc_neg[:-1])):
                    layer_enc_delta_unsig_neg = layer_box_embed(layer_hs_enc_neg)
                    layer_enc_outputs_coord_unsig_neg = layer_enc_delta_unsig_neg + inverse_sigmoid(layer_ref_enc_neg)
                    layer_enc_outputs_coord_neg = layer_enc_outputs_coord_unsig_neg.sigmoid()

                    layer_enc_outputs_class_neg = layer_class_embed(layer_hs_enc_neg)
                    enc_outputs_coord_neg.append(layer_enc_outputs_coord_neg)
                    enc_outputs_class_neg.append(layer_enc_outputs_class_neg)

                out['enc_outputs_neg'] = [
                    {'pred_logits_neg': a, 'pred_boxes_neg': b} for a, b in zip(enc_outputs_class_neg, enc_outputs_coord_neg)
                ]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            if opt.use_decoder_neg:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b, 'pred_logits_neg': c, 'pred_spans_neg': d} \
                        for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_neg[:-1], outputs_coord_neg[:-1])]
            else:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        
        return out, mask_dict

    def forward_sf_clip(self, src_txt, src_txt_mask, src_txt_neg, src_txt_mask_neg, src_vid, src_vid_mask, opt, src_aud=None, src_aud_mask=None, dn_args=None, mode='train'):
        if src_aud is not None:
            src_vid = torch.cat([src_vid, src_aud], dim=2)
        
        src_txt = self.input_txt_proj(src_txt)
        pos_txt = self.txt_position_embed(src_txt) if self.use_txt_pos else torch.zeros_like(src_txt)

        if opt.dset_name == 'tvsum':
            slowfast_vid = src_vid[:, :, :1024]  # shape: (batch_size, seq_len, 1024)
            clip_vid = src_vid[:, :, 1024:2048]  # shape: (batch_size, seq_len, 1024)
            tef_vid = src_vid[:, :, -2:]
        else:
            slowfast_vid = src_vid[:, :, :2304]  # shape: (batch_size, seq_len, 2306)
            clip_vid = src_vid[:, :, 2304:2816]  # shape: (batch_size, seq_len, 512)
            tef_vid = src_vid[:, :, -2:]
    
        slowfast_vid = torch.cat([slowfast_vid, tef_vid], dim=-1)
        clip_vid = torch.cat([clip_vid, tef_vid], dim=-1)
        
        slowfast_vid = self.input_slowfast_vid_proj(slowfast_vid) # 75,256
        clip_vid = self.input_clip_vid_proj(clip_vid) # 75, 256

        src_sf = torch.cat([slowfast_vid, src_txt], dim=1)
        src_clip = torch.cat([clip_vid, src_txt], dim=1)

        mask_sf = mask_clip = torch.cat([src_vid_mask, src_txt_mask], dim=1).bool()

        pos_sf = self.position_embed(slowfast_vid, src_vid_mask)
        pos_sf = torch.cat([pos_sf, pos_txt], dim=1)
        pos_clip = self.position_embed(clip_vid, src_vid_mask)
        pos_clip = torch.cat([pos_clip, pos_txt], dim=1)

        mask_ = torch.tensor([[True]]).to(mask_sf.device).repeat(mask_sf.shape[0], 1)
        mask_sf = torch.cat([mask_, mask_sf], dim=1)
        mask_clip = torch.cat([mask_, mask_clip], dim=1)

        src_ = self.global_rep_token.reshape([1, 1, self.hidden_dim]).repeat(src_sf.shape[0], 1, 1)
        src_sf = torch.cat([src_, src_sf], dim=1)
        src_clip = torch.cat([src_, src_clip], dim=1)

        pos_ = self.global_rep_pos.reshape([1, 1, self.hidden_dim]).repeat(pos_sf.shape[0], 1, 1)
        pos_sf = torch.cat([pos_, pos_sf], dim=1)
        pos_clip = torch.cat([pos_, pos_clip], dim=1)

        video_length_sf = slowfast_vid.shape[1]
        video_length_clip = clip_vid.shape[1]

        src_sf_new, mask_sf1, pos_sf1, src_txt_sf = self.transformer_sf(src_sf, ~mask_sf, pos_sf, video_length=video_length_sf)
        src_clip_new, mask_clip1, pos_clip1, src_txt_clip = self.transformer_clip(src_clip, ~mask_clip, pos_clip, video_length=video_length_clip)
        
        assert torch.equal(src_txt_sf, src_txt_clip)
        assert torch.equal(mask_sf1, mask_clip1)
        assert torch.equal(pos_sf1, pos_clip1)

        src_combine = torch.cat([src_sf_new, src_clip_new], dim=-1)
        src_combine = self.concat_vid_proj(src_combine)

        assert video_length_clip == video_length_sf
        video_length = video_length_clip

        if opt.use_dn:
            input_query_label, input_query_bbox, attn_mask, mask_dict = \
            prepare_for_cdn(dn_args, src_sf.size(0), mode, self.num_queries, self.num_classes,
                        self.hidden_dim, self.label_enc, opt)
        else:
            input_query_label, input_query_bbox, attn_mask, mask_dict = None, None, None, None
         
        if opt.use_tgt:
            hs, reference, memory, memory_global, hs_enc, ref_enc, init_span_proposal = \
                self.transformer(src_combine, ~mask_sf, self.tgt_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_sf1, video_length=video_length)
        else:
            hs, reference, memory, memory_global = self.transformer(src_combine, ~mask_sf, self.query_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_sf1, video_length=video_length)
        
        # hs, reference, memory, memory_global, hs_enc, ref_enc, init_span_proposal = self.transformer(src_combine, ~mask_sf, self.tgt_embed.weight, input_query_label, input_query_bbox, attn_mask, pos_sf1, video_length=video_length)

        outputs_class = torch.stack([layer_cls_embed(layer_hs) for
                                     layer_cls_embed, layer_hs in zip(self.class_embed, hs)])
        
        outputs_coord_list = []
        for dec_lid, (layer_ref_sig, layer_span_embed, layer_hs) in enumerate(zip(reference, self.span_embed, hs)):
            layer_delta_unsig = layer_span_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig  + inverse_sigmoid(layer_ref_sig)
            if self.span_loss_type == "l1":
                outputs_coord = layer_outputs_unsig.sigmoid()
            else:
                outputs_coord = layer_outputs_unsig
            outputs_coord_list.append(outputs_coord)
        outputs_coord = torch.stack(outputs_coord_list) 
        
        if opt.use_dn:
            outputs_class, outputs_coord = dn_post_process(outputs_class, outputs_coord, mask_dict)

        out = {'pred_logits': outputs_class[-1], 'pred_spans': outputs_coord[-1]}

        # for encoder output
        if opt.use_tgt and opt.use_interm:
            interm_coord = ref_enc[-1]
            interm_class = self.transformer.enc_out_class_embed(hs_enc[-1])
            out['interm_outputs'] = {'pred_logits': interm_class, 'pred_spans': interm_coord}
            out['interm_outputs_for_matching_pre'] = {'pred_logits': interm_class, 'pred_spans': init_span_proposal}

            if hs_enc.shape[0] > 1:
                enc_outputs_coord = []
                enc_outputs_class = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc, layer_ref_enc) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc[:-1], ref_enc[:-1])):
                    layer_enc_delta_unsig = layer_box_embed(layer_hs_enc)
                    layer_enc_outputs_coord_unsig = layer_enc_delta_unsig + inverse_sigmoid(layer_ref_enc)
                    layer_enc_outputs_coord = layer_enc_outputs_coord_unsig.sigmoid()

                    layer_enc_outputs_class = layer_class_embed(layer_hs_enc)
                    enc_outputs_coord.append(layer_enc_outputs_coord)
                    enc_outputs_class.append(layer_enc_outputs_class)

                out['enc_outputs'] = [
                    {'pred_logits': a, 'pred_boxes': b} for a, b in zip(enc_outputs_class, enc_outputs_coord)
                ]
 
        txt_mem = memory[:, src_vid.shape[1]:]  # (bsz, L_txt, d)
        vid_mem = memory[:, :src_vid.shape[1]]  # (bsz, L_vid, d)
        if self.contrastive_align_loss:
            proj_queries = F.normalize(self.contrastive_align_projection_query(hs), p=2, dim=-1)
            proj_txt_mem = F.normalize(self.contrastive_align_projection_txt(txt_mem), p=2, dim=-1)
            proj_vid_mem = F.normalize(self.contrastive_align_projection_vid(vid_mem), p=2, dim=-1)
            out.update(dict(
                proj_queries=proj_queries[-1],
                proj_txt_mem=proj_txt_mem,
                proj_vid_mem=proj_vid_mem
            ))
            
        # !!! this is code for test
        if src_txt.shape[1] == 0:
            print("There is zero text query. You should change codes properly")
            exit(-1)

        ### Neg Pairs ###
        if mode == 'train' and opt.use_neg_captions:
            src_txt_neg = self.input_txt_proj(src_txt_neg)
            pos_txt_neg = self.txt_position_embed(src_txt_neg) if self.use_txt_pos else torch.zeros_like(src_txt_neg)
            pos_sf_neg = self.position_embed(slowfast_vid, src_vid_mask)
            pos_sf_neg = torch.cat([pos_sf_neg, pos_txt_neg], dim=1)
            pos_clip_neg = self.position_embed(clip_vid, src_vid_mask)
            pos_clip_neg = torch.cat([pos_clip_neg, pos_txt_neg], dim=1)
            pos_sf_neg = torch.cat([pos_, pos_sf_neg], dim=1)
            pos_clip_neg = torch.cat([pos_, pos_clip_neg], dim=1)
        else:
            src_txt_neg = torch.cat([src_txt[1:], src_txt[0:1]], dim=0)
            src_txt_mask_neg = torch.cat([src_txt_mask[1:], src_txt_mask[0:1]], dim=0)
            pos_sf_neg = pos_sf.clone()
            pos_clip_neg = pos_clip.clone()

        src_sf_neg = torch.cat([slowfast_vid, src_txt_neg], dim=1)
        src_clip_neg = torch.cat([clip_vid, src_txt_neg], dim=1)

        mask_sf_neg = mask_clip_neg = torch.cat([src_vid_mask, src_txt_mask_neg], dim=1).bool()

        src_sf_neg = torch.cat([src_, src_sf_neg], dim=1)
        src_clip_neg = torch.cat([src_, src_clip_neg], dim=1)
        mask_sf_neg = torch.cat([mask_, mask_sf_neg], dim=1)
        mask_clip_neg = torch.cat([mask_, mask_clip_neg], dim=1)

        video_length_sf = slowfast_vid.shape[1]
        video_length_clip = clip_vid.shape[1]

        src_sf_new_neg, mask_sf1_neg, pos_sf1_neg, src_txt_sf_neg = self.transformer_sf(src_sf_neg, ~mask_sf_neg, pos_sf_neg, video_length=video_length_sf)
        src_clip_new_neg, mask_clip1_neg, pos_clip1_neg, src_txt_clip_neg = self.transformer_clip(src_clip_neg, ~mask_clip_neg, pos_clip_neg, video_length=video_length_clip)
        
        assert torch.equal(src_txt_sf_neg, src_txt_clip_neg)
        assert torch.equal(mask_sf1_neg, mask_clip1_neg)
        assert torch.equal(pos_sf1_neg, pos_clip1_neg)

        src_combine_neg = torch.cat([src_sf_new_neg, src_clip_new_neg], dim=-1)
        src_combine_neg = self.concat_vid_proj(src_combine_neg)

        if opt.use_tgt:
            hs_neg, reference_neg, memory_neg, memory_global_neg, hs_enc_neg, ref_enc_neg, init_span_proposal_neg = \
                self.transformer(src_combine_neg, ~mask_sf_neg, self.tgt_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_sf1_neg, video_length=video_length)
        else:
            hs_neg, reference_neg, memory_neg, memory_global_neg = \
                self.transformer(src_combine_neg, ~mask_sf_neg, self.query_embed.weight, input_query_label, input_query_bbox, attn_mask, opt, pos_sf1_neg, video_length=video_length)
            
        vid_mem_neg = memory_neg[:, :src_vid.shape[1]]

        if opt.use_decoder_neg:
            outputs_class_neg = torch.stack([layer_cls_embed(layer_hs_neg) for
                                        layer_cls_embed, layer_hs_neg in zip(self.class_embed, hs_neg)])
            outputs_coord_list_neg = []
            for dec_lid, (layer_ref_sig_neg, layer_span_embed, layer_hs_neg) in enumerate(zip(reference_neg, self.span_embed, hs_neg)):
                layer_delta_unsig_neg = layer_span_embed(layer_hs_neg)
                layer_outputs_unsig_neg = layer_delta_unsig_neg  + inverse_sigmoid(layer_ref_sig_neg)
                if self.span_loss_type == "l1":
                    outputs_coord_neg = layer_outputs_unsig_neg.sigmoid()
                else:
                    outputs_coord_neg = layer_outputs_unsig_neg
                outputs_coord_list_neg.append(outputs_coord_neg)
            outputs_coord_neg = torch.stack(outputs_coord_list_neg) 
            
            outputs_class_neg, outputs_coord_neg = dn_post_process(outputs_class_neg, outputs_coord_neg, mask_dict)
            out['pred_logits_neg'] = outputs_class_neg[-1]
            out['pred_spans_neg'] = outputs_coord_neg[-1]

        # for encoder output
        if opt.use_tgt and self.interm_neg:
            # prepare intermediate outputs
            interm_coord_neg = ref_enc_neg[-1]
            interm_class_neg = self.transformer.enc_out_class_embed(hs_enc_neg[-1])
            out['interm_outputs_neg'] = {'pred_logits': interm_class_neg, 'pred_spans': interm_coord_neg}
            out['interm_outputs_for_matching_pre_neg'] = {'pred_logits': interm_class_neg, 'pred_spans': init_span_proposal_neg}

            # prepare enc outputs
            if hs_enc_neg.shape[0] > 1:
                enc_outputs_coord_neg = []
                enc_outputs_class_neg = []
                for layer_id, (layer_box_embed, layer_class_embed, layer_hs_enc_neg, layer_ref_enc_neg) in enumerate(zip(self.enc_bbox_embed, self.enc_class_embed, hs_enc_neg[:-1], ref_enc_neg[:-1])):
                    layer_enc_delta_unsig_neg = layer_box_embed(layer_hs_enc_neg)
                    layer_enc_outputs_coord_unsig_neg = layer_enc_delta_unsig_neg + inverse_sigmoid(layer_ref_enc_neg)
                    layer_enc_outputs_coord_neg = layer_enc_outputs_coord_unsig_neg.sigmoid()

                    layer_enc_outputs_class_neg = layer_class_embed(layer_hs_enc_neg)
                    enc_outputs_coord_neg.append(layer_enc_outputs_coord_neg)
                    enc_outputs_class_neg.append(layer_enc_outputs_class_neg)

                out['enc_outputs_neg'] = [
                    {'pred_logits_neg': a, 'pred_boxes_neg': b} for a, b in zip(enc_outputs_class_neg, enc_outputs_coord_neg)
                ]

        out["saliency_scores"] = (torch.sum(self.saliency_proj1(vid_mem) * self.saliency_proj2(memory_global).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))
        out["saliency_scores_neg"] = (torch.sum(self.saliency_proj1(vid_mem_neg) * self.saliency_proj2(memory_global_neg).unsqueeze(1), dim=-1) / np.sqrt(self.hidden_dim))

        out["video_mask"] = src_vid_mask
        if self.aux_loss:
            if opt.use_decoder_neg:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b, 'pred_logits_neg': c, 'pred_spans_neg': d} \
                        for a, b, c, d in zip(outputs_class[:-1], outputs_coord[:-1], outputs_class_neg[:-1], outputs_coord_neg[:-1])]
            else:
                out['aux_outputs'] = [
                    {'pred_logits': a, 'pred_spans': b} for a, b in zip(outputs_class[:-1], outputs_coord[:-1])]
            if self.contrastive_align_loss:
                assert proj_queries is not None
                for idx, d in enumerate(proj_queries[:-1]):
                    out['aux_outputs'][idx].update(dict(proj_queries=d, proj_txt_mem=proj_txt_mem))
        
        return out, mask_dict

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """

    def __init__(self, matcher, focal_alpha, weight_dict, eos_coef, losses, temperature, span_loss_type, max_v_l,
                 saliency_margin=1, use_matcher=True):
        
        super().__init__()
        self.matcher = matcher
        self.focal_alpha = focal_alpha
        self.weight_dict = weight_dict
        self.losses = losses
        self.temperature = temperature
        self.span_loss_type = span_loss_type
        self.max_v_l = max_v_l
        self.saliency_margin = saliency_margin

        # foreground and background classification
        self.foreground_label = 0
        self.background_label = 1
        self.eos_coef = eos_coef
        empty_weight = torch.ones(2)
        empty_weight[-1] = self.eos_coef  # lower weight for background (index 1, foreground index 0)
        self.register_buffer('empty_weight', empty_weight)
        
        # for tvsum,
        self.use_matcher = use_matcher

    def loss_spans(self, outputs, targets, indices):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "spans" containing a tensor of dim [nb_tgt_spans, 2]
           The target spans are expected in format (center_x, w), normalized by the image size.
        """
        assert 'pred_spans' in outputs
        targets = targets["span_labels"]
        idx = self._get_src_permutation_idx(indices)
        src_spans = outputs['pred_spans'][idx]  # (#spans, max_v_l * 2)
        tgt_spans = torch.cat([t['spans'][i] for t, (_, i) in zip(targets, indices)], dim=0)  # (#spans, 2)
        if self.span_loss_type == "l1":
            loss_span = F.l1_loss(src_spans, tgt_spans, reduction='none')
            loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_spans), span_cxw_to_xx(tgt_spans)))
        else:  # ce
            n_spans = src_spans.shape[0]
            src_spans = src_spans.view(n_spans, 2, self.max_v_l).transpose(1, 2)
            loss_span = F.cross_entropy(src_spans, tgt_spans, reduction='none')
          
            loss_giou = loss_span.new_zeros([1])

        losses = {}
        losses['loss_span'] = loss_span.mean()
        losses['loss_giou'] = loss_giou.mean()
        return losses

    def loss_labels(self, outputs, targets, indices, log=True):
        
        if log:
            assert 'pred_logits' in outputs
            src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
            idx = self._get_src_permutation_idx(indices)
            target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                        dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
            target_classes[idx] = self.foreground_label

            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
            losses = {'loss_label': loss_ce.mean()}

            if log:
                losses['class_error'] = 100 - accuracy(src_logits[idx], self.foreground_label)[0]

            if 'pred_logits_neg' in outputs.keys():
                src_logits_neg = outputs['pred_logits_neg']  # (batch_size, #queries, #classes=2)
                target_classes_neg = torch.full(src_logits_neg.shape[:2], self.background_label,
                                            dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
                loss_ce_neg = F.cross_entropy(src_logits_neg.transpose(1, 2), target_classes_neg, self.empty_weight, reduction="none")
                losses['loss_label'] += loss_ce_neg.mean()

            return losses
        else:
            src_logits = outputs['pred_logits']  # (batch_size, #queries, #classes=2)
            target_classes = torch.full(src_logits.shape[:2], self.background_label,
                                        dtype=torch.int64, device=src_logits.device)  # (batch_size, #queries)
            loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight, reduction="none")
            losses = {'loss_label': loss_ce.mean()}

            return losses

    def loss_saliency(self, outputs, targets, indices, log=True):
        """higher scores for positive clips"""
        if "saliency_pos_labels" not in targets:
            return {"loss_saliency": 0}

        vid_token_mask = outputs["video_mask"]

        # Neg pair loss
        saliency_scores_neg = outputs["saliency_scores_neg"].clone()  # (N, L)
        
        loss_neg_pair = (- torch.log(1. - torch.sigmoid(saliency_scores_neg)) * vid_token_mask).sum(dim=1).mean()

        saliency_scores = outputs["saliency_scores"].clone()  # (N, L)
        saliency_contrast_label = targets["saliency_all_labels"]

        saliency_scores = torch.cat([saliency_scores, saliency_scores_neg], dim=1)
        saliency_contrast_label = torch.cat([saliency_contrast_label, torch.zeros_like(saliency_contrast_label)], dim=1)

        vid_token_mask = vid_token_mask.repeat([1, 2])
        saliency_scores = vid_token_mask * saliency_scores + (1. - vid_token_mask) * -1e+3

        tau = 0.5
        loss_rank_contrastive = 0.

        for rand_idx in range(1, 12):
            drop_mask = ~(saliency_contrast_label > 100)  # no drop
            pos_mask = (saliency_contrast_label >= rand_idx)  # positive when equal or higher than rand_idx

            if torch.sum(pos_mask) == 0:  # no positive sample
                continue
            else:
                batch_drop_mask = torch.sum(pos_mask, dim=1) > 0  # negative sample indicator

            # drop higher ranks
            cur_saliency_scores = saliency_scores * drop_mask / tau + ~drop_mask * -1e+3
            # numerical stability
            logits = cur_saliency_scores - torch.max(cur_saliency_scores, dim=1, keepdim=True)[0]
            # softmax
            exp_logits = torch.exp(logits)
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-6)
            mean_log_prob_pos = (pos_mask * log_prob * vid_token_mask).sum(1) / (pos_mask.sum(1) + 1e-6)
            loss = - mean_log_prob_pos * batch_drop_mask
            loss_rank_contrastive = loss_rank_contrastive + loss.mean()

        loss_rank_contrastive = loss_rank_contrastive / 12

        saliency_scores = outputs["saliency_scores"]  # (N, L)
        pos_indices = targets["saliency_pos_labels"]  # (N, #pairs)
        neg_indices = targets["saliency_neg_labels"]  # (N, #pairs)
        num_pairs = pos_indices.shape[1]  # typically 2 or 4
        batch_indices = torch.arange(len(saliency_scores)).to(saliency_scores.device)
        pos_scores = torch.stack(
            [saliency_scores[batch_indices, pos_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        neg_scores = torch.stack(
            [saliency_scores[batch_indices, neg_indices[:, col_idx]] for col_idx in range(num_pairs)], dim=1)
        loss_saliency = torch.clamp(self.saliency_margin + neg_scores - pos_scores, min=0).sum() \
                        / (len(pos_scores) * num_pairs) * 2  # * 2 to keep the loss the same scale

        loss_saliency = loss_saliency + loss_rank_contrastive + loss_neg_pair
        return {"loss_saliency": loss_saliency}

    def loss_contrastive_align(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def loss_contrastive_align_vid_txt(self, outputs, targets, indices, log=True):
        """encourage higher scores between matched query span and input text"""
        # TODO (1)  align vid_mem and txt_mem;
        # TODO (2) change L1 loss as CE loss on 75 labels, similar to soft token prediction in MDETR
        normalized_text_embed = outputs["proj_txt_mem"]  # (bsz, #tokens, d)  text tokens
        normalized_img_embed = outputs["proj_queries"]  # (bsz, #queries, d)
        logits = torch.einsum(
            "bmd,bnd->bmn", normalized_img_embed, normalized_text_embed)  # (bsz, #queries, #tokens)
        logits = logits.sum(2) / self.temperature  # (bsz, #queries)
        idx = self._get_src_permutation_idx(indices)
        positive_map = torch.zeros_like(logits, dtype=torch.bool)
        positive_map[idx] = True
        positive_logits = logits.masked_fill(~positive_map, 0)

        pos_term = positive_logits.sum(1)  # (bsz, )
        num_pos = positive_map.sum(1)  # (bsz, )
        neg_term = logits.logsumexp(1)  # (bsz, )
        loss_nce = - pos_term / num_pos + neg_term  # (bsz, )
        losses = {"loss_contrastive_align": loss_nce.mean()}
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx  # two 1D tensors of the same length

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, **kwargs):
        loss_map = {
            "spans": self.loss_spans,
            "labels": self.loss_labels,
            "contrastive_align": self.loss_contrastive_align,
            "saliency": self.loss_saliency,
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, **kwargs)

    def forward(self, outputs, targets, mask_dict, opt, mode='val'):
       
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}
        # only for HL, do not use matcher
        if self.use_matcher:
            indices = self.matcher(outputs_without_aux, targets)
            losses_target = self.losses

            # if opt.use_decoder_neg:
            #     outputs_without_aux['pred_logits'] = outputs_without_aux['pred_logits_neg']
            #     outputs_without_aux['pred_spans'] = outputs_without_aux['pred_spans_neg']
            #     neg_indices = self.matcher(outputs_without_aux, targets)
            # else:
            #     neg_indices = None
        else:
            indices = None
            losses_target = ["saliency"]

        # Compute all the requested losses
        losses = {}
        # for loss in self.losses:
        for loss in losses_target:
            losses.update(self.get_loss(loss, outputs, targets, indices))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                if self.use_matcher:
                    indices = self.matcher(aux_outputs, targets)
                    losses_target = self.losses
                else:
                    indices = None
                    losses_target = ["saliency"]    
                # for loss in self.losses:
                for loss in losses_target:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, aux_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        # interm_outputs loss
        if opt.dset_name not in ["tvsum", "youtube_uni"]:
            if 'interm_outputs' in outputs:
                interm_outputs = outputs['interm_outputs']
                indices = self.matcher(interm_outputs, targets)
    
                for loss in self.losses:
                    if "saliency" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {}
                    l_dict = self.get_loss(loss, interm_outputs, targets, indices, **kwargs)
                    l_dict = {k + f'_interm': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
            if 'interm_outputs_neg' in outputs:
                interm_outputs_neg = outputs['interm_outputs_neg']
                indices = None

                for loss in self.losses:
                    if "saliency" == loss or "spans" == loss:  # skip as it is only in the top layer
                        continue
                    kwargs = {'log': False}
                    l_dict = self.get_loss(loss, interm_outputs_neg, targets, indices, **kwargs)
                    l_dict = {k + f'_interm_neg': v for k, v in l_dict.items()}
                    losses.update(l_dict)
        
            aux_num = 0
            if 'aux_outputs' in outputs:
                aux_num = len(outputs['aux_outputs'])
            
            if opt.use_dn:
                dn_losses = compute_dn_loss(mask_dict, mode, aux_num, self.focal_alpha)
                losses.update(dn_losses)
        
        return losses


class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class LinearLayer(nn.Module):
    """linear layer configurable with layer normalization, dropout, ReLU."""

    def __init__(self, in_hsz, out_hsz, layer_norm=True, dropout=0.1, relu=True):
        super(LinearLayer, self).__init__()
        self.relu = relu
        self.layer_norm = layer_norm
        if layer_norm:
            self.LayerNorm = nn.LayerNorm(in_hsz)
        layers = [
            nn.Dropout(dropout),
            nn.Linear(in_hsz, out_hsz)
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """(N, L, D)"""
        if self.layer_norm:
            x = self.LayerNorm(x)
        x = self.net(x)
        if self.relu:
            x = F.relu(x, inplace=True)
        return x  # (N, L, D)


def build_model(args):

    device = torch.device(args.device)

    transformer = build_transformer(args)
    if args.use_sf:
        transformer_sf  = build_transformer_single(args)
        transformer_clip = build_transformer_single(args)
    else:
        transformer_sf  = None
        transformer_clip = None

    position_embedding, txt_position_embedding = build_position_encoding(args)

    if args.a_feat_dir is None:
        model = MS_DETR(
            transformer,
            transformer_sf,
            transformer_clip,
            position_embedding,
            txt_position_embedding,
            interm_neg=args.interm_neg,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            slowfast_dim=args.slowfast_dim,
            clip_dim=args.clip_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            use_sf=args.use_sf,
            use_dn=args.use_dn,
            use_tgt=args.use_tgt,
            use_dn_neg=args.use_dn_neg,
        )
    else:
        model = MS_DETR(
            transformer,
            transformer_sf,
            transformer_clip,
            position_embedding,
            txt_position_embedding,
            interm_neg=args.interm_neg,
            txt_dim=args.t_feat_dim,
            vid_dim=args.v_feat_dim,
            slowfast_dim=args.slowfast_dim,
            clip_dim=args.clip_dim,
            aud_dim=args.a_feat_dim,
            num_queries=args.num_queries,
            input_dropout=args.input_dropout,
            aux_loss=args.aux_loss,
            contrastive_align_loss=args.contrastive_align_loss,
            contrastive_hdim=args.contrastive_hdim,
            span_loss_type=args.span_loss_type,
            use_txt_pos=args.use_txt_pos,
            n_input_proj=args.n_input_proj,
            use_sf=args.use_sf,
            use_dn=args.use_dn,
            use_tgt=args.use_tgt,
            use_dn_neg=args.use_dn_neg,
        )

    matcher = build_matcher(args)
    weight_dict = {"loss_span": args.span_loss_coef,
                   "loss_giou": args.giou_loss_coef,
                   "loss_label": args.label_loss_coef,
                   "loss_saliency": args.lw_saliency}
    
    clean_weight_dict_wo_dn = copy.deepcopy(weight_dict)

    if args.contrastive_align_loss:
        weight_dict["loss_contrastive_align"] = args.contrastive_align_loss_coef
    if args.use_dn or args.use_dn_neg:
        weight_dict['tgt_loss_ce'] = args.label_loss_coef
        weight_dict['tgt_loss_bbox'] = args.span_loss_coef
        weight_dict['tgt_loss_giou'] = args.giou_loss_coef

    # TODO this is a hack
    if args.aux_loss:
        aux_weight_dict = {}
        for i in range(args.dec_layers - 1):
            aux_weight_dict.update({k + f'_{i}': v for k, v in weight_dict.items() if k != "loss_saliency"})
        weight_dict.update(aux_weight_dict)
        
    if args.use_interm:
        interm_weight_dict = {}
        _coeff_weight_dict = {
            'loss_label': 1.0,
            'loss_span': 1.0,
            'loss_giou': 1.0,
        }
        interm_loss_coef = args.interm_loss_coef
        interm_weight_dict.update({k + f'_interm': v * interm_loss_coef * _coeff_weight_dict[k] for k, v in clean_weight_dict_wo_dn.items() if k in _coeff_weight_dict.keys()})
        weight_dict.update(interm_weight_dict)
        if args.interm_neg:
            weight_dict['loss_label_interm_neg'] = weight_dict['loss_label_interm']

    losses = ['spans', 'labels', 'saliency']
    if args.contrastive_align_loss:
        losses += ["contrastive_align"]
        
    # For tvsum dataset
    use_matcher = not (args.dset_name == 'tvsum')
    focal_alpha = args.focal_alpha
        
    criterion = SetCriterion(
        matcher=matcher, focal_alpha=focal_alpha, 
        weight_dict=weight_dict, losses=losses,
        eos_coef=args.eos_coef, temperature=args.temperature,
        span_loss_type=args.span_loss_type, max_v_l=args.max_v_l,
        saliency_margin=args.saliency_margin, use_matcher=use_matcher,
    )
    criterion.to(device)
    return model, criterion
