import torch
from core.misc import (NestedTensor, nested_tensor_from_tensor_list,
                       accuracy, get_world_size, interpolate,
                       is_dist_avail_and_initialized, inverse_sigmoid)
# from .DABDETR import sigmoid_focal_loss
from util import box_ops
import torch.nn.functional as F
from ms_detr.span_utils import generalized_temporal_iou, span_cxw_to_xx


def sigmoid_focal_loss(inputs, targets, num_boxes, alpha: float = 0.25, gamma: float = 2):
    
    prob = inputs.sigmoid()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss.mean(1).sum() / num_boxes

def prepare_for_cdn(dn_args, batch_size, mode, num_queries, num_classes, hidden_dim, label_enc, opt):
    
    training = (mode == 'train')
    if training and opt.use_dn:
        targets, dn_number, label_noise_ratio, box_noise_scale, num_patterns = dn_args
    else:
        num_patterns = dn_args[-1]

    if num_patterns == 0:
        num_patterns = 1
        
    if training and opt.use_dn:
        if opt.use_dn_neg:
            dn_number = dn_number
        
        known = [(torch.ones_like(t['labels'])).cuda() for t in targets] 
        know_idx = [torch.nonzero(t) for t in known] 
        known_num = [sum(k) for k in known]

        if int(max(known_num)) == 0:
            dn_number = 1
        else:
            if dn_number >= 100:
                dn_number = dn_number // int(max(known_num))
            elif dn_number < 1:
                dn_number = 1
        if dn_number == 0:
            dn_number = 1

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t['labels'] for t in targets])
        boxes = torch.cat([t['boxes'] for t in targets])
        batch_idx = torch.cat([torch.full_like(t['labels'].long(), i) for i, t in enumerate(targets)]) 

        known_indice = torch.nonzero(unmask_label + unmask_bbox)
        known_indice = known_indice.view(-1)
        
        if opt.use_dn_neg:
            known_indice = known_indice.repeat(2 * dn_number, 1).view(-1) 
            known_labels = labels.repeat(2 * dn_number, 1).view(-1)
            known_bid = batch_idx.repeat(2 * dn_number, 1).view(-1)
            known_bboxs = boxes.repeat(2 * dn_number, 1)
        else:
            known_indice = known_indice.repeat(dn_number, 1).view(-1)
            known_labels = labels.repeat(dn_number, 1).view(-1)
            known_bid = batch_idx.repeat(dn_number, 1).view(-1)
            known_bboxs = boxes.repeat(dn_number, 1)

        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if label_noise_ratio > 0:
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (label_noise_ratio)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, num_classes+1)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)

        positive_idx = torch.tensor(range(len(boxes))).long().cuda().unsqueeze(0).repeat(dn_number, 1)
        if opt.use_dn_neg:
            positive_idx += (torch.tensor(range(dn_number)) * len(boxes) * 2).long().cuda().unsqueeze(1)
            positive_idx = positive_idx.flatten()
            negative_idx = positive_idx + len(boxes)
        else:
            positive_idx += (torch.tensor(range(dn_number)) * len(boxes)).long().cuda().unsqueeze(1)
            positive_idx = positive_idx.flatten()

        if box_noise_scale > 0:
            known_bbox_ = torch.zeros_like(known_bboxs) 
            known_bbox_[:, :1] = known_bboxs[:, :1] - known_bboxs[:, 1:] / 2 # (cx, cy, w, h)->(x,y,x,y)
            known_bbox_[:, 1:] = known_bboxs[:, :1] + known_bboxs[:, 1:] / 2

            diff = torch.zeros_like(known_bboxs) 
            diff[:, :1] = known_bboxs[:, 1:] / 2 
            diff[:, 1:] = known_bboxs[:, 1:] / 2

            rand_sign = torch.randint_like(known_bboxs, low=0, high=2, dtype=torch.float32) * 2.0 - 1.0 
            rand_part = torch.rand_like(known_bboxs) 
            if opt.use_dn_neg:
                rand_part[negative_idx] += 1.0
            rand_part *= rand_sign
            known_bbox_ = known_bbox_ + torch.mul(rand_part, 
                                                  diff).cuda() * box_noise_scale
            known_bbox_ = known_bbox_.clamp(min=0.0, max=1.0) 
            known_bbox_expand[:, :1] = (known_bbox_[:, :1] + known_bbox_[:, 1:]) / 2 
            known_bbox_expand[:, 1:] = known_bbox_[:, 1:] - known_bbox_[:, :1]

        m = known_labels_expaned.long().to('cuda')
        input_label_embed = label_enc(m).cuda()
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        
        single_pad = int(max(known_num))
        if opt.use_dn_neg:  
            pad_size = int(single_pad * 2 * dn_number)
        else:
            pad_size = int(single_pad * dn_number)
        padding_label = torch.zeros(pad_size, hidden_dim).cuda()
        padding_bbox = torch.zeros(pad_size, 2).cuda()

        input_query_label = padding_label.repeat(batch_size, 1, 1)
        input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([]).to('cuda')
        if len(known_num):
            map_known_indice = torch.cat([torch.tensor(range(num)) for num in known_num])  # [0,1, 0,1,2,3]
            if opt.use_dn_neg:
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(2 * dn_number)]).long() # [0,1, 0,1,2,3, 4,5, 4,5,6,7]
            else:
                map_known_indice = torch.cat([map_known_indice + single_pad * i for i in range(dn_number)]).long()        
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + num_queries * num_patterns
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        if opt.use_dn_neg:
            for i in range(dn_number):
                if i == 0:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                if i == dn_number - 1:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * i * 2] = True
                else:
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), single_pad * 2 * (i + 1):pad_size] = True
                    attn_mask[single_pad * 2 * i:single_pad * 2 * (i + 1), :single_pad * 2 * i] = True
            known_labels[negative_idx] = 1 
        else:
            for i in range(dn_number):
                if i == 0:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                if i == dn_number - 1:
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
                else:
                    attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                    attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

        mask_dict = {
            'known_indice': torch.as_tensor(known_indice).long(),
            'batch_idx': torch.as_tensor(batch_idx).long(),
            'map_known_indice': torch.as_tensor(map_known_indice).long(),
            'known_lbs_bboxes': (known_labels, known_bboxs[positive_idx, :]),
            'know_idx': know_idx,
            'pad_size': pad_size,
            'negative_idx': negative_idx if opt.use_dn_neg else None,
            'positive_idx': positive_idx,
        }

    else:
        input_query_label = None
        input_query_bbox = None
        attn_mask = None
        mask_dict = None

    return input_query_label, input_query_bbox, attn_mask, mask_dict


def dn_post_process(outputs_class, outputs_coord, mask_dict):
    """
    post process of dn after output from the transformer
    put the dn part in the mask_dict
    """
    if mask_dict and mask_dict['pad_size'] > 0:
        output_known_class = outputs_class[:, :, :mask_dict['pad_size'], :]
        output_known_coord = outputs_coord[:, :, :mask_dict['pad_size'], :]
        outputs_class = outputs_class[:, :, mask_dict['pad_size']:, :]
        outputs_coord = outputs_coord[:, :, mask_dict['pad_size']:, :]
        mask_dict['output_known_lbs_bboxes']=(output_known_class, output_known_coord)
    return outputs_class, outputs_coord


def prepare_for_loss(mask_dict):
    """
    prepare dn components to calculate loss
    Args:
        mask_dict: a dict that contains dn information
    """
    output_known_class, output_known_coord = mask_dict['output_known_lbs_bboxes']
    known_labels, known_bboxs = mask_dict['known_lbs_bboxes']
    map_known_indice = mask_dict['map_known_indice']
    positive_idx = mask_dict['positive_idx']

    known_indice = mask_dict['known_indice']

    batch_idx = mask_dict['batch_idx']
    bid = batch_idx[known_indice]
    if len(output_known_class) > 0:
        output_known_class = output_known_class.permute(1, 2, 0, 3)[(bid, map_known_indice)].permute(1, 0, 2)
        output_known_coord = output_known_coord.permute(1, 2, 0, 3)[(bid, map_known_indice)][positive_idx].permute(1, 0, 2) # bbox只算正例的
    num_tgt = known_indice.numel()
    return known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt


def tgt_loss_boxes(src_boxes, tgt_boxes, num_tgt,):
    """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
       targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
       The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
    """
    if len(tgt_boxes) == 0:
        return {
            'tgt_loss_bbox': torch.as_tensor(0.).to('cuda'),
            'tgt_loss_giou': torch.as_tensor(0.).to('cuda'),
        }

    loss_bbox = F.l1_loss(src_boxes, tgt_boxes, reduction='none')

    losses = {}
    losses['tgt_loss_bbox'] = loss_bbox.sum() / num_tgt
    loss_giou = 1 - torch.diag(generalized_temporal_iou(span_cxw_to_xx(src_boxes), span_cxw_to_xx(tgt_boxes)))
    losses['tgt_loss_giou'] = loss_giou.sum() / num_tgt
    return losses


def tgt_loss_labels(src_logits_, tgt_labels_, num_tgt, focal_alpha, log=True):
    """Classification loss (NLL)
    targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
    """
    if len(tgt_labels_) == 0:
        return {
            'tgt_loss_ce': torch.as_tensor(0.).to('cuda'),
            'tgt_class_error': torch.as_tensor(0.).to('cuda'),
        }

    src_logits, tgt_labels= src_logits_.unsqueeze(0), tgt_labels_.unsqueeze(0)

    target_classes_onehot = torch.zeros([src_logits.shape[0], src_logits.shape[1], src_logits.shape[2]],
                                        dtype=src_logits.dtype, layout=src_logits.layout, device=src_logits.device)
    target_classes_onehot.scatter_(2, tgt_labels.unsqueeze(-1), 1)

    target_classes_onehot = target_classes_onehot[:, :, :]
    loss_ce = sigmoid_focal_loss(src_logits, target_classes_onehot, num_tgt, alpha=focal_alpha, gamma=2) * src_logits.shape[1]

    losses = {'tgt_loss_ce': loss_ce}

    losses['tgt_class_error'] = 100 - accuracy(src_logits_, tgt_labels_)[0]
    return losses


def compute_dn_loss(mask_dict, mode, aux_num, focal_alpha):
    """
    compute dn loss in criterion
    Args:
        mask_dict: a dict for dn information
        training: training or inference flag
        aux_num: aux loss number
        focal_alpha:  for focal loss
    """
    losses = {}
    training = (mode == 'train')
    if training and 'output_known_lbs_bboxes' in mask_dict:
        known_labels, known_bboxs, output_known_class, output_known_coord, num_tgt = prepare_for_loss(mask_dict)
        losses.update(tgt_loss_labels(output_known_class[-1], known_labels, num_tgt, focal_alpha))
        losses.update(tgt_loss_boxes(output_known_coord[-1], known_bboxs, num_tgt))
    else:
        losses['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
        losses['tgt_class_error'] = torch.as_tensor(0.).to('cuda')

    if aux_num:
        for i in range(aux_num):
            # dn aux loss
            if training and 'output_known_lbs_bboxes' in mask_dict:
                l_dict = tgt_loss_labels(output_known_class[i], known_labels, num_tgt, focal_alpha)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
                l_dict = tgt_loss_boxes(output_known_coord[i], known_bboxs, num_tgt)
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
            else:
                l_dict = dict()
                l_dict['tgt_loss_bbox'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_class_error'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_giou'] = torch.as_tensor(0.).to('cuda')
                l_dict['tgt_loss_ce'] = torch.as_tensor(0.).to('cuda')
                l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                losses.update(l_dict)
    return losses