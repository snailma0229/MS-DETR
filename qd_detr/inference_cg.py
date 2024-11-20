import pprint
from tqdm import tqdm, trange
import numpy as np
import os
from collections import OrderedDict, defaultdict
from utils.basic_utils import AverageMeter

import torch
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader

from qd_detr.config import TestOptions
from qd_detr.model import build_model
from qd_detr.span_utils import span_cxw_to_xx
from qd_detr.start_end_dataset_cg import StartEndDataset, start_end_collate, prepare_batch_inputs  # NOTE
# from qd_detr.start_end_dataset import StartEndDataset, start_end_collate, prepare_batch_inputs
from qd_detr.postprocessing_qd_detr import PostProcessorDETR
# from standalone_eval.eval import eval_submission # NOTE
from standalone_eval.eval_cg import eval_submission
from utils.basic_utils import save_jsonl, save_json
from utils.temporal_nms import temporal_nms

import logging

logger = logging.getLogger(__name__)
logging.basicConfig(format="%(asctime)s.%(msecs)03d:%(levelname)s:%(name)s - %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S",
                    level=logging.INFO)


def post_processing_mr_nms(mr_res, nms_thd, max_before_nms, max_after_nms):
    mr_res_after_nms = []
    for e in mr_res:
        e["pred_relevant_windows"] = temporal_nms(
            e["pred_relevant_windows"][:max_before_nms],
            nms_thd=nms_thd,
            max_after_nms=max_after_nms
        )
        mr_res_after_nms.append(e)
    return mr_res_after_nms

def get_dn_targets(targets):
    new_targets = []
    target_item = {}
    for i in targets['span_labels']:
        target_item['labels'] = torch.zeros(i['spans'].size(0)).long().cuda()
        target_item['boxes'] = i['spans']
        new_targets.append(target_item)
    return new_targets

def eval_epoch_post_processing(submission, opt, gt_data, save_submission_filename):
    # IOU_THDS = (0.5, 0.7)
    logger.info("Saving/Evaluating before nms results")
    submission_path = os.path.join(opt.results_dir, save_submission_filename)
    save_jsonl(submission, submission_path)
    # 如果是在验证集上评估（测试集没有真实标签）
    if opt.eval_split_name in ["val"]:  # since test_public has no GT
        metrics = eval_submission( # 计算提交结果的指标
            submission, gt_data,
            verbose=opt.debug, match_number=not opt.debug
        )   
        save_metrics_path = submission_path.replace(".jsonl", "_metrics.json")  # 指标保存路径
        save_json(metrics, save_metrics_path, save_pretty=True, sort_keys=False) # 保存指标到json文件
        latest_file_paths = [submission_path, save_metrics_path] # 记录保存的文件路径
    else: 
        metrics = None # 如果不是验证集，则不计算指标
        latest_file_paths = [submission_path, ] # 只记录提交文件的路径

    # 如果设置了NMS阈值（nms_thd），则执行NMS
    if opt.nms_thd != -1:
        logger.info("[MR] Performing nms with nms_thd {}".format(opt.nms_thd)) # 日志输出，开始进行NMS
        submission_after_nms = post_processing_mr_nms( # 对预测结果执行NMS
            submission, nms_thd=opt.nms_thd,
            max_before_nms=opt.max_before_nms, max_after_nms=opt.max_after_nms
        )
        # 日志输出，保存/评估 NMS 后的结果
        logger.info("Saving/Evaluating nms results")
        submission_nms_path = submission_path.replace(".jsonl", "_nms_thd_{}.jsonl".format(opt.nms_thd))
        save_jsonl(submission_after_nms, submission_nms_path) # 保存NMS后的结果到jsonl文件
        if opt.eval_split_name == "val": # 如果是验证集，计算NMS后的指标
            metrics_nms = eval_submission(
                submission_after_nms, gt_data,
                verbose=opt.debug, match_number=not opt.debug
            )
            save_metrics_nms_path = submission_nms_path.replace(".jsonl", "_metrics.json")
            save_json(metrics_nms, save_metrics_nms_path, save_pretty=True, sort_keys=False) # 保存NMS后的指标到json文件
            latest_file_paths += [submission_nms_path, save_metrics_nms_path] # 记录NMS后的文件路径
        else:
            # 如果不是验证集，不计算NMS后的指标
            metrics_nms = None
            latest_file_paths = [submission_nms_path, ]
    else: # 如果没有执行NMS，NMS后的指标为None
        metrics_nms = None
    return metrics, metrics_nms, latest_file_paths # 返回计算的指标、NMS后的指标和所有相关文件的路径 


# for HL
@torch.no_grad()
def compute_hl_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    mode = 'eval'
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter)
    write_tb = tb_writer is not None and epoch_i is not None

    mr_res = []

    topk = 5 # top-5 map

    video_ap_collected = []
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0]
        
        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, opt.use_neg_captions, non_blocking=opt.pin_memory, mode=mode)
        if opt.use_dn:
            targets_dn = get_dn_targets(targets)
            dn_args=(targets_dn, opt.scalar, opt.label_noise_scale, opt.box_noise_scale, opt.num_patterns)
            model_inputs['dn_args'] = dn_args
        model_inputs['opt'] = opt
        
        outputs, mask_dict = model(**model_inputs)

        preds = outputs['saliency_scores'].clone().detach()

        for meta, pred in zip(query_meta, preds):
            pred = pred
            label = meta['label'] # raw label

            video_ap = []
            # Follow the UMT code "https://github.com/TencentARC/UMT/blob/main/datasets/tvsum.py"
            
            if opt.dset_name in ["tvsum"]:
                for i in range(20):
                    pred=pred.cpu()
                    cur_pred = pred[:len(label)]
                    inds = torch.argsort(cur_pred, descending=True, dim=-1)

                    # video_id = self.get_video_id(idx)
                    cur_label = torch.Tensor(label)[:, i]
                    cur_label = torch.where(cur_label > cur_label.median(), 1.0, .0)

                    cur_label = cur_label[inds].tolist()[:topk]

                    # if (num_gt := sum(cur_label)) == 0:
                    num_gt = sum(cur_label)
                    if num_gt == 0:
                        video_ap.append(0)
                        continue

                    hits = ap = rec = 0
                    prc = 1

                    for j, gt in enumerate(cur_label):
                        hits += gt

                        _rec = hits / num_gt
                        _prc = hits / (j + 1)

                        ap += (_rec - rec) * (prc + _prc) / 2
                        rec, prc = _rec, _prc

                    video_ap.append(ap)
            
            elif opt.dset_name in ["youtube_uni"]:
                cur_pred = pred[:len(label)]
                # if opt.dset_name == "tvsum_sfc":
                cur_pred = cur_pred.cpu()
                inds = torch.argsort(cur_pred, descending=True, dim=-1)


                cur_label = torch.Tensor(label).squeeze()[inds].tolist()
                
                num_gt = sum(cur_label)
                if num_gt == 0:
                    video_ap.append(0)
                    continue

                hits = ap = rec = 0
                prc = 1

                for j, gt in enumerate(cur_label):
                    hits += gt

                    _rec = hits / num_gt
                    _prc = hits / (j + 1)

                    ap += (_rec - rec) * (prc + _prc) / 2
                    rec, prc = _rec, _prc
                
                video_ap.append(float(ap))
            else:
                print("No such dataset")
                exit(-1)
                    
            video_ap_collected.append(video_ap)

    mean_ap = np.mean(video_ap_collected)
    submmission = dict(mAP=round(mean_ap, 5))
    

    # tensorboard writer
    if write_tb and criterion:
        for k, v in loss_meters.items():
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1)

    return submmission, loss_meters 


@torch.no_grad() # 计算模型在评估数据集上的性能，并在需要时计算损失
def compute_mr_results(model, eval_loader, opt, epoch_i=None, criterion=None, tb_writer=None):
    model.eval()
    mode = 'eval'
    if criterion:
        assert eval_loader.dataset.load_labels
        criterion.eval()

    loss_meters = defaultdict(AverageMeter) # 用于跟踪各种损失的平均值
    write_tb = tb_writer is not None and epoch_i is not None # 确定是否需要写入TensorBoard

    mr_res = [] # 用于收集模型预测结果的列表
    for batch in tqdm(eval_loader, desc="compute st ed scores"):
        query_meta = batch[0] # 元数据

        model_inputs, targets = prepare_batch_inputs(batch[1], opt.device, opt.use_neg_captions, non_blocking=opt.pin_memory, mode=mode)
        if opt.use_dn:
            targets_dn = get_dn_targets(targets)
            dn_args=(targets_dn, opt.scalar, opt.label_noise_scale, opt.box_noise_scale, opt.num_patterns)
            model_inputs['dn_args'] = dn_args
    
        model_inputs['opt'] = opt
        
        outputs, mask_dict = model(**model_inputs)
        prob = F.softmax(outputs["pred_logits"], -1) # 对预测的逻辑值应用softmax，获得概率 # (batch_size, #queries, #classes=2)
        if opt.span_loss_type == "l1":
            scores = prob[..., 0]   # 如果损失类型是l1，直接使用前景概率作为得分 # * (batch_size, #queries)  foreground label is 0, we directly take it
            pred_spans = outputs["pred_spans"]  # (bsz, #queries, 2) # 获取预测的时间跨度
            _saliency_scores = outputs["saliency_scores"].half()  # (bsz, L) # 获取显著性得分，并转换为半精度以节省内存
            saliency_scores = [] # 用于存储处理后的显著性得分
            valid_vid_lengths = model_inputs["src_vid_mask"].sum(1).cpu().tolist() # 计算每个视频有效长度
            for j in range(len(valid_vid_lengths)):
                saliency_scores.append(_saliency_scores[j, :int(valid_vid_lengths[j])].tolist())
        else:
            bsz, n_queries = outputs["pred_spans"].shape[:2]  # # (bsz, #queries, max_v_l *2)
            pred_spans_logits = outputs["pred_spans"].view(bsz, n_queries, 2, opt.max_v_l)
            pred_span_scores, pred_spans = F.softmax(pred_spans_logits, dim=-1).max(-1)  # 2 * (bsz, #queries, 2)
            scores = torch.prod(pred_span_scores, 2)  # (bsz, #queries)
            pred_spans[:, 1] += 1
            pred_spans *= opt.clip_length

        # compose predictions # 根据模型预测和元数据，构造预测结果
        for idx, (meta, spans, score) in enumerate(zip(query_meta, pred_spans.cpu(), scores.cpu())):
            if opt.span_loss_type == "l1":
                spans = span_cxw_to_xx(spans) * meta["duration"]  # 将预测的中心宽度格式转换为起始终止格式，并缩放到视频持续时间
                spans = torch.clamp(spans, 0, meta["duration"]) # 将预测的时间跨度限制在视频持续时间内
            # # (#queries, 3), [st(float), ed(float), score(float)]
            cur_ranked_preds = torch.cat([spans, score[:, None]], dim=1).tolist() # 将预测的跨度和得分合并为列表
            if not opt.no_sort_results: # 如果需要对结果进行排序
                cur_ranked_preds = sorted(cur_ranked_preds, key=lambda x: x[2], reverse=True) # 根据得分降序排序预测
            cur_ranked_preds = [[float(f"{e:.4f}") for e in row] for row in cur_ranked_preds] # 格式化预测的数字，保留4位小数
            cur_query_pred = dict(
                qid=meta["qid"],
                query=meta["query"],
                vid=meta["vid"],
                pred_relevant_windows=cur_ranked_preds,
                pred_saliency_scores=saliency_scores[idx]
            )
            mr_res.append(cur_query_pred)

        if criterion:
            loss_dict = criterion(outputs, targets, mask_dict, opt, mode)
            weight_dict = criterion.weight_dict
            losses = sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)
            loss_dict["loss_overall"] = float(losses)  # for logging only
            for k, v in loss_dict.items():
                loss_meters[k].update(float(v) * weight_dict[k] if k in weight_dict else float(v))

        if opt.debug:
            break

    if write_tb and criterion: # 如果需要写入TensorBoard且提供了损失函数
        for k, v in loss_meters.items(): # 遍历损失计数器
            tb_writer.add_scalar("Eval/{}".format(k), v.avg, epoch_i + 1) # 将损失的平均值写入TensorBoard
    
    if opt.dset_name in ['hl']: # 根据数据集名称和特征维度选择合适的后处理器
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
            min_w_l=2, max_w_l=150, move_window_method="left",
            process_func_names=("clip_ts", "round_multiple")
        )
    elif opt.dset_name in ['charadesSTA']:
        if opt.v_feat_dim == 4096: # vgg
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=360,
                min_w_l=12, max_w_l=360, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
        else:
            post_processor = PostProcessorDETR(
                clip_length=opt.clip_length, min_ts_val=0, max_ts_val=150,
                min_w_l=2, max_w_l=60, move_window_method="left",
                process_func_names=("clip_ts", "round_multiple")
            )
    else:
        post_processor = PostProcessorDETR(
            clip_length=opt.clip_length, min_ts_val=0, max_ts_val=50000,
            min_w_l=0, max_w_l=50000, move_window_method="left",
            process_func_names=(["round_multiple"])
        )

    mr_res = post_processor(mr_res)
    # return mr_res
    return mr_res, loss_meters

def get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer):
    """compute and save query and video proposal embeddings"""
    eval_res, eval_loss_meters = compute_mr_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)  # list(dict)
    # return eval_res
    return eval_res, eval_loss_meters


def eval_epoch(model, eval_dataset, opt, save_submission_filename, epoch_i=None, criterion=None, tb_writer=None):
    logger.info("Generate submissions")
    model.eval()

    if criterion is not None and eval_dataset.load_labels:
        criterion.eval()
    else:
        criterion = None

    if opt.dset_name == 'tacos':
        shuffle = True
    else:
        shuffle = False

    eval_loader = DataLoader(
        eval_dataset,
        collate_fn=start_end_collate,
        batch_size=opt.eval_bsz,
        num_workers=opt.num_workers,
        shuffle=shuffle,
        pin_memory=opt.pin_memory
    )


    # tvsum 
    if opt.dset_name in ['tvsum', 'youtube_uni']:
        metrics, eval_loss_meters = compute_hl_results(model, eval_loader, opt, epoch_i, criterion, tb_writer)
        
        # to match original save format
        submission = [
            {"brief": metrics}
        ]
        submission_path = os.path.join(opt.results_dir, "latest_metric.jsonl")
        save_jsonl(submission, submission_path)

        return submission[0], submission[0], eval_loss_meters, [submission_path]

    else:
        submission, eval_loss_meters = get_eval_res(model, eval_loader, opt, epoch_i, criterion, tb_writer)

        if opt.dset_name in ['charadesSTA', 'tacos', 'nlq']:
            new_submission = []
            for s in submission:
                s.pop('pred_saliency_scores', None)
                new_submission.append(s)
            submission = new_submission

        if opt.no_sort_results:
            save_submission_filename = save_submission_filename.replace(".jsonl", "_unsorted.jsonl")
        metrics, metrics_nms, latest_file_paths = eval_epoch_post_processing(
            submission, opt, eval_dataset.data, save_submission_filename)
        # return metrics, metrics_nms, latest_file_paths
        return metrics, metrics_nms, eval_loss_meters, latest_file_paths


def setup_model(opt):
    """setup model/optimizer/scheduler and load checkpoints when needed"""
    logger.info("setup model/optimizer/scheduler")
    model, criterion = build_model(opt)
    if opt.device.type == "cuda":
        logger.info("CUDA enabled.")
        model.to(opt.device)
        criterion.to(opt.device)

    param_dicts = [{"params": [p for n, p in model.named_parameters() if p.requires_grad]}]
    optimizer = torch.optim.AdamW(param_dicts, lr=opt.lr, weight_decay=opt.wd)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, opt.lr_drop)

    if opt.resume is not None:
        logger.info(f"Load checkpoint from {opt.resume}")
        checkpoint = torch.load(opt.resume, map_location="cpu")
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        if 'pt' in opt.resume[:-4]:
            if 'asr' in opt.resume[:25]:
                model.load_state_dict(checkpoint["model"])
            else:
                for k, v in checkpoint["model"].items():
                    name = k[7:]  # remove `module.`
                    new_state_dict[name] = v
                # model.load_state_dict(checkpoint["model"])
                model.load_state_dict(new_state_dict)
        else:
            model.load_state_dict(checkpoint["model"])
        if opt.resume_all:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            opt.start_epoch = checkpoint['epoch'] + 1
        logger.info(f"Loaded model saved at epoch {checkpoint['epoch']} from checkpoint: {opt.resume}")
    else:
        logger.warning("If you intend to evaluate the model, please specify --resume with ckpt path")

    return model, criterion, optimizer, lr_scheduler


def start_inference(train_opt=None, split=None, splitfile=None):
    if train_opt is not None:
        opt = TestOptions().parse(train_opt.a_feat_dir)
    else:
        opt = TestOptions().parse()
    if split is not None:
        opt.eval_split_name = split
    if splitfile is not None:
        opt.eval_path = splitfile

    print(opt.eval_split_name)
    print(opt.eval_path)
    logger.info("Setup config, data and model...")


    cudnn.benchmark = True
    cudnn.deterministic = False

    assert opt.eval_path is not None
    if opt.eval_split_name == 'val':
        loadlabel = True
    else:
        loadlabel = False

    eval_dataset = StartEndDataset(
        dset_name=opt.dset_name,
        data_path=opt.eval_path,
        v_feat_dirs=opt.v_feat_dirs,
        q_feat_dir=opt.t_feat_dir,
        q_feat_type="last_hidden_state",
        max_q_l=opt.max_q_l,
        max_v_l=opt.max_v_l,
        ctx_mode=opt.ctx_mode,
        data_ratio=opt.data_ratio,
        normalize_v=not opt.no_norm_vfeat,
        normalize_t=not opt.no_norm_tfeat,
        clip_len=opt.clip_length,
        max_windows=opt.max_windows,
        load_labels=loadlabel,  # opt.eval_split_name == "val",
        span_loss_type=opt.span_loss_type,
        txt_drop_ratio=0,
        dset_domain=opt.dset_domain,
        mode='val',
        use_neg_captions = opt.use_neg_captions
    )

    model, criterion, _, _ = setup_model(opt)

    save_submission_filename = "hl_{}_submission.jsonl".format(
        opt.eval_split_name)
    logger.info("Starting inference...")
    with torch.no_grad():
        metrics_no_nms, metrics_nms, eval_loss_meters, latest_file_paths = \
            eval_epoch(model, eval_dataset, opt, save_submission_filename, criterion=criterion)
    if opt.eval_split_name == 'val':
        logger.info("metrics_no_nms {}".format(pprint.pformat(metrics_no_nms["brief"], indent=4)))
    if metrics_nms is not None:
        logger.info("metrics_nms {}".format(pprint.pformat(metrics_nms["brief"], indent=4)))

from sys import argv
if __name__ == '__main__':
    _,_,_,_,split,_,splitfile = argv

    start_inference(split=split, splitfile=splitfile)
