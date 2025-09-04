#!/usr/bin/env bash
set -euo pipefail

# ===== Basic configuration =====
dset_name=hl
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip
results_root=results
exp_id=exp

# ===== Data paths =====
train_path_default="xxx/data/highlight_train_release_captions_iou025_normalize_modify.jsonl"
train_path="${1:-$train_path_default}"

eval_path="xxx/data/highlight_val_release.jsonl"
eval_split_name=val

# ===== Feature root path =====
feat_root="xxx/features"

# ===== Video/Text feature settings =====
v_feat_dim=0
v_feat_dirs=()

# video features
if [[ ${v_feat_types} == *"slowfast"* ]]; then
  v_feat_dirs+=("${feat_root}/slowfast_features")
  (( v_feat_dim += 2304 ))
fi
if [[ ${v_feat_types} == *"clip"* ]]; then
  v_feat_dirs+=("${feat_root}/clip_features")
  (( v_feat_dim += 512 ))
fi

# text features
if [[ ${t_feat_type} == "clip" ]]; then
  t_feat_dir="${feat_root}/clip_text_features/"
  t_feat_dim=512
else
  echo "Wrong arg for t_feat_type: ${t_feat_type}"
  exit 1
fi

# ===== Start training =====
PYTHONPATH="${PYTHONPATH:-}:." python ms_detr/train.py \
  --dset_name "${dset_name}" \
  --train_path "${train_path}" \
  --eval_path "${eval_path}" \
  --eval_split_name "${eval_split_name}" \
  --v_feat_dirs "${v_feat_dirs[@]}" \
  --v_feat_dim "${v_feat_dim}" \
  --t_feat_dir "${t_feat_dir}" \
  --t_feat_dim "${t_feat_dim}" \
  --results_root "${results_root}" \
  --exp_id "${exp_id}" \
  --use_dn \
  --use_tgt \
  --use_sf \
  --use_decoder_neg \
  --interm_neg \
  --use_interm \
  --eval_epoch 1 \
  --num_queries 20 \
  "${@:2}"