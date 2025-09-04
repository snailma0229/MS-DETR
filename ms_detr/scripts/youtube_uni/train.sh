dset_name=youtube_uni
ctx_mode=video_tef
v_feat_types=slowfast_clip
t_feat_type=clip 
results_root=results_youtubeuni
exp_id=exp


######## data paths
train_path=data/youtube_uni/youtube_train.jsonl
eval_path=data/youtube_uni/youtube_valid.jsonl
eval_split_name=val

######## setup video+text features
# feat_root=../features/tvsum
feat_root=features/youtube_uni

# # video features
v_feat_dim=2816
v_feat_dirs=()
v_feat_dirs+=(${feat_root}/vid_clip)
v_feat_dirs+=(${feat_root}/vid_slowfast)

# # text features
t_feat_dir=${feat_root}/txt_clip/ # maybe not used
t_feat_dim=512


#### training
bsz=4
lr=2e-4

 
for seed in 0 1 2 3 2018
do 
    for dset_domain in dog gymnastics parkour skating skiing surfing
    do
        PYTHONPATH=$PYTHONPATH:. python ms_detr/train.py \
        --dset_name ${dset_name} \
        --ctx_mode ${ctx_mode} \
        --train_path ${train_path} \
        --eval_path ${eval_path} \
        --eval_split_name ${eval_split_name} \
        --v_feat_dirs ${v_feat_dirs[@]} \
        --v_feat_dim ${v_feat_dim} \
        --t_feat_dir ${t_feat_dir} \
        --t_feat_dim ${t_feat_dim} \
        --bsz ${bsz} \
        --results_root ${results_root}_${dset_domain} \
        --exp_id ${exp_id} \
        --max_v_l -1 \
        --n_epoch 2000 \
        --lr_drop 2000 \
        --max_es_cnt -1 \
        --dset_domain ${dset_domain} \
        --seed $seed \
        --lr ${lr} \
        ${@:1}
    done
done


