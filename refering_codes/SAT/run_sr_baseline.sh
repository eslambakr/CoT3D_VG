module load cuda/10.1.243

cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/SAT
/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_referit3d.py \
    -scannet-file /home/abdelrem/3d_codes/scannet_dataset/scannet/scan_4_sr3d_org/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl \
    -referit3D-file /home/abdelrem/3d_codes/scannet_dataset/scannet/sr3d.csv \
    --log-dir logs/cot_sr_40%_2d_fixed_reportanchoracc_5lr_0.1lrmmt_warmupFalse_mvtlrschedule \
    --patience 100 \
    --max-train-epochs 100 \
    --init-lr 5e-4 \
    --batch-size 16 \
    --transformer \
    --model mmt_referIt3DNet \
    --n-workers 6 \
    --gpu 2 \
    --unit-sphere-norm True \
    --feat2d clsvecROI \
    --context_2d unaligned \
    --mmt_mask train2d \
    --warmup False \
    --anchors "cot" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors True \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/sr3d/' \
    --visaug_pc_augment False \
    --train_data_percent 0.4 \