# nohup ./run_nr_baseline.sh  > out.txt 2> out.err < /dev/null &

module load cuda/10.1.243
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_ref_paraphrases_num_anchors.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_OUR_GT.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_TRUE_GT.csv

cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/SAT
/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_referit3d.py \
    -scannet-file /home/abdelrem/3d_codes/scannet_dataset/scannet/scan_4_nr3d_org/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_TRUE_GT.csv \
    --log-dir logs/CoT_GT_nr_40%_510-4_batch16 \
    --patience 100 \
    --max-train-epochs 100 \
    --init-lr 5e-4 \
    --batch-size 16 \
    --transformer \
    --model mmt_referIt3DNet \
    --n-workers 6 \
    --gpu 3 \
    --unit-sphere-norm True \
    --feat2d clsvecROI \
    --context_2d unaligned \
    --mmt_mask train2d \
    --warmup False \
    --anchors "cot" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors True \
    --distractor_aux_loss_flag True \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --shuffle_objects_percentage 0 \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/nr3d/' \
    --visaug_pc_augment False \
    --train_data_percent 0.4 \
    --max_num_anchors 7 \
    --dropout-rate 0.15 \
    --textaug_paraphrase_percentage 0 \
    --target_aug_percentage 0 \
    --gaussian_latent False \
    --train_data_repeatation 1 \
    --anchors_ids_type "GT" \
    --obj_cls_post False \
    --scanrefer False \
    --max-seq-len 24 \
    --feedGTPath False \
    --multicls_multilabel False \
    --remove_repeated_anchors False \
    --include_anchor_distractors False
