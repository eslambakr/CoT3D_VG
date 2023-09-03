module load cuda/10.1.243
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_ref_paraphrases_num_anchors.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_OUR_GT.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_TRUE_GT.csv

cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/MVT-3DVG
/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_referit3d.py \
    -scannet-file /home/abdelrem/3d_codes/scannet_dataset/scannet/scan_4_nr3d_org/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_TRUE_GT.csv \
    --bert-pretrain-path /home/abdelrem/3d_codes/MVT-3DVG/weights/bert-base-uncased/ \
    --log-dir logs/nr/MVT_nr3d_baseline_drop15%_data100% \
    --n-workers 8 \
    --model 'referIt3DNet_transformer' \
    --unit-sphere-norm True \
    --batch-size 24 \
    --encoder-layer-num 3 \
    --decoder-layer-num 4 \
    --decoder-nhead-num 8 \
    --gpu "0" \
    --view_number 4 \
    --rotate_number 4 \
    --label-lang-sup True \
    --anchors "none" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors False \
    --distractor_aux_loss_flag False \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --shuffle_objects_percentage 0 \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/nr3d/' \
    --visaug_pc_augment False \
    --train_data_percent 1.0 \
    --max_num_anchors 3 \
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
    --include_anchor_distractors False \
    --init-lr 0.0005
