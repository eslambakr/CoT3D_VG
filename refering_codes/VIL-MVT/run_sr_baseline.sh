module load cuda/10.1.243

cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/VIL-MVT
/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_referit3d.py \
    -scannet-file /home/abdelrem/3d_codes/scannet_dataset/scannet/scan_4_sr3d_org/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl \
    -referit3D-file /home/abdelrem/3d_codes/scannet_dataset/scannet/sr3d.csv \
    --bert-pretrain-path /home/abdelrem/3d_codes/MVT-3DVG/weights/bert-base-uncased/ \
    --log-dir logs/sr/student_vil_sr3d_bs24_cot_data100%_ClsLossPost \
    --n-workers 16 \
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
    --anchors "cot" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors True \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/sr3d/' \
    --visaug_pc_augment False \
    --train_data_percent 1.0 \
    --max_num_anchors 1 \
    --vil_flag True \
    --train_objcls_alone_flag False \
    --freezed_pointnet_weights 'none' \
    --obj_cls_post True \
    --dist_type "teacher_student" \
    --cat2glove "/home/abdelrem/3d_codes/vil3d_preprocessed_data/annotations/meta_data/cat2glove42b.json" \
    --category_file "/home/abdelrem/3d_codes/vil3d_preprocessed_data/annotations/meta_data/scannetv2_raw_categories.json" \
    --distill_cross_attns 1 \
    --distill_self_attns 1 \
    --distill_hiddens 0.02 \
    --teacher_weights "logs/sr/teacher_vil_sr3d_bs24_cot_data100%_ClsLossPost/07-30-2023-16-26-54/checkpoints/best_model.pth"
    #--freezed_pointnet_weights 'logs/sr/train_objcls_alone_lang/07-08-2023-15-35-17/checkpoints/best_model.pth'