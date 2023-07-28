module load cuda/10.1.243

cd /home/mohama0e/3D_Codes/SAT
python -u train_referit3d.py \
    -scannet-file /lustre/scratch/project/k1546/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/mohama0e/3D_Codes/SAT/referit3d/data/nr3d_cot_ref_paraphrases_num_anchors.csv\
    --bert-pretrain-path /home/mohama0e/3D_Codes/SAT/bert-base-uncased\
    --log-dir logs/MVT_nr3d_cot_cross_100%_filtered_anchors=7_CoTLang_drop15%_targetaug50 \
    --model 'mmt_referIt3DNet' \
    --unit-sphere-norm True \
    --encoder-layer-num 3 \
    --decoder-layer-num 4 \
    --decoder-nhead-num 8 \
    --gpu "0" \
    --view_number 1 \
    --label-lang-sup True \
    --anchors "cot" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors True \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --shuffle_objects_percentage 0 \
    --visaug_extracted_obj_path '/home/mohama0e/3D_Codes/SAT/referit3d/data/nr3d.csv' \
    --visaug_pc_augment False \
    --train_data_percent 0.1 \
    --max_num_anchors 7 \
    --dropout-rate 0.15 \
    --textaug_paraphrase_percentage 0 \
    --target_aug_percentage 0 \
    --gaussian_latent False \
    --distractor_aux_loss_flag False \
    --train_data_repeatation 1 \
    --patience 100 \
    --max-train-epochs 100\
    --init-lr 1e-4\
    --batch-size 16\
    --transformer --model mmt_referIt3DNet\
    --n-workers 2\
    --gpu 0\
    --unit-sphere-norm True\
    --feat2d clsvecROI\
    --context_2d unaligned\
    --mmt_mask train2d\
    --warmup

# python train_referit3d.py\
#  --patience 100\
#  --max-train-epochs 100\
#  --init-lr 1e-4\
#  --batch-size 16\
#  --transformer\
#  --model mmt_referIt3DNet\
#  -scannet-file /lustre/scratch/project/k1546/keep_all_points_00_view_with_global_scan_alignment.pkl \
#  -referit3D-file /home/mohama0e/3D_Codes/sat_org/SAT/referit3d/data/nr3d_cot_ref_paraphrases_num_anchors.csv\
#  --log-dir logs/MVT_nr3d_cot_cross_100%_filtered_anchors=7_CoTLang_drop15%_targetaug50 \
#   --n-workers 2 --gpu 0 --unit-sphere-norm True --feat2d clsvecROI --context_2d unaligned --mmt_mask train2d --warmup