#!/bin/bash
#SBATCH --job-name=cot_nr_100%__7anchors_batch64_GT
#SBATCH -N 1
#SBATCH -o cot_nr_100%__7anchors_batch64_GT.out
#SBATCH -e cot_nr_100%__7anchors_batch64_GT.err
#SBATCH --mail-user=mohamed.mohamed.2@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=100:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:v100:1
conda activate sat_env
module load cuda/10.1.243

cd /home/ahmems0a/CoT3D_VG/refering_codes/SAT
python train_referit3d.py \
    -scannet-file /lustre/scratch/project/k1546/scannet/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/ahmems0a/CoT3D_VG/refering_codes/SAT/referit3d/data/nr3D_with_TRUE_GT.csv \
    --log-dir logs/cot_nr_100%__7anchors_batch64_GT \
    --patience 100 \
    --max-train-epochs 100 \
    --init-lr 5e-4 \
    --batch-size 64 \
    --transformer \
    --model mmt_referIt3DNet \
    --n-workers 6 \
    --gpu 0 \
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
    --shuffle_objects_percentage 0 \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/nr3d/' \
    --visaug_pc_augment False \
    --train_data_percent 1.0 \
    --max_num_anchors 7 \
    --dropout-rate 0.15 \
    --textaug_paraphrase_percentage 0 \
    --target_aug_percentage 0 \
    --gaussian_latent False \
    --distractor_aux_loss_flag True \
    --train_data_repeatation 1\
    --scanrefer False \
    --feedGTPath False \
    --multicls_multilabel False \
    --include_anchor_distractors False\
    --anchors_ids_type "GT" 