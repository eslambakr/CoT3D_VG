#!/bin/bash
#SBATCH --job-name=cot_sr_10%__1anchors_batch64_GT
#SBATCH -N 1
#SBATCH -o cot_sr_10%__1anchors_batch64_GT.out
#SBATCH -e cot_cot_sr_10%__1anchors_batch64_GT.err
#SBATCH --mail-user=mahmoud.ahmed@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=23:00:00
#SBATCH --mem=15G
#SBATCH --gres=gpu:gtx1080ti:1
module load cuda/10.1.243

cd /home/ahmems0a/CoT3D_VG/refering_codes/SAT
python train_referit3d.py \
    -scannet-file /lustre/scratch/project/k1546/scannet/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/ahmems0a/CoT3D_VG/refering_codes/SAT/referit3d/data/sr3d.csv \
    --log-dir logs/cot_sr_10% \
    --patience 100 \
    --max-train-epochs 100 \
    --init-lr 5e-4 \
    --batch-size 16 \
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
    --visaug_extracted_obj_path '/home/ahmems0a/CoT3D_VG/refering_codes/SAT/referit3d/data/' \
    --visaug_pc_augment False \
    --train_data_percent 0.1 \