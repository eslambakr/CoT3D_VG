#!/bin/bash
#SBATCH --job-name=SAT_COT_SR_10
#SBATCH -N 1
#SBATCH -o SAT_COT_SR_10.out
#SBATCH -e SAT_COT_SR_10.err
#SBATCH --mail-user=mohamed.mohamed.2@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=40:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
source /home/mohama0e/miniconda3/bin/activate
conda activate sat_env
cd /home/mohama0e/3D_Codes/SAT
python -u train_referit3d.py \
    -scannet-file /lustre/scratch/project/k1546/keep_all_points_with_global_scan_alignment.pkl \
    -referit3D-file /home/mohama0e/3D_Codes/SAT/referit3d/data/sr3d.csv \
    --log-dir logs/COT_IN_SAT_SR_10% \
    --model 'mmt_referIt3DNet' \
    --unit-sphere-norm True \
    --encoder-layer-num 3 \
    --decoder-layer-num 4 \
    --decoder-nhead-num 8 \
    --gpu "0" \
    --view_number 4 \
    --rotate_number 1 \
    --label-lang-sup True \
    --anchors "cot" \
    --max-test-objects 52 \
    --cot_type "cross" \
    --predict_lang_anchors True \
    --lang_filter_objs False \
    --visaug_shuffle_mode 'none' \
    --visaug_extracted_obj_path '/home/abdelrem/3d_codes/CoT3D_VG/data/sr3d/' \
    --visaug_pc_augment True \
    --train_data_percent 0.1 \
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