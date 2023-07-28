#!/bin/bash
#SBATCH --job-name=SAT_ORG
#SBATCH -N 1
#SBATCH -o SAT_ORG.out
#SBATCH -e SAT_ORG.err
#SBATCH --mail-user=mohamed.mohamed.2@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
source /home/mohama0e/miniconda3/bin/activate
conda activate sat_env
cd /home/mohama0e/3D_Codes/SAT
python -u train_referit3d.py \
    -scannet-file /lustre/scratch/project/k1546/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/mohama0e/3D_Codes/SAT/referit3d/data/nr3d_cot_ref_paraphrases_num_anchors.csv\
    --bert-pretrain-path /home/mohama0e/3D_Codes/SAT/bert-base-uncased\
    --log-dir logs/SAT_ORG \
    --model 'mmt_referIt3DNet' \
    --gpu "0" \
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