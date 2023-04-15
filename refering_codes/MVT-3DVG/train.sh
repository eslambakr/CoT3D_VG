#!/bin/bash
#SBATCH --job-name=run_v_train_mvt_lang_test
#SBATCH -N 1
#SBATCH -o /ibex/scratch/ahmems0a/mvt/run_v_train_mvt_lang_test.out
#SBATCH -e /ibex/scratch/ahmems0a/mvt/run_v_train_mvt_lang_test.err
#SBATCH --mail-type=ALL
#SBATCH --time=1:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:v100:1

source /home/ahmems0a/miniconda3/bin/activate
conda activate refer

# cd /home/ahmems0a/repos/MVT-3DVG/referit3d/external_tools/pointnet2
# python setup.py install

python train_referit3d.py \
    -scannet-file /home/ahmems0a/repos/MVT-3DVG/data/keep_all_points_with_global_scan_alignment/keep_all_points_with_global_scan_alignment.pkl \
    -referit3D-file /home/ahmems0a/repos/MVT-3DVG/data/sr3d.csv \
    --log-dir logs/MVT_nr3d \
    --n-workers 4 \
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
    --predict-lang-anchors True \
    --anchors "none"

