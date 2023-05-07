#!/bin/bash
#SBATCH --job-name=run_v_test_coarse_with_shape_prior_1024
#SBATCH -N 1
#SBATCH -o run_v_test_coarse_with_shape_prior_1024.out
#SBATCH -e run_v_test_coarse_with_shape_prior_1024.err
#SBATCH --mail-user=mohamed.mohamed.2@kaust.edu.sa
#SBATCH --mail-type=ALL
#SBATCH --time=24:00:00
#SBATCH --mem=64G
source /home/mohama0e/miniconda3/bin/activate
conda activate cot3d
cd /home/mohama0e/CoT3D_VG/automatic_loc_module
python benchmark_nr3d_opt1.py