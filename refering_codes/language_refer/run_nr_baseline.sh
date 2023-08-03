module load cuda/10.1.243
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3d_cot_ref_paraphrases_num_anchors.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_OUR_GT.csv
# /home/abdelrem/3d_codes/CoT3D_VG/extract_anchors/nr3D_with_TRUE_GT.csv

cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/language_refer
CUDA_VISIBLE_DEVICES=3 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
    --experiment-tag nr_baseline \
    --output-dir-prefix logs/nr \
    --dataset-name nr3d \
    --per-device-train-batch-size 35 \