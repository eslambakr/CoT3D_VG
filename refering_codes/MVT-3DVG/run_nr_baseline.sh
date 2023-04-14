module load cuda/10.1.243

cd /home/abdelrem/3d_codes/MVT-3DVG/
/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_referit3d.py \
    -scannet-file /home/abdelrem/3d_codes/scannet_dataset/scannet/scan_4_nr3d_org/keep_all_points_00_view_with_global_scan_alignment/keep_all_points_00_view_with_global_scan_alignment.pkl \
    -referit3D-file /home/abdelrem/3d_codes/scannet_dataset/scannet/nr3d.csv \
    --bert-pretrain-path /home/abdelrem/3d_codes/MVT-3DVG/weights/bert-base-uncased/ \
    --log-dir logs/MVT_nr3d_bs128 \
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
    --label-lang-sup True
