cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

CUDA_VISIBLE_DEVICES=1 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_pcd_backbone.py \
    --config configs/pcd_classifier.yaml \
    --output_dir ../log/del_pcd_clf_pre