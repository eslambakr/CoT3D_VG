cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

#CUDA_VISIBLE_DEVICES=1 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
#    --config configs/nr3d_gtlabel_model.yaml \
#    --output_dir ../log/vil_teacher_cot_nr3d_100_100epoch_tgtaug0_7anchor_w5_noobj_lr0.1

CUDA_VISIBLE_DEVICES=2 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
    --config configs/nr3d_gtlabelpcd_mix_model.yaml \
    --output_dir ../log/vil_student_cot_nr3d_100 \
    --resume_files ../log/pcd_clf_pre/ckpts/model_epoch_99.pt \
    ../log/vil_teacher_cot_nr3d_100_100epoch_tgtaug0_7anchor_w5_noobj_lr0.1/ckpts/model_epoch_84.pt