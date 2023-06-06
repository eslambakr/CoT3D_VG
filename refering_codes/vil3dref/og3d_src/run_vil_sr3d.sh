cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

#CUDA_VISIBLE_DEVICES=3 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
#    --config configs/sr3d_gtlabel_model.yaml \
#    --output_dir ../log/vil_teacher_sr3d_100_25epoch

CUDA_VISIBLE_DEVICES=3 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
    --config configs/sr3d_gtlabelpcd_mix_model.yaml \
    --output_dir ../log/vil_student_cot_sr3d_10_50epoch_lr0.1_w1_repeat6 \
    --resume_files ../log/pcd_clf_pre/ckpts/model_epoch_99.pt \
    ../log/vil_teacher_cot_sr3d_10_25epoch_lr0.1_w1/ckpts/model_epoch_25.pt