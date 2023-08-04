cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

CUDA_VISIBLE_DEVICES=0 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
    --config configs/nr3d_gtlabel_model.yaml \
    --output_dir ../log/teacher/GT_vil_teacher_nr3d_cot_PredLang_100%_100epoch_7anchor_w1_distLoss

#CUDA_VISIBLE_DEVICES=1 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
#    --config configs/nr3d_gtlabelpcd_mix_model.yaml \
#    --output_dir ../log/student/vil_student_nr3d_cot_PredLang_40%_100epoch_7anchor_w1_b24_distLoss \
#    --resume_files ../log/pcd_clf_pre/ckpts/model_epoch_99.pt \
#    ../log/teacher/vil_teacher_nr3d_cot_PredLang_100%_100epoch_7anchor_w1_distLoss/ckpts/model_epoch_65.pt

#    ../log/teacher/vil_teacher_nr3d_cot_PredLang_100%_100epoch_7anchor_w1/ckpts/model_epoch_94.pt
#    ../log/vil_teacher_cot_nr3d_100_100epoch_tgtaug0_7anchor_w5_noobj_lr0.1/ckpts/model_epoch_84.pt