cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

#CUDA_VISIBLE_DEVICES=0 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
#    --config configs/sr3d_gtlabel_model.yaml \
#    --output_dir ../log/teacher/vil_teacher_sr3d_cot_predAnchorsLang_100%_50epoch_cotTeacherLoss

CUDA_VISIBLE_DEVICES=0 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
    --config configs/sr3d_gtlabelpcd_mix_model.yaml \
    --output_dir ../log/student/del_vil_student_sr3d_cot_100data_objs80_testobjs80_tgtmul1_100epoch_mvtlr_adam_repeat1_b16_cotTeacherLoss \
    --resume_files ../log/pcd_clf_pre/ckpts/model_epoch_99.pt \
    ../log/teacher/vil_teacher_sr3d_cot_predAnchorsLang_100%_50epoch_cotTeacherLoss/ckpts/model_epoch_42.pt
#    ../log/teacher/vil_teacher_sr3d_cot_predAnchorsLang_100_50epoch_objs80/ckpts/model_epoch_32.pt
#    ../log/vil_teacher_sr3d_100_25epoch/ckpts/model_epoch_22.pt
    

#CUDA_VISIBLE_DEVICES=2 /home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
#    --config configs/sr3d_gtlabelpcd_mix_model.yaml \
#    --output_dir ../log/student/del_vil_student_sr3d_baseline_100data_objs52_testobjs52_txtlen24_tgtmul1_100epoch_mvtlr_adamw_repeat1_ZeroTeacherLoss \
#    --resume_files ../log/pcd_clf_pre/ckpts/model_epoch_99.pt \
#    ../log/teacher/vil_teacher_sr3d_cot_predAnchorsLang_10_50epoch_objs52_testobjs52_txtlen24/ckpts/model_epoch_49.pt