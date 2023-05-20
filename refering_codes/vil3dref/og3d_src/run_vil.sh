cd /home/abdelrem/3d_codes/CoT3D_VG/refering_codes/vil3dref/og3d_src

/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train.py \
    --config configs/nr3d_gtlabel_model.yaml \
    --output_dir ../log/del_vil_teacher_rerpoduce_nr3d

/home/abdelrem/anaconda3/envs/refer3d_cuda/bin/python -u train_mix.py \
    --config configs/nr3d_gtlabelpcd_mix_model.yaml \
    --output_dir ../log/vil_student_rerpoduce_nr3d \
    --resume_files ../log/exprs_neurips22/pcd_clf_pre/ckpts/model_epoch_100.pt \
    ../log/vil_teacher_rerpoduce_nr3d/ckpts/model_epoch_41.pt