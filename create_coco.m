original_dir = pwd;
cd '~/MVMLCCA/MVMLCCA_Features/MS-COCO/';
%Into Raw Feature Folder
cd 'Raw Features'/;
%Loading features for image and captions
disp("Starting loading data for MS-COCO dataset");
val = load("COCO_val_Resnet101_Word2Vec.mat");
C_x_te = {val.I_val_ResNet101, val.T_val_Word2Vec_caption};
disp("Data loading for test label only");
C_z_te = {val.Z_val, val.Z_val};
clear val;
z = load("COCO_Z_tr_val_Word2Vec.mat");
z_tr_300 = z.Z_tr_coco_vec;
z_te_300 = z.Z_val_coco_vec;
cd ../Cells/;
%Loading data from cell folder
c =load("COCO_all_Resnet101_Word2Vec.mat");
C_x_tr = {c.I_tr_ResNet101, c.T_tr_Word2Vec_caption};
C_z_tr = {c.Z_tr,c.Z_tr};
disp("Data loading for MS-COCO over");
cd(original_dir);
