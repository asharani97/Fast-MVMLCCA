original_dir = pwd;
cd '~/MVMLCCA/MVMLCCA_Features/IAPRTC-12/';
%Into Raw Feature Folder
cd 'Raw Features'/;
%Loading features for image and captions
disp("Starting loading data for IAPRTC-12 dataset");
img = load("IAPRTC_Images_ResNet101.mat");
eng = load("IAPRTC_Eng_Word2Vec.mat");
ger = load("IAPRTC_Ger_Word2Vec.mat");
spa = load("IAPRTC_Spa_Word2Vec.mat");
z = load("IAPRTC_Z_tr_te_Word2Vec.mat");
z_tr = z.Z_tr_iaprtc_vec;
z_te = z.Z_te_iaprtc_vec;
IZ_300_tr = {z_tr, z_tr, z_tr, z_tr};
IZ_300_te = {z_te, z_te, z_te, z_te};
I_x_tr = {img.I_tr_ResNet101, eng.T_tr_Eng_Word2Vec, ger.T_tr_Ger_Word2Vec, spa.T_tr_Spa_Word2Vec};
I_x_te = {img.I_te_ResNet101, eng.T_te_Eng_Word2Vec, ger.T_te_Ger_Word2Vec, spa.T_te_Spa_Word2Vec};
cd ../Cells/;
%Loading data from cell folder
or_c = load("IAPRTC_all_4_cells.mat");
I_z_tr = or_c.C_z;
I_z_te = or_c.C_z_te;
disp("Data loading for IAPRTC-12 over")
clear img eng spa ger or_c z z_te z_tr;
cd(original_dir);
