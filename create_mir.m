original_dir = pwd;
cd '~/MVMLCCA/MVMLCCA_Features/MirFlickr_2007/';
%Into Raw Feature Folder
cd 'Raw Features'/;
%Loading features for image and captions
disp("Starting loading data for Mir Flickr 2007 dataset");
img = load("MirFlickr_2007_Image_ResNet101.mat");
tag = load("MirFlickr_2007_Tag_fastText_mix.mat");
c = load("MirFlickr_2007_Labels.mat");
M_z_tr = c.M_z_tr;
M_z_te = c.M_z_te;
M_x_tr = {img.M_tr_ResNet101, tag.M_tr_Tag_fastText_mix};
M_x_te = {img.M_te_ResNet101, tag.M_te_Tag_fastText_mix};
disp("Data loading for Mir Flickr 2007")
clear img tag c;
cd(original_dir);
