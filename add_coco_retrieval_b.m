function [I_val_projected_1,T_val_projected_1] = add_coco_retrieval_b(Wx,D,I_val_ResNet101,T_val_Word2Vec_caption,Z_val_1,Z_val_2,ld,D_power,k)
tic;
[a, index] = sort(diag(D),'descend');
D = diag(a);
Wx = Wx(:,index);
D = D^D_power;
I_val_projected = full(I_val_ResNet101)*Wx(1:size(I_val_ResNet101,2),:)*D;
T_val_projected = full(T_val_Word2Vec_caption)*Wx(size(I_val_ResNet101,2)+1: size(I_val_ResNet101,2) + size(T_val_Word2Vec_caption,2),:)*D;
I_val_projected_1 = NormFeat(I_val_projected(:,1:ld));
T_val_projected_1 = NormFeat(T_val_projected(:,1:ld));
mAP12 =0;
mAP21 = 0;
[mAP , mAP21] = coco_common_retrieval_b(I_val_projected_1, T_val_projected_1 ,Z_val_1,Z_val_2,k,ld);

