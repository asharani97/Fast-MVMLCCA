function [mAP12,mAP21] = add_coco_retrieval_all(Wx,D, p_each,index_1,index_2,I_val_ResNet101,T_val_Word2Vec_caption,Z_val_1,Z_val_2,ld,D_power,k)
    tic;
    new_Wx = zeros(p_each(index_1,1)+p_each(index_2,1),sum(p_each)); %11 4387 12 2369 all 3239
    new_Wx(1:p_each(index_1,1),:) = Wx(sum(p_each(1:index_1-1)) + 1 : sum(p_each(1:index_1-1))+p_each(index_1,1) ,:);
    new_Wx(p_each(index_1,1)+1 :p_each(index_1,1)+p_each(index_2,1),:) = Wx(sum(p_each(1:index_2-1)) + 1 : sum(p_each(1:index_2-1))+p_each(index_2,1) ,:);
    [n_1_t,p_1_t] = size(I_val_ResNet101);
    [n_2_t,p_2_t] = size(T_val_Word2Vec_caption);
    [a, index] = sort(diag(D),'descend');
    D = diag(a);
    Wx = new_Wx(:,index);
    W_1 = Wx(1:p_1_t,1:ld);%128 x ld %temp = Wx*(D^D_power);%138*d
    W_2 = Wx(p_1_t+1:p_1_t+p_2_t,1:ld);%p_1_t + ld %10 x ld
    D_ld = D(1:ld,1:ld);

    D_1 = D_ld;
    D_2 = D_ld;
    D_1 = D_1^D_power;
    D_2 = D_2^D_power;

    I_val_projected = full(I_val_ResNet101)*W_1*D_1;
    T_val_projected = full(T_val_Word2Vec_caption)*W_2*D_2;
    I_val_projected_1 = NormFeat(I_val_projected(:,1:ld));
    T_val_projected_1 = NormFeat(T_val_projected(:,1:ld));
    toc;
    mAP12 =0;
    mAP21 = 0;
    [mAP , mAP21] = coco_common_retrieval_b(I_val_projected_1, T_val_projected_1 ,Z_val_1, Z_val_2, k, ld);
    toc;
  
    
