function [mAP , mAP21] = coco_common_retrieval(I_val_projected , T_val_projected,Z_1_test,Z_2_test,k, ld)
st = tic;
total_columns = size(Z_1_test, 2);
    if k == 0
        k = total_columns;
    end
I_val_projected_norm = NormFeat(I_val_projected);%I_val_projected;%
T_val_projected_norm = NormFeat(T_val_projected);%T_val_projected;%NormFeat(T_val_projected);

X_1_test = I_val_projected_norm(:,1:ld);
X_2_test = T_val_projected_norm(:,1:ld);

[n_1_t,p_1_t] = size(X_1_test);
[n_2_t,p_2_t] = size(X_2_test);

result_matrix = -1*ones(n_1_t,k);
index_matrix = zeros(n_1_t,k);
result_matrix2 = -1*ones(k,n_2_t);
index_matrix2 = zeros(k,n_2_t);

countk = 0;
for i = 1:n_1_t
    for j = 1 :n_2_t
        features_similarity = exp(-1*(norm(X_1_test(i,:)-X_2_test(j,:))^2));%f_similarity(X_1_test(i,:),X_2_test(j,:),"squared_exponent");
        if(features_similarity>result_matrix(i,k))
            result_matrix(i,k) = features_similarity;
            [sorted,indexi] = sort(result_matrix(i,:),'descend');
            result_matrix(i,:) = sorted;
            
            index_matrix(i,k) = j ;
            tempi = index_matrix(i,indexi);
            index_matrix(i,:) = tempi;
        end
        
        if(features_similarity>result_matrix2(k,j))
           
            result_matrix2(k,j) = features_similarity;
            [sorted,indexi] = sort(result_matrix2(:,j),'descend');
            result_matrix2(:,j) = sorted;
            
            index_matrix2(k,j) = i ;
            tempi = index_matrix2(indexi,j);
            index_matrix2(:,j) = tempi;
        end
    end
end


precision_all = zeros(n_1_t,k);
avg_precision_all = zeros(n_1_t,1);
mAP = 0 ;

precision_all21 = zeros(n_2_t,n_1_t);
avg_precision_all21 = zeros(n_2_t,1);
mAP21 = 0 ;


for i = 1:n_1_t  
    temp = 0;
    count = 0;
    for j = 1 :k
        label_similarity = 0;
        if(f_similarity(Z_1_test(i,:),Z_2_test(index_matrix(i,j),:),"dot_product")>0)%(i,1:k)
            label_similarity =1;
        end
        temp = temp + (label_similarity==1);    
        precision_all(i,j)= temp/j;
        if(label_similarity==1)
            avg_precision_all(i) = avg_precision_all(i) + precision_all(i,j);
            count = count + 1;
        end
        
    end
    if(count~=0)
        avg_precision_all(i) = avg_precision_all(i)/count;
    end

end

for i = 1:n_2_t
    temp = 0;
    count = 0;
    for j = 1 :k
        label_similarity = 0;
        if(f_similarity(Z_2_test(i,:),Z_1_test(index_matrix2(j,i),:),"dot_product")>0)
            label_similarity =1;
        end
        temp = temp + (label_similarity==1);    
        precision_all21 (i,j)= temp/j;
        if(label_similarity==1)
            avg_precision_all21(i) = avg_precision_all21(i) + precision_all21(i,j);
            count = count + 1;
        end
    end
    if(count~=0)
        avg_precision_all21(i) = avg_precision_all21(i)/count;
    end
end

mAP = sum(avg_precision_all)/n_1_t;
mAP21 = sum(avg_precision_all21)/n_2_t;
disp(mAP);
disp(mAP21);

toc(st)
