function [Wx,D] = MyUnpairedCCA3_1(C_x ,C_z,f_type,dataset,mods)
%C_x : cell of size 1xN, where N is number of modality. Each cell contains
%the feature matrix in the form nxp. where n is number of data samples in
%that modality and p is number of fetures of that data.

%C_z : cell of size 1xN, where each cell contains label matrix of
%corresponding modality. 

%f_type : determines type of similarity function "dot_product" or "squared_exponent"

%Wx : matrix of eigen vectors , size (sum of features x sum of features)
%D : diagnol matrix of eigen values , size (sum of features x sum of features)
%p_each = matrix of size n_modality x 1 , where each row contains number of
%features in that corresponding modality. 



%f_type = 'squared_exponent'; 

tic;
[~,n_modality] = size(C_x);
if(size(C_x) ~= size(C_z))
    disp("CHECK INPUT");
    return;
end
reg = 0.0001;
n_each=zeros(n_modality,1);
p_each=zeros(n_modality,1);

for i = 1:n_modality
    C_x{1,i} = full(MyNormalization(C_x{1,i}));
    [n_each(i),p_each(i)]=size(C_x{1,i}); 
end

C_all = zeros(sum(p_each),sum(p_each));
C_diag = zeros(sum(p_each),sum(p_each));
start_point_x  = 1;start_point_y = 1;
for i = 1:n_modality%
    disp(["i is = ",i]);
    for j = i:n_modality
        disp([i,j]);
	    F_ij = 2*C_z{1,i}*C_z{1,j}'; 
	    ata = sum((C_z{1,i}.^2)')'; 
	    F_ij = bsxfun(@minus,F_ij,ata); 
	    clear ata; 
    
	    btb = sum((C_z{1,j}.^2)')'; 
	    F_ij = bsxfun(@minus,F_ij',btb)'; 
	    clear btb; 
    
	    F_ij = exp(0.5*F_ij); 
        S_ij = (C_x{1,i}'*F_ij*C_x{1,j})/(n_each(i,1)*n_each(j,1)); 
	    clear F_ij; 
        
        C_all(start_point_x:start_point_x+p_each(i,1)-1, start_point_y:start_point_y+p_each(j,1)-1) = S_ij;
        
	if(i~=j)
         C_all(start_point_y:start_point_y+p_each(j,1)-1,start_point_x:start_point_x+p_each(i,1)-1) = S_ij';
    else
         C_diag(start_point_x:start_point_x+p_each(i,1)-1 , start_point_y:start_point_y+p_each(j,1)-1) = S_ij;
    end 
	clear S_ij; 

        start_point_y = start_point_y + p_each(j,1); 

       

    end

    start_point_x = start_point_x+ p_each(i,1);
    start_point_y = start_point_x;

end 


disp('Start eigen decomposition.'); 
C_all = C_all + reg*eye(sum(p_each),sum(p_each));
C_diag = C_diag + reg*eye(sum(p_each),sum(p_each));
[Wx,D] = eig(double(C_all),double(C_diag));
disp('done eigen decomposition');

       

toc;
%save(sprintf('Wx_D_mods_%s_dataset_%s.mat', mods, dataset), 'Wx','D');

%clear all; 

