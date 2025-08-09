dataset = input('Dataset : options : "iapr, "mscoco", "mir" ? ', 's'); 
switch dataset
case "iapr"
run create_iapr.m;  ld = 30;  
mods = input('Enter number of modality to work on 2 or 4? ');
idx1 = input('Enter first index for modality: 1 : IMAGE, 2:ENG, 3:GER, 4:SPANISH ');
idx2 = input('Enter second index for modality: 1 : IMAGE, 2:ENG, 3:GER, 4:SPANISH ');
if mods ==2
    Cx = {I_x_tr{1,idx1}, I_x_tr{1,idx2}};  Cz = {I_z_tr{1,idx1}, I_z_tr{1,idx2}};
else
    Cx = {I_x_tr{1,1}, I_x_tr{1,2}, I_x_tr{1,3}, I_x_tr{1,4}};
    Cz = {I_z_tr{1,1}, I_z_tr{1,2}, I_z_tr{1,3}, I_z_tr{1,4}};
end
Xte1 = I_x_te{1,idx1}; Xte2 = I_x_te{1,idx2};
Zte1 = I_z_te{1,idx1}; Zte2 = I_z_te{1,idx2};
A = MyNorm_new(I_x_te{1, idx1}, I_x_tr{1,idx1});
B = MyNorm_new(I_x_te{1,idx2},I_x_tr{1,idx2});

case "mir"
run create_mir.m;   ld = 30; mods = 2;
idx1 = input('Enter first index for modality: 1 : IMAGE, 2:ENG');
idx2 = input('Enter second index for modality: 1 : IMAGE, 2:ENG');
Cx = {M_x_tr{1,idx1}, M_x_tr{1,idx2}};  Cz = {M_z_tr, M_z_tr};
Xte1 = M_x_te{1,idx1}; Xte2 = M_x_te{1,idx2};
Zte1 = M_z_te; Zte2 = M_z_te;
A = MyNorm_new(M_x_te{1, idx1}, M_x_tr{1,idx1});
B = MyNorm_new(M_x_te{1,idx2}, M_x_tr{1,idx2});

case "mscoco"
run create_coco.m;  ld = 60; mods = 2;
idx1 = input('Enter first index for modality: 1 : IMAGE, 2:ENG');
idx2 = input('Enter second index for modality: 1 : IMAGE, 2:ENG');
Cx = {C_x_tr{1,idx1}, C_x_tr{1,idx2}};  Cz = {C_z_tr{1,idx1}, C_z_tr{1,idx2}};
Xte1 = C_x_te{1,idx1}; Xte2 = C_x_te{1,idx2};
Zte1 = C_z_te{1,idx1}; Zte2 = C_z_te{1,idx2};
A = MyNorm_new(C_x_te{1, idx1}, C_x_tr{1,idx1});
B = MyNorm_new(C_x_te{1,idx2}, C_x_tr{1,idx2});

otherwise
error('Unsupported dataset.');
end

% Train
fprintf('Training FMVMLCCA (%s)\n', dataset);
[Wx, D] = MyUnpairedCCA3(Cx, Cz, "squared_exponent", dataset, mods);
% Retrieve for each k
fprintf('Retrieval\n');
k=50;
if mods == 2
    add_coco_retrieval_b(Wx, D, A, B , Zte1, Zte2, ld, 1, k);
else
    p_each =[2048;300;300;300];
    add_coco_ret_all(Wx, D, p_each, idx1, idx2,A, B, Zte1, Zte2,ld ,1,k);% d.X_te{1,index1}, d.X_te{1,index2}
end









