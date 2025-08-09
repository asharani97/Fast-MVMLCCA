function X_out = MyNorm_new(X_1,X_tr)
    [n_1,p_1] = size(X_1);
    %calculating mean accross the columns, which will give mean of values for a
    %particular feature for all the n data samples.
    mean_X_1 =mean(X_tr);
    %variance across rows
    var_X_1 =sqrt(var(X_tr));
    %making the effective mean 0 and deviation 1. Gaussian distribution
    temp_X_1 = (X_1 - repmat(mean_X_1,n_1,1))./repmat(var_X_1,n_1,1);  
    X_out = temp_X_1;
end
