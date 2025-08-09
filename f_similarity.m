function [similarity_value] = f_similarity(Z_i_g,Z_j_h,f_type)

if (f_type == "dot_product")
    similarity_value = dot(Z_i_g,Z_j_h)/(norm(Z_i_g)*norm(Z_j_h));
end
if (f_type == "squared_exponent")
    labelsimilaritysigma = 1; %1
    similarity_value = exp(-1*(pdist2(Z_i_g,Z_j_h,'euclidean'))/labelsimilaritysigma);%pdist2 is not squared
end





