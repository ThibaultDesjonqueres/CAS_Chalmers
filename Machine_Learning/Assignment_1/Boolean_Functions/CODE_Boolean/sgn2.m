function res = sgn2(arg)
    mat=[];
    for i = 1:length(arg)
        if arg(i) < 0
            mat(end+1) = -1 ;
        
        elseif arg(i) >= 0 
            mat(end+1) = 1;
        end
    end
    res = mat;
end