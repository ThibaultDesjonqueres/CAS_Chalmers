function res = sgn(arg)
%UNTITLED2 Summary of this function goes here
%   Detailed explanation goes here
    if arg < 0
        res=-1;    
    elseif arg >= 0 
        res=+1;
    end
end