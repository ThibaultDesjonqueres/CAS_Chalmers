function bool = CreateBool(numberOfVariables)

    decimalVector= [0:1:(2^numberOfVariables)-1];
    newChr = [];
    mat = dec2bin(decimalVector)';
    A = size(mat,1);
    B = size(mat,2);
    mat = reshape(mat,1,[]);

    mat = strrep(mat,'0',"-1");
    mat = convertCharsToStrings(mat);
    mat = strrep(mat,'1','1,');
    mat = str2num(mat);
    bool = reshape(mat,[A,B])';
end