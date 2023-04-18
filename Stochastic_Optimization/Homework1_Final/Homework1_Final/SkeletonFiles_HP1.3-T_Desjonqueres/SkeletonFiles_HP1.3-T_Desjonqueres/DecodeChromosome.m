% Note: Each component of x should take values in [-a,a], where a = maximumVariableValue.

function x = DecodeChromosome(chromosome,numberOfVariables,maximumVariableValue);
    m = length(chromosome);  % m = Chromosome Length
    n = numberOfVariables;   
    k = m/n;

    Mat = [];

    if k >= 1
        MatrixGenes = reshape(chromosome,[k,n])'; % Reshaping the 
                                                  % chromosome into a matrix
                                                  % where each row is a
                                                  % slice
    elseif k == 0
        disp("ERROR k==0 in DecodeChromosome.m, choose numberOfVariables < N ");
        return
    end

    for i = 1:n
        for j = 1:k
            Mat(i,j) =  2^(-j)*MatrixGenes(i,j);
        end
    end

    x = [sum(Mat,2)]; %Sum of all the matrix element along the second dimension
    X = [];
    for i = 1:size(x,1)
        X(i) = -maximumVariableValue + ((2*maximumVariableValue)/(1-2^(-k)))*x(i); 
                                                 % Adatping the chromosome
                                                 % to our desired range
    end
    x = X;
end



