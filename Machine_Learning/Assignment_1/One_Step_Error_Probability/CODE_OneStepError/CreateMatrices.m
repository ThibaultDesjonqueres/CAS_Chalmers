function values = CreateMatrices(p, N)
    for i = 1:length(p)
        values = randi([0,1], p(i),N)*2-1;
    end
end