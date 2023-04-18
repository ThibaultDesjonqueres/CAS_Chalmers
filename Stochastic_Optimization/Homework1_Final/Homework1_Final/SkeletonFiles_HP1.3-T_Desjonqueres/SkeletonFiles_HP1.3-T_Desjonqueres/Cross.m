function newIndividuals = Cross(individual1, individual2)
    r=randi([1,size(individual1,2)]);
    if r == size(individual1,2)
        r = r-1;
    end
    zeros1 = zeros(1,size(individual1,2));
    zeros2 = zeros(1,size(individual1,2));

    section1 = individual1(1:r);
    section2 = individual2(1:r);

    individual1 = individual1(r+1:end);
    individual2 = individual2(r+1:end);

    zeros1(1:r) = section2;
    zeros1(r+1:end) = individual1;

    zeros2(1:r) = section1;
    zeros2(r+1:end) = individual2;

    newIndividuals = [[zeros1]
                      [zeros2]];

end
