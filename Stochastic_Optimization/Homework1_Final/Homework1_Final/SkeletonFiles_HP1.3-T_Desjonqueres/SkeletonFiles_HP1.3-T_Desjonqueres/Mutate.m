function mutatedIndividual = Mutate(individual, mutationProbability)

    for i = 1:size(individual,2)  
        r = rand();
        if r < mutationProbability
            if individual(i) == 0
                individual(i) = 1;
            elseif individual(i) == 1
                individual(i) = 0;
            end
        end
    end

mutatedIndividual = individual;
end
