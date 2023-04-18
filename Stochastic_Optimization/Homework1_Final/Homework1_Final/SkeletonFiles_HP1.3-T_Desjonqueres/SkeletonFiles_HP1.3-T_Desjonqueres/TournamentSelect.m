function selectedIndividualIndex = TournamentSelect(fitnessList, tournamentProbability, tournamentSize);
    fitnessList = fitnessList';
    for i=1:size(fitnessList,1)  
        Matrix(i,2) = i;
        Matrix(i,1)= fitnessList(i);
    end
    Matrix;

    ind = [];
    randselected = datasample(Matrix,tournamentSize,'Replace',false);
    randselected = sortrows(randselected, "descend");


     for i = 1:size(fitnessList,1)
        r = rand;
        r;
        if r < tournamentProbability
                [max_values,idx]=max(randselected(1,:));

                out=[randselected(idx') max_values'];
                out = out(2);
                
                selectedIndividualIndex = floor(out);

                break
        elseif  r > tournamentProbability

            
            randselected =randselected(2:end,:);
            if size(randselected,1) == 1
                selectedIndividualIndex = randselected(2);      
                return
            end   
        end

        continue
     end
end