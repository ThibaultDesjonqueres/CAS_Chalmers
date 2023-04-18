%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Parameter specifications
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

numberOfRuns = 10;                % Do NOT change
populationSize = 100;              % Do NOT change
maximumVariableValue = 5;          % Do NOT change (x_i in [-a,a], where a = maximumVariableValue)
numberOfGenes = 50;                % Do NOT change
numberOfVariables = 2;		       % Do NOT change
numberOfGenerations = 300;         % Do NOT change
tournamentSize = 2;                % Do NOT change
tournamentProbability = 0.75;      % Do NOT change
crossoverProbability = 0.8;        % Do NOT change


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Batch runs
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

LIST = [0,0.01,0.02,0.03,0.04,0.05,0.07,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1];
%LIST = [0,0.01,0.02,0.03];
indices = (1:length(LIST));
% Define more runs here (pMut < 0.02) ...
plt = zeros(1,length(LIST));
BestVar1 = [];
BestVar2 = [];
MaxFit = [];
Med = [];
    for k = LIST
    
    
        mutationProbability = k
        sprintf('Mutation rate = %0.5f', mutationProbability);
        maximumFitnessList002 = zeros(numberOfRuns,1);
        for i = 1:numberOfRuns 
            i;
         [maximumFitness, bestVariableValues]  = RunFunctionOptimization(populationSize, numberOfGenes, numberOfVariables, maximumVariableValue, tournamentSize, ...
                                               tournamentProbability, crossoverProbability, mutationProbability, numberOfGenerations);
         sprintf('Run: %d, Score: %0.10f', i, maximumFitness);
         MaxFit(i) = maximumFitness;
         BestVar1(i)= bestVariableValues(1);
         BestVar2(i)= bestVariableValues(2);
         maximumFitnessList002(i,1) = maximumFitness;
         median0 = median(maximumFitnessList002);
%           Med(i)= median0
        end
       
    
        % ... and here (pMut > 0.02)
        
        
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %% Summary of results
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        
        % Add more results summaries here (pMut < 0.02) ...
        
        average002 = mean(maximumFitnessList002);
        median002 = median(maximumFitnessList002)
       
        std002 = sqrt(var(maximumFitnessList002));
        
        plt(find(LIST==k)) = median002;

         
        sprintf('PMut = 0.02: Median: %0.10f, Average: %0.10f, STD: %0.10f', median002, average002, std002);
    end

scatter(LIST,plt)
% ... and here (pMut > 0.02)


% Obtained result : 3.08540159718399	0.520427689565054





