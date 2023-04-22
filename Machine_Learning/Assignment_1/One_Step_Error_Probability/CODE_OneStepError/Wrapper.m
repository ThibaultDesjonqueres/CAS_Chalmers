% Completed Code

clear all;
p = [12,24,48,70,100,120];
N = 120 ;

A=[];
trials = 1000000;

goal = [];
patternSet = [];

Res = [];
for k = 1:length(p)
    counterGood = 0;
    counterBad = 0;
    p(k)
    for trial = 1:trials
        
        patternSet = CreateMatrices(p(k),N);
        A = size(patternSet);
        rowPatternSet = A(1);
        R = randi([1,rowPatternSet]);
        goal = transpose(patternSet(R,:));
        W = zeros(N);
%         for i = 1:rowPatternSet
            W =  ((patternSet'* patternSet))/N; 
%         end
%         for i = 1:N
        W = W - diag(diag(W));
%             W(i,i)=0;
%         end
        afterW = W*goal;
        
        Random = randi([1,N]);
        bitGoal = goal(Random);
        bitCheck = afterW(Random);
        if sgn(bitCheck) == bitGoal
            counterGood = counterGood+1;
%         else
%             counterBad = counterBad+1;
        end
    end
cGood(k)= counterGood     %       99953       98864       94494       90504       86323       84249
end
