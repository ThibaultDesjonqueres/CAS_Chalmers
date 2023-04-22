clear all;

% THIS SECTION CREATE A TARGET LIST. NEEDS TO RUN ONLY ONCE.

n =3;   % 2,3,4,5
nTrials = 10E2; %10E4;
ZerosCheck = zeros(1,2^n);
nEpochs = 10E2;

eta = 0.05;
target_List = zeros(2^n,(2^2)^n)'   

for i = 1:nTrials
    t = randi(2,2^n)-1;
    t = t(:,1);
    t(~t) = -1;
    target = t';
    
    if ismember(target, target_List, 'rows') == 0
        target_List(i,:) = target;
        target_List;
    elseif ismember(target, target_List, 'rows') == 1
        continue
    end

    target_List = target_List(any(target_List,2),:);
end

%% THIS SECTION PERFORMS THE CONVERGENCE CALCULATIONS
W = ones(1,n);
used_bool = [];
theta = 0;
for i = 1:n
    W(i) = rand/sqrt(n);
end

B = CreateBool(n)';

for epoch = 1:size(target_List,1)

    target = target_List(epoch,:); %Loop over each pregenerated potential solutions
    target;
    for i = 1:nTrials
        y = sgn2(W*B-theta);
        error = target  - y;
        if error == ZerosCheck  %ZerosCheck = [0,0,0,0,...]    (dim =[1x2^n])
            used_bool(end+1,:) = target;  % add target to a list and go next iteration
            break
        else
            deltaW = eta*(target - y)*B';   %Update  error(1)*B(:,1)'
            deltaTheta = sum(-eta*(target - y)); %Update 
        end

        W = W + deltaW; %Update 
        theta = theta + deltaTheta;
    end
end

