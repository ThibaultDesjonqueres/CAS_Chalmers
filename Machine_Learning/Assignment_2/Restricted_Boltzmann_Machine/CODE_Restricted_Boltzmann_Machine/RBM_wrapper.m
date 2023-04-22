%%%%%%%%%% RBM_Code %%%%%%%%%%%

clear all;

Vj = [[-1,-1,-1],
    [1,-1,1],
    [-1,1,1],
    [1,1,-1]];

ListTot = CreateBool(3);
ListT= [ListTot(1,:)',ListTot(6,:)',ListTot(4,:)',ListTot(7,:)',ListTot(2,:)',ListTot(3,:)',ListTot(5,:)',ListTot(8,:)'];
Vj =Vj';
N = size(Vj,1);
Mlist = [1,2,4,8];
% Mlist = [2];

thetav = zeros(3,1);


deltawList = [];
deltathetavList = [];

deltathetahList = [];

nTrials = 2000;
minibatchnb = 400;
onepatternnb = 400;
eta = 0.008;
nOutter =300;

% nTrials = 1;
% minibatchnb = 4;
% onepatternnb = 20;
% eta = 0.005;
% nOutter =30;


thetahM = [];
thetavM = [];


count = zeros(4,4);
countres = zeros(4,4);
fail = zeros(4,1);
tic
for M = [Mlist]
    M
    w = normrnd(0,1,M,3)./sqrt(3);
    thetah = zeros(1,M)';
    for trials = 1:nTrials
        deltawList = [];
        deltathetavList = [];
        deltathetahList = [];
        for miniBatch = 1:minibatchnb
            
            idx=randi(length(Vj'),1);               
            V = Vj(:,idx);
            bih0 = w*Vj(:,idx) - thetah;   %V(0) and bih(0)

            for onepattern = 1:onepatternnb 
                
                bih = w*V - thetah;
                pbih = 1./((1+exp(-2.*bih)));
                r = rand(size(pbih,1),size(pbih,2));
                h = pbih-r;
                h(h<0) = -1;
                h(h>=0) = 1;
        
                bjv = (h'*w - thetav')';
                pbjv = 1./((1+exp(-2.*bjv)));
                R = rand(size(pbjv,1),size(pbjv,2));
                V = pbjv-R;
                V( V < 0 ) = -1;
                V( V >= 0 ) = 1;
                
                
            end
    
        delw =(eta*(tanh(bih0).*Vj(:,idx)'-tanh(bih).*V'));
        delThetav = -eta*(Vj(:,idx)-V);
        delThetah = -eta*(tanh(bih0)-tanh(bih));
    
        deltawList = cat(3,deltawList,delw);
        deltathetavList = cat(3,deltathetavList,delThetav);
        deltathetahList = cat(3,deltathetahList,delThetah);
        end
    w = w + sum(delw,3);
    thetav = thetav + sum(delThetav,3);
    thetah = thetah + sum(delThetah,3);
    end
    

    for miniBatch = 1:nOutter
        IDX=randi(length(ListT'),1);
        V2 = ListT(:,IDX);
        IndexM = find(Mlist==M);

%         if all(V2 == Vj(:,1))
%             count(IndexM,1) =count(IndexM,1)+ 1;
%         elseif all(V2 == Vj(:,2))
%             count(IndexM,2) =count(IndexM,2)+ 1;
%         elseif all(V2 == Vj(:,3))
%             count(IndexM,3) =count(IndexM,3)+ 1;
%         elseif all(V2 == Vj(:,4))
%             count(IndexM,4) =count(IndexM,4)+ 1;
%         end

        BIH0 = w*ListT(:,IDX) - thetah;
        for onepattern = 1:nOutter
            
            BIH = w*V2 - thetah;
            
            PBIH = 1./((1+exp(-2.*BIH)));
            
            r = rand(size(pbih,1),size(PBIH,2));
            H = PBIH-r;
            H( H < 0 ) = -1;
            H( H >= 0 ) = 1;
            
            
           
            BJV = (sum(H.*w,1))' - thetav;
            
            PBJV = 1./((1+exp(-2.*BJV)));
            R = rand(size(PBJV,1),size(PBJV,2));
            V2 = PBJV-R;
            V2( V2 < 0 ) = -1;
            V2( V2 >= 0 ) = 1;
            V2;
            %return
            
        end

        IndexM = find(Mlist==M);
        
        if all(V2 == ListT(:,1))
            countres(IndexM,1) =countres(IndexM,1)+ 1;
        elseif all(V2 == ListT(:,2))
            countres(IndexM,2) =countres(IndexM,2)+ 1;
        elseif all(V2 == ListT(:,3))
            countres(IndexM,3) =countres(IndexM,3)+ 1;
        elseif all(V2 == ListT(:,4))
            countres(IndexM,4) =countres(IndexM,4)+ 1;
        else
            fail(IndexM,1) =fail(IndexM,1)+ 1;
        end
    end
    countres
end



%%
% Pdata = 1/4;
Pdata = 1/4;
Pcountres = countres./nOutter;
Dk = Pdata*log(Pdata./Pcountres)
GoodCount = sum(Pcountres,2);
FailCount = 1-sum(Pcountres,2);
Dksum = sum(Dk,2)
toc








%% Theory Curve


% NN = size(Vj,1);
NN = 3;
AA = [];
Res = [0.7289,0.4370,0.0844, 0.0795]; 
r = [1,2,3,4,5,6,7,8];
%LinM = linspace(0,M,10);
BB = 2^(NN-1)-1;
CC = zeros(1,5);
LinN = linspace(0,10,1000);
for i = 1:BB 
    i
    AA(1,end+1) =NN - abs(log2(i+1)) - ((i+1)/(2.^(abs(log2(i+1)))));

end
AA = [AA,CC];

hold on
plot(r,AA);
hold on
scatter(Mlist,Res)


%%



RESULT = [[53    20    21    77],
    [22    47    73    73],
    [68    85    62    63],
    [73    50    76    83]];

