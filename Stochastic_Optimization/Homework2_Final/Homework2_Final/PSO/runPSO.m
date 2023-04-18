clear all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%% Read Me %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 1) Run the code until the contour plot appears (it takes 2 minutes).
% 2) Use the Brush functionality to select each one of the 4 regions
% containing clouds of point. Name them "brushedDataTL","brushedDataTR",
% "brushedDataBL";"brushedDataBR" (TL = Top Left, BR = Bottom Right...)
% 3) Close the contour plot. The coordinates of the best points will
% appear.
% 4) The  plot of thosw 4 best points will appear. 
% INITIALISATION
bestPositionTracker = [];
bestPerformanceTracker = [];

boundMin = -5;  
boundMax = 5;
boundary = [boundMin,boundMax];
[x,y] = meshgrid(-5:0.1:5);

epochs = 100;
iterationsNb = 1000;
dimension = 2;
population = 40;  

w=1;
damp = 0.99999;
c1=2;
c2=2;
vMax = 5;

for epoch = 1:epochs

    %Initialize positions
    xCoord = boundMin + (boundMax-boundMin)*rand(population,1);
    yCoord = boundMin + (boundMax-boundMin)*rand(population,1);
    pPosition = [xCoord,yCoord];  
    
    %Initialize velocities
    xVelocity = -((boundMax-boundMin)/2)+(boundMax-boundMin)*rand(population,1);
    yVelocity = -((boundMax-boundMin)/2)+(boundMax-boundMin)*rand(population,1);
    pVelocity = [xVelocity,yVelocity];
    
    %Create structures 
    particle.position = pPosition;
    particle.velocity = pVelocity;
    particle.performance = f(particle.position(:,1),particle.position(:,2));
    particle.bestPos = particle.position;
    particle.bestPerf = particle.performance;
    
    %Initialize Best Global Position & Performance
    globalBestInitial = Inf;
    if particle.bestPerf < globalBestInitial
        [M,idx] = min(particle.bestPerf);
        globalBestPerf = min(particle.bestPerf);
    end
    globalBestPos = particle.position(idx,:);

% UPDATES

    for it = 1:iterationsNb
        Update(particle,c1,c2,globalBestPos,dimension,population,w,vMax,globalBestPerf)        
        w = w*damp;
    end
    bestPositionTracker(epoch,:) = globalBestPos;
    bestPerformanceTracker(epoch) = globalBestPerf;
end

bestPerformanceTracker = bestPerformanceTracker';


%% PLOT
%surf(x,y,f(x,y));
%plot(log(0.01+f(x,y)))

contourf(x,y,f(x,y))
hold on;
brush on;
k = plot(bestPositionTracker(:,1),bestPositionTracker(:,2),'.')
k = gcf
uiwait(k)


%%
squareTL = [];
for i = 1:size(brushedDataTL,1)
    squareTL(i,:) = find(bestPositionTracker == brushedDataTL(i,:));
end
squareTL = squareTL(:,1);
perfTL = min(bestPerformanceTracker(squareTL,:));
posTL = find(bestPerformanceTracker == perfTL);
bestPosInSquareTL = bestPositionTracker(posTL,:);
disp(["The minimum in the Top Left region is", num2str(perfTL),...
    "with coordinates",num2str(bestPosInSquareTL)])
%%
squareTR = [];
for i = 1:size(brushedDataTR,1)
    squareTR(i,:) = find(bestPositionTracker == brushedDataTR(i,:));
end
squareTR = squareTR(:,1);
perfTR = min(bestPerformanceTracker(squareTR,:));
posTR = find(bestPerformanceTracker == perfTR);
bestPosInSquareTR = bestPositionTracker(posTR,:);
disp(["The minimum in the Top Left region is", num2str(perfTR), ...
    "with coordinates",num2str(bestPosInSquareTR)])
%%
squareBL = [];
for i = 1:size(brushedDataBL,1)
    squareBL(i,:) = find(bestPositionTracker == brushedDataBL(i,:));
end
squareBL = squareBL(:,1);
perfBL = min(bestPerformanceTracker(squareBL,:));
posBL = find(bestPerformanceTracker == perfBL);
bestPosInSquareBL = bestPositionTracker(posBL,:);
disp(["The minimum in the Top Left region is", num2str(perfBL), ...
    "with coordinates",num2str(bestPosInSquareBL)])
%%
squareBR = [];
for i = 1:size(brushedDataBR,1)
    squareBR(i,:) = find(bestPositionTracker == brushedDataBR(i,:));
end
squareBR = squareBR(:,1);
perfBR = min(bestPerformanceTracker(squareBR,:));
posBR = find(bestPerformanceTracker == perfBR);
bestPosInSquareBR = bestPositionTracker(posBR,:);
disp(["The minimum in the Top Left region is", num2str(perfBR), ...
    "with coordinates",num2str(bestPosInSquareBR)])


%% PLOT ONLY BEST

Best = [bestPosInSquareBL;bestPosInSquareTL;bestPosInSquareBR;bestPosInSquareTR];
contourf(x,y,f(x,y),70,":w")
hold on;
brush on;
k = scatter(Best(:,1),Best(:,2),50,"filled", "white")







