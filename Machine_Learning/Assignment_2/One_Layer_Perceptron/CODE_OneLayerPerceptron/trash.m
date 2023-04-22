clear all;

xTest2 = loadmnist2();
[xTrain, tTrain, xValid, tValid, xTest, tTest] = LoadMNIST(3);


%%
layers = [
    imageInputLayer([28 28 1])
    
    convolution2dLayer(3,8,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,16,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    maxPooling2dLayer(2,'Stride',2)
    
    convolution2dLayer(3,32,'Padding','same')
    batchNormalizationLayer
    reluLayer
    
    fullyConnectedLayer(10)
    softmaxLayer
    classificationLayer];

%Play With the parameters. 30 epochs recommended but 5 works fine.

options = trainingOptions('sgdm', ...
    'InitialLearnRate',0.01, ...
    'MaxEpochs',5, ...
    'Shuffle','every-epoch', ...
    'ValidationData',{xValid,tValid}, ...
    'ValidationFrequency',30, ...
    'ValidationPatience',5,...
    'Verbose',false, ...
    'Plots','training-progress');



net = trainNetwork(xTrain,tTrain,layers,options);
%%
YPred = classify(net,xTest2);
writematrix(YPred,"classifications.csv");