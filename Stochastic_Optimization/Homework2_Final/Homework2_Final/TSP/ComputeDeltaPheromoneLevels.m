function deltaPheromoneLevel = ComputeDeltaPheromoneLevels(pathCollection,pathLengthCollection)
    delta = zeros(size(pathCollection,2));

    deltaTau = 1./pathLengthCollection; %Eq.(4,4)

    for i = 1:size(pathCollection,2)-1
            indices(:,:,i) = [pathCollection(:,i+1),pathCollection(:,i)];
    end
    
    for i = 1:size(pathCollection,1)   % i indicates which ant we look at.
        for k = 1:size(pathCollection,2)-1   % n-th xy coordinates set...
            delta(indices(i,1,k),indices(i,2,k),i) = deltaTau(i);
        end
    end

    deltaPheromoneLevel = sum(delta,3); %Eq.(4.5)
end