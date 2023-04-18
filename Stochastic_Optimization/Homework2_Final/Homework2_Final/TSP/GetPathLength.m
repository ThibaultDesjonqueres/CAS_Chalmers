function pathlength = GetPathLength(path,cityLocation)
length = [];
for i = path
    if i~=(size(cityLocation,1))
        length(end+1,1) = pdist2([cityLocation(path(i),1), cityLocation(path(i),2)], ...
            [cityLocation(path(i+1),1), cityLocation(path(i+1),2)]);
    end
    if i==path(size(cityLocation,1)) % Special case when reaching last i
        length(end+1,1) = pdist2([cityLocation(path(i),1), cityLocation(path(i),2)],...
            [cityLocation(1,1), cityLocation(1,2)]);
    end

pathlength = sum(length);
end
end
