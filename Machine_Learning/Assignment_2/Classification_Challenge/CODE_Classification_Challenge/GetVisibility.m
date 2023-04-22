function visibility = GetVisibility(cityLocation)
    visibility = pdist2([cityLocation(:,1), cityLocation(:,2)], [cityLocation(:,1), cityLocation(:,2)]);
    visibility = 1./visibility;
    visibility(visibility==Inf)= 0;
end