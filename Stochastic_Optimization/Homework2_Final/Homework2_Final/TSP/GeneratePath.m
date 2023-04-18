function path = GeneratePath(pheromoneLevel, visibility, alpha, beta)
    
    Pnum = (pheromoneLevel.^alpha).*(visibility.^beta);   %Eq.(4.3)
    Pden = sum((pheromoneLevel.^alpha).*(visibility.^beta),1);  %Eq.(4.3)
    
    visibility = visibility.*~eye(size(visibility)); 
    P = (Pnum./Pden);
    P = P.*~eye(size(P));
    
    Tabu = (1:size(visibility,1));
    Tot = Tabu;
    List = [];
    for i = 1:size(visibility,1)
        
        if i == 1
            
            node = randsample(Tabu,1, true);   %Pick random column
        end
        Tabu = Tabu(Tabu~=node);
        List(end+1,1) = node;

        R = randsample(P(:,node), 1, true, P(:,node));  %Pick next according to prob
        next = (find(P(:,node)== R));
        if i~= size(visibility,1)-1
            P(:,node) = 0;
            P(node,:) = 0;
        end
        node = next;
    end

path = List'; 


