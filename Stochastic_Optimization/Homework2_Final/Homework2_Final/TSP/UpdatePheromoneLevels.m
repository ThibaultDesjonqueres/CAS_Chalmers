function  pheromoneLevel = UpdatePheromoneLevels(pheromoneLevel,deltaPheromoneLevel,rho)
    pheromoneLevel = (1-rho)*pheromoneLevel + deltaPheromoneLevel;
    pheromoneLevel(pheromoneLevel<10E-15) = 10E-15;  
end