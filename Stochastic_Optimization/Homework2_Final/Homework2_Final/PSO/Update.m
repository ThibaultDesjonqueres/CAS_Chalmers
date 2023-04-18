function Res = Update(particle,c1,c2,globalBestPos,dimension,population,w,vMax,globalBestPerf)
        for i = 1:population
            % Update Velocities
            particle.velocity(i,:) = w.*particle.velocity(i,:) ...
                + c1.*rand([1 dimension]).*(particle.bestPos(i,:) - particle.position(i,:)) ...
                + c2.*rand([1 dimension]).*(globalBestPos - particle.position(i,:));
            % Bound the velocities
            for j = 1:dimension
                if abs(particle.velocity(i,j)) > vMax
                    if particle.velocity(i,j) > 0
                        particle.velocity(i,j) = vMax;
                    else
                        particle.velocity(i,j) = -vMax;
                    end
                end
            end
            % Update Positions
            particle.position(i,:) = particle.position(i,:) + particle.velocity(i,:);
            % Evaluate the Performance
            particle.performance(i) = f(particle.position(i,1),particle.position(i,2));
            % Update the particle best and global best
            if  particle.performance(i) < particle.bestPerf(i)
                particle.bestPos(i,:) = particle.position(i,:);
                particle.bestPerf(i,:) = particle.performance(i,:); 
    
                if particle.bestPerf(i,:) < globalBestPerf
                    globalBestPerf = particle.bestPerf(i,:);            
                end
            end
        end
end