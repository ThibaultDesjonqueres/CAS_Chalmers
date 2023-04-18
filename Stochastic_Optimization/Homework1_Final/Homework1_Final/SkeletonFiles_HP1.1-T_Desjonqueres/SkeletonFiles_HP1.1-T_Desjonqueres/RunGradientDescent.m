% This function should run gradient descent until the L2 norm of the
% gradient falls below the specified threshold.

function x = RunGradientDescent(xStart, mu, eta, gradientTolerance)
    xo = xStart;
    for i = 1:100000000
        xi = xo - eta*ComputeGradient(xo, mu);
        l2Norm=norm(ComputeGradient(xo, mu));
      
        if l2Norm < gradientTolerance
            break
        end
        xo = xi;     
    end
    x=xi;
end