%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% Penalty method for minimizing
%
% (x1-1)^2 + 2(x2-2)^2, s.t.
%
% x1^2 + x2^2 - 1 <= 0.
%
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% The values below are suggestions - you may experiment with
% other values of eta and other (increasing) sequences of the
% µ parameter (muValues).
clear all;
X1 = [];
X2 = [];
muValues = [1 10 100 1000];
eta = 0.0001;
xStart =  [1,2]; 
gradientTolerance = 1E-4;
%muValues = linspace(1,10,10); %Uncomment to plot (mu,X1) and (mu,X2)

for i = 1:length(muValues)
 mu = muValues(i);
 x = RunGradientDescent(xStart,mu,eta,gradientTolerance);
 sprintf('x(1) = %3f, x(2) = %3f mu = %d',x(1),x(2),mu);
 X1(end+1) = x(1);
 X2(end+1) = x(2);
end

%scatter(muValues,X2)
% plot(muValues,X1)
% title("x1,x2 versus muValues")
% hold on
% plot(muValues,X2)
% hold off


