% First compute the function value, then compute the fitness
% value; see also the problem formulation.

function fitness = EvaluateIndividual(x)
    a = (1.5-x(1) + x(1)*x(2))^2;
    b = (2.25-x(1)+x(1)*x(2)^2)^2;
    c = (2.625-x(1)+x(1)*x(2)^3)^2;
    func = (a + b + c);
    fitness = 1/(func+1);
end