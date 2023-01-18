% This function implements the BFGS optimizer with Wolfe conditions line
% search. The implementation follows the algorithm provided in the
% classical book of Nocedal and Wright:
% J. Nocedal and S. Wright, "Numerical Optimization", Springer, 2006.
%
% The current file is self-contained, i.e., it includes all the
% necessary functions of the BFGS optimizer and line search.
%
% The initial point is the input of the algorithm, while as output, the 
% algorithm returns the best solution it has detected and its function 
% value. 

function [x,fx] = so_bfgs(x0)

% GLOBAL PARAMETERS

% Check the main program for an explanation of each global parameter.
global paramVCB;
global runCount;
global paramSO;
global trainingSet;

% Network evaluations so far in the current VCB run.
nfev = runCount.eval;

% Available evaluations for the current BFGS run.
optiFev = (paramVCB.MAXeval-nfev)/(paramVCB.MAXcycles-runCount.cycles+1);

% INITIALIZATION 

% Identity matrix.
I = eye(paramSO.dim);

% sol.x      : This vector is the current point of the algorithm.
% sol.fx     : This is the function value of the current point.
% sol.gx     : This vector is the gradient of the current point.
% sol.H      : This matrix is the current inverse Hessian approximation.
% sol.xbest  : This vector is the best point so far.
% sol.fbest  : This is the function value of the best point.
% sol.gbest  : This is the gradient of the best point so far.
sol.x = x0;
sol.fx = evalObjective(sol.x);
sol.gx = grad(sol.x);
sol.H = I;
sol.xbest = sol.x;
sol.fbest = sol.fx;
sol.gbest = sol.gx;

% START BFGS ITERATIONS 

% Stopping flag.
stop = 0;
% Iterations counter.
iter = 0;

% Main loop
while (stop == 0)
    
    % Update iterations counter.
    iter = iter + 1;
    
    % Keep current function value and gradient.
    fprev = sol.fx;
    gprev = sol.gx;
    
    % Calculate the new search direction.
    pnew = -sol.H*sol.gx; 
    
    % LINE SEARCH
    
    % Apply the Wolfe conditions line search of reference above. The
    % line search procedure returns the proper step size "alpha" for the
    % determination of the new search point.
    alpha = lineSearch(sol,pnew);
    
    % Scale search direction. 
    s = alpha*pnew;
    
    % NEW SEARCH POINT
    
    sol.x = sol.x + s;
    sol.fx = evalObjective(sol.x);
    sol.gx = grad(sol.x);
    
    % UPDATE INVERSE HESSIAN
    
    % Calculate the necessary quantities.
    y = sol.gx-gprev;
    sy = s'*y;
    Hy = sol.H*y;
    
    % BFGS update of inverse Hessian. 
    % Uncomment below to use the BFGS update of the inverse Hessian.
    sol.H = sol.H + (sy+y'*Hy)*(s*s')./(sy^2) - (Hy*s'+s*Hy')./sy;
    
    % Check symmetry and positive definiteness of the inverse Hessian. The 
    % BFGS algorithm is expected to retain these properties if the line search 
    % using Wolfe conditions has been successfully applied. In case the 
    % inverse Hessian is not symmetric or positive definite, then it is 
    % reset to the identity matrix.
    ispdf = all(eig(sol.H) > 0);
    issym = issymmetric(sol.H);
    if (~(ispdf && issym))
        sol.H = I;
    end
    
    % UPDATE BEST POINT
    
    if (sol.fx < sol.fbest)
        sol.xbest = sol.x;
        sol.fbest = sol.fx;
        sol.gbest = sol.gx;
    end
    
    % TERMINATION CONDITION
    
    % First, we check if the number of network evaluations has already
    % exceeded its maximum value.
    if (runCount.eval >= paramVCB.MAXeval)
        % Maximum total fevals reached and BFGS will stop 
        stop = 1;
    % Then, we check if a fixed number of VCB cycles is determined by the
    % user. This implies a specific number of network evaluations per
    % cycle.
    elseif (paramVCB.MAXcycles > 0)
        % Check if the stopping condition is fulfilled.
        if (runCount.eval-nfev >= optiFev-trainingSet.size)
            % Maximum current run fevals reached and BFGS will stop
            stop = 1;
        % Otherwise, a restart may be needed (checked later).
        else
            stop = 2;
        end
    % If no fixed number of VCB cycles has been set by the user, the
    % algorithm terminates on its stopping conditions.
    else
        % Calculate the relative function improvement.
        relf = abs((fprev-sol.fx)/sol.fx);
        % Check termination conditions.
        if (iter >= paramSO.maxit)
            % Maximum current run iters reached and BFGS will stop
            stop = 1;
        elseif (relf <= paramSO.eps)
            % Relative improvement in solution value is small and BFGS will stop
            stop = 1;
        elseif (norm(sol.gx) <= paramSO.geps)
            % Gradient norm is small and BFGS will stop
            stop = 1;
        end
    end
    
    % CHECK RESTART CONDITION
    
    if (stop == 2)
        % Calculate the relative function improvement.
        relf = abs((fprev-sol.fx)/sol.fx);
        if (relf <= paramSO.eps)
            % BFGS will restart because relative improvement is small
            stop = 3;
        elseif (norm(sol.gx) <= paramSO.geps)
            % BFGS will restart because gradient norm is small
            stop = 4;
        end
        
        if (stop > 2)
            % Generate a new initial point.
            x0 = paramSO.Xmin + rand(paramSO.dim,1).*(paramSO.Xmax-paramSO.Xmin);
            % Set it as the current point.
            sol.x = x0;
            sol.fx = evalObjective(sol.x);
            sol.gx = grad(sol.x);
            sol.H = I;
            % BFGS restarted
            % Update best point so far.
            if (sol.fx < sol.fbest)
                sol.xbest = sol.x;
                sol.fbest = sol.fx;
                sol.gbest = sol.gx;
            end
        end
        % Reset stopping flag.
        if ((runCount.eval-nfev >= optiFev-trainingSet.size) || (norm(sol.gbest) <= paramSO.geps))
            % BFGS stops
            stop = 1;
        else
            % BFGS will continue to new iteration
            stop = 0;
        end
    end
    
end

% SET OUTPUT VALUES 

x = sol.xbest;
fx = sol.fbest;

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function calls the implemented neural network to evaluate the
% current mini-batches set. 
function fx = evalObjective(x)

global paramSO;

MSEvec = so_RBF_set(x);

% Calculate the objective function value using the VCB penalty function.
fx = MSEvec(1) + paramSO.L*MSEvec(2);

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function implements the gradient of the objective function. The
% output is the gradient vector "gx" of the objective function for the
% input vector "x". The specific gradient function refers to the RBF 
% network of our example. If a different neural network is desirable, 
% the gradient function shall be properly modified.
function gx = grad(x)

% GLOBAL PARAMETERS

% Check the main program for an explanation of each global parameter.
global trainingSet; 
global neuralModel;
global paramVCB;
global runCount;
global paramSO;
global miniBatch;

% BUILD THE RBF NETWORK

% Preallocate vectors.
w = zeros(1,neuralModel.nn);
s = zeros(1,neuralModel.nn);
muvec = zeros(neuralModel.nn,trainingSet.dim);
% Loop on the number of neurons.
for k=1:neuralModel.nn
    % Index offset of the k-th neuron parameters in "x".
    os = (k-1)*(trainingSet.dim+2)+1;
    % Weight of k-th neuron.
    w(k) = x(os+1);
    % Standard deviation of k-th neuron.
    s(k) = x(os+2);
    % Mean vector of k-th neuron.
    muvec(k,:) = x(os+3:os+2+trainingSet.dim); 
end
% Bias weight.
w0 = x(1);

% CALCULATE THE MSE PER MINI-BATCH

% Preallocate vectors and matrices.
E = zeros(1,paramVCB.nbat);
diff = zeros(paramVCB.nbat,paramVCB.npat);
mm = cell([1 paramVCB.nbat]);
G = cell([1 paramVCB.nbat]);
% Loop on the mini-batches of the current mini-batch set.
for i=1:paramVCB.nbat
    % Initialize cell matrices.
    mm{i} = zeros(paramVCB.npat,neuralModel.nn);
    G{i} = zeros(paramVCB.npat,neuralModel.nn);
    % Calculate the MSE of the i-th mini-batch.
    E(i) = 0;
    % Loop on the pattern vectors of the current mini-batch.
    for j=1:paramVCB.npat
        % Get the index of the j-th pattern vector of the i-th mini-batch 
        % in the training set.
        pati = miniBatch(i,j);
        % Get the current pattern vector from the training set.
        pat = trainingSet.patterns(pati,1:trainingSet.dim); 
        % Get the current pattern value.
        fpat = trainingSet.patterns(pati,trainingSet.dim+1);
        % Calculate the RBF error for the current pattern.
        y = w0;
        for k=1:neuralModel.nn
            mm{i}(j,k) = norm(pat-muvec(k,:))^2;
            G{i}(j,k) = exp(-0.5*mm{i}(j,k)/(s(k)^2));
            y = y + w(k)*G{i}(j,k);
        end
        % Difference between RBF and true output for j-th pattern of i-th
        % mini-batch.
        diff(i,j) = (y-fpat); 
        % Update the counter of single-pattern RBF evaluations.
        runCount.eval = runCount.eval+1;
    end
    % MSE of the RBF for the i-th mini-batch.
    E(i) = mean(diff(i,:).^2); 
end
% Average MSE over the mini-batches of the current mini-batch set.
Ebar = mean(E); 

% CALCULATE DERIVATIVES

% Derivative of the average error term.
dEidw = cell([1 paramVCB.nbat]);
dEbardw = zeros(1,paramSO.dim);
% Loop on the mini-batches of the current mini-batch set.
for i=1:paramVCB.nbat
    % Initialize matrices.
    q = zeros(1,paramSO.dim);
    dEidw{i} = zeros(1,paramSO.dim);
    % Loop on the pattern vectors of the current mini-batch.
    for j=1:paramVCB.npat
        % Get the index of the j-th pattern vector of the i-th mini-batch 
        % in the training set.
        pati = miniBatch(i,j);
        % Get the current pattern vector from the training set.
        pat = trainingSet.patterns(pati,1:trainingSet.dim); 
        % Derivative vector.
        dNdw = zeros(1,paramSO.dim);
        dNdw(1) = 1; % dNdw0
        for k=1:neuralModel.nn % k-th neuron
            os = (k-1)*(trainingSet.dim+2)+1; % index offset
            dNdw(os+1) = G{i}(j,k); % dN/dw_k
            dNdw(os+2) = w(k)*mm{i}(j,k)*G{i}(j,k)/(s(k)^3); % dN/ds_k
            dNdw(os+3:os+2+trainingSet.dim) = w(k)*G{i}(j,k)*(pat-muvec(k,:))/(s(k)^2);
        end
        % Update the counter of single-pattern RBF evaluations.
        runCount.eval = runCount.eval+1;
        q = q + 2*diff(i,j)*dNdw;
    end
    dEidw{i} = q./paramVCB.npat;
    dEbardw = dEbardw + dEidw{i};
end
dEbardw = dEbardw./paramVCB.nbat;
% Derivative of the error variance term.
q = zeros(1,paramSO.dim);
for i=1:paramVCB.nbat
    q = q + 2*(E(i)-Ebar)*(dEidw{i}-dEbardw);
end

% GRADIENT VECTOR

gx = dEbardw + (paramSO.L/paramVCB.nbat)*q;
gx = gx';

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function is the derivative of the unidimensional problem that is
% solved in line search.
function phid = phid(x,p,a)

x1 = x + a*p;
phid = p'*grad(x1);

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function implements the Wolfe conditions line search that is
% presented by Nocedal and Wright in their classical book "Numerical
% Optimization" (see reference at the top of the current file). The
% line search procedure admits as input the structure "sol" and the new 
% search direction "pnew" and returns as output the step size "alpha".
function alpha = lineSearch(sol,pnew)

global paramSO;

% INITIALIZATION

% Initial step size.
a = 1;
% Maximum step size.
amax = 100;
% Initial step size interval.
a0 = 0;
a1 = a;
fx = sol.fx;
f0 = sol.fx;
% Directional derivative at point of step size a0.
d0 = phid(sol.x,pnew,a0);

% START LINE SEARCH ITERATIONS

% Stopping flag.
stop = 0;
% Initialize line search iteration counter.
iter = 0;

% Main loop.
while (stop == 0)
    
    % Update iterations counter.
    iter = iter + 1;
    
    % Calculate the new point and its directional derivative.
    x1 = sol.x + a1*pnew;
    f1 = evalObjective(x1);
    d1 = phid(sol.x,pnew,a1);
    
    % WOLFE CONDITIONS 
    
    % Armijo condition.
    if  ((f1 > f0 + paramSO.C1*a1*d0) || ((f1 >= fx) && (iter > 0)))
        % A proper point exists in [a0,a1]. Applying zoom procedure.
        alpha = zoomInterval(sol,pnew,d0,a0,a1,f0,fx);
        stop = 1;
    else
        % Curvature condition.
        if(abs(d1) <= -paramSO.C2*d0)
            % Strong Wolfe conditions hold.
            alpha = a1;
            stop = 1;
        else
            % Check if the current point is in ascending region.
            if (d1 >= 0)
                % A proper point exists in [a1,a0]. Applying zoom procedure.
                alpha = zoomInterval(sol,pnew,d0,a1,a0,f0,f1);
                stop = 1;
            end
        end
    end
    
    % UPDATE STEP SIZE
    
    if (stop == 0)
        a0 = a1;
        a1 = min(2*a1,amax);
        fx = f1;
    end
    
    % CHECK TERMINATION CONDITION
    
    if (iter >= paramSO.LSmaxit)
        stop = 1;
    end
end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function implements the interval zoom function of the line search 
% that Nocedal and Wright present in their classical book "Numerical
% Optimization" (see reference at the top of the current file).
function alpha = zoomInterval(sol,pnew,d0,aL,aR,f0,fL)

global paramSO;

% START ZOOM PROCEDURE

% Initialize iteration counter.
iter = 0;
% Stopping flag.
stop = 0;

% Main zoom loop.
while (stop == 0)
    
    % Update iteration counter.
    iter = iter + 1;
    
    % New step size.
    a1 = 0.5*(aL+aR);

    % New point.
    x1 = sol.x + a1*pnew;
    f1 = evalObjective(x1);
    d1 = phid(sol.x,pnew,a1);
    
    % Check for termination. If the directional gradient of the new point
    % is smaller than the user-defined tolerance, or the step size is
    % smaller than the same tolerance, then the zoom procedure stops.
    if ((abs(d1) <= paramSO.geps) || (abs(aR-aL) <= paramSO.geps))
        stop = 1;
    else
        % Armijo condition.
        if ((f1 > f0 + paramSO.C1*a1*d0) || (f1 >= fL))
            aR = a1;       
        else
            % Curvature condition.
            if (abs(d1) <= -paramSO.C2*d0)
                stop = 1;
            else
                if ((aR-aL)*d1 >= 0)
                    aR = aL;
                    aL = a1;
                    fL = f1;
                else
                    aL = a1;
                    fL = f1;
                end
            end
        end
    end
    
end

% Set output value.
alpha = a1;

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% MIT License

% Copyright 2023 Dimitra G. Triantali, Konstantinos E. Parsopoulos, Isaac E. Lagaris

% Permission is hereby granted, free of charge, to any person obtaining 
% a copy of this software and associated documentation files (the "Software"), 
% to deal in the Software without restriction, including without limitation 
% the rights to use, copy, modify, merge, publish, distribute, sublicense, 
% and/or sell copies of the Software, and to permit persons to whom the 
% Software is furnished to do so, subject to the following conditions:

% The above copyright notice and this permission notice shall be included 
% in all copies or substantial portions of the Software.

% THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
% OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
% FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL 
% THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
% LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
% OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN 
% THE SOFTWARE.



