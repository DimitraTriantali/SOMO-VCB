% This function evaluates vectors of the Pareto set provided by MOPSO 
% according to their MSE over the whole training set. The number of vectors
% to be evaluated is determined by the "evalStrategy" parameter.
function mo_update_best(ParetoSet)

% GLOBAL PARAMETERS

% Check the main program for an explanation of each global parameter.
global runCount;
global evalStrategy;

% EVALUATION

% Counter of so far evaluated vectors.
neval = 0;

% Permute indices of Pareto set vectors.
permInd = randperm(ParetoSet.crs);

% Loop on the number of Pareto set vectors.
for i=1:ParetoSet.crs
    % Get i-th Pareto set vector in random order.
    x = ParetoSet.sol(permInd(i),:);
    % Evaluate the i-th vector using the whole training set.
    fx = mo_RBF_full(x);
    % Update the overall best solution of the current run.
    if (fx < runCount.fbest)
        runCount.xbest = x;
        runCount.fbest = fx;
    end
    % Check termination condition. The procedure stops if the requested 
    % number of "evalStrategy" vectors has been evaluated. If
    % "evalStrategy" is negative or larger than the size of the Pareto 
    % set, the procedure will continue until all Pareto set vectors have 
    % been evaluated.
    neval = neval + 1;
    if ((evalStrategy > 0) && (evalStrategy <= neval))
        break;
    end
end

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
