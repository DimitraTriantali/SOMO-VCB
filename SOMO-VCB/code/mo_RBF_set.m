% This function calculates the Mean Squared Error (MSE) and its 
% standard deviation of the Radial Basis Network (RBF) configuration 
% that is defined in the input vector "x", over all mini-batches of the 
% current mini-batches set. 
% The average MSE and its standard deviation over the whole mini-batches 
% set are returned as output values in the vector "MSEvec=[MSEmean,MSEstd]".
function MSEvec = mo_RBF_set(x)

% The input vector "x" contains all decision variables of the RBF. If "d" 
% denotes the dimension of each training vector, and "L" is the number of 
% neurons of the RBF network, then "x" has the following form:
% x = [ w_0 | w1 s1 M1_1 M1_2 ... M1_d | w2 s2 M2_1 M2_2 ... M2_d | ... 
%       ... | w_L s_L ML_1 ML_2 ... ML_d ]
% where "wi" is the i-th neuron weight, "si" is its spread, and "Mi" is 
% its mean vector, respectively. (i=1...L)

% GLOBAL PARAMETERS

% Check the main program for an explanation of each global parameter.
global trainingSet; 
global neuralModel;
global paramVCB;
global miniBatch;
global runCount;

% BUILD THE RBF NETWORK

% w: This vector contains the weights of all neurons.
% s: This vector contains the standard deviations of all neurons.
% muvec: This vector contains the mean vectors of all neurons.
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

% COMPUTE MSE

% Initialize vector of MSEs of all mini-batches of the current mini-batches set.
MSEset = zeros(1,paramVCB.nbat); 

% Loop on the number of mini-batches (i=1,...,paramVCB.nbat).
for i=1:paramVCB.nbat
    % Initialize total squared error for the current mini-batch.
    Ei = 0; 
    % Loop on the patterns of the current mini-batch (j=1,...,paramVCB.npat).
    for j=1:paramVCB.npat
        % Get the index of the j-th pattern vector of the i-th mini-batch 
        % in the training set.
        pati = miniBatch(i,j);
        % Get the current pattern vector from the training set.
        pat = trainingSet.patterns(pati,1:trainingSet.dim); 
        % Get the current pattern value.
        fpat = trainingSet.patterns(pati,trainingSet.dim+1);
        % Calculate the RBF squared error for the current pattern.
        y = w0;
        for k=1:neuralModel.nn
            y = y + w(k)*exp(-0.5*(norm(pat-muvec(k,:))^2)/(s(k)^2));
        end
        % Update the total squared error of the current mini-batch.
        Ei = Ei + (y-fpat)^2;
        % Update the counter of single-pattern RBF evaluations.
        runCount.eval = runCount.eval+1;
    end
    % Save MSE of current mini-batch.
    MSEset(i) = Ei/paramVCB.npat;
end

% Calculate output values.
MSEvec = [mean(MSEset) std(MSEset)];

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
