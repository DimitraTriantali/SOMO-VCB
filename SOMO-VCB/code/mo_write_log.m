% This function writes all input parameter values set in the 
% global variables of the main program into the file named according to 
% the "file_log" parameter to facilitate their verification.
function mo_write_log(file_log)

% GLOBAL VARIABLES

% Check the main program for an explanation of each global parameter.
global trainingSet; 
global neuralModel;
global paramVCB;
global paramMO;
global evalStrategy;

% ERROR TRAPS

% Checking the values of various input parameters for possible mistakes or
% unusual settings. Error messages are produced, and the program execution
% is aborted in case of critical errors. Unusual values produce warning
% messages without aborting the program.

if (neuralModel.nn <= 0)
    error('< ERROR > The number of neurons shall be positive: neuralModel.nn=%d',neuralModel.nn);
elseif (neuralModel.nn >= 100)
    warning('The number of neurons is unusually high: neuralModel.nn=%d',neuralModel.nn);
elseif (sum(neuralModel.sRange <= 0) > 0)
    error('< ERROR >  The range of the standard deviation of the network contains nonpositive number(s): neuralModel.sRange=[%f %f]',neuralModel.sRange(1),neuralModel.sRange(2));
elseif (paramVCB.nbat <= 0)
    error('< ERROR > The number of mini-batches in the mini-batch set at each VCB cycle shall be positive: paramVCB.nbat=%d',paramVCB.nbat);
elseif (paramVCB.nbat > trainingSet.size)
    error('< ERROR > The number of mini-batches is higher than the total number of training patterns: paramVCB.nbat=%d > trainingSet.size=%d',paramVCB.nbat,trainingSet.size);
elseif (paramVCB.npat <= 0)
    error('< ERROR > The number of training patterns per mini-batch shall be positive: paramVCB.npat=%d',paramVCB.npat);
elseif (paramVCB.npat > trainingSet.size)
    error('< ERROR > The number of training patterns per mini-batch exceeds the total number of training patterns: paramVCB.npat=%d > trainingSet.size=%d',paramVCB.npat,trainingSet.size);
elseif (paramVCB.MAXeval <= 0)
    error('< ERROR > The total number of single-pattern evaluations of the RBF network shall be positive: paramVCB.MAXeval=%d',paramVCB.MAXeval);
elseif (paramMO.nexp <= 0)
    error('< ERROR > The total number of experiments shall be positive: paramSO.nexp=%d',paramMO.nexp);
elseif (paramMO.nexp > 100)
    warning('The total number of experiments is unusually high: paramSO.nexp=%d',paramMO.nexp);
end

% WRITE PARAMETER VALUES

fout = fopen(file_log,'w');
if (fout ~= -1)
    fprintf(fout,'***** Training Set ***** \n');
    fprintf(fout,'trainingSet.size    = %d [Size of the complete training set] \n',trainingSet.size);
    fprintf(fout,'trainingSet.dim     = %d [Dimension of the training pattern-vectors] \n',trainingSet.dim);
    fprintf(fout,'\n');
    fprintf(fout,'***** RBF Network ***** \n');
    fprintf(fout,'neuralModel.nn      = %d [Number of neurons of the RBF network] \n',neuralModel.nn);
    fprintf(fout,'neuralModel.muRange = [%f %f] [Range for the mu parameter] \n',neuralModel.muRange(1),neuralModel.muRange(2));
    fprintf(fout,'neuralModel.sRange  = [%f %f] [Range for the sigma parameter (st.dev.)] \n',neuralModel.sRange(1),neuralModel.sRange(2));
    fprintf(fout,'neuralModel.wRange  = [%f %f] [Range for the network weights] \n',neuralModel.wRange(1),neuralModel.wRange(2));
    fprintf(fout,'\n');
    fprintf(fout,'***** VCB Algorithm ***** \n');
    fprintf(fout,'paramVCB.nbat       = %d [Number of mini-batches per VCB cycle] \n',paramVCB.nbat);
    fprintf(fout,'paramVCB.npat       = %d [Number of patterns used per mini-batch set] \n',paramVCB.npat);
    fprintf(fout,'paramVCB.cycles     = %d [Number of cycles of the VCB algorithm] \n',paramVCB.MAXcycles);
    fprintf(fout,'paramVCB.NNeval     = %d [Number of single-pattern evaluations of the RBF network] \n',paramVCB.MAXeval);
    fprintf(fout,'\n');
    fprintf(fout,'***** MOPSO and Problem ***** \n');
    fprintf(fout,'paramMO.nobj        = %d [Number of objective functions] \n',paramMO.nobj);
    fprintf(fout,'paramMO.nexp        = %d [Number of experiments] \n',paramMO.nexp);
    fprintf(fout,'paramMO.dim         = %d [Dimension of the optimization problem] \n',paramMO.dim);
    fprintf(fout,'paramMO.ss          = %d [Swarm size] \n',paramMO.ss);
    fprintf(fout,'paramMO.rs          = %d [Repository size] \n',paramMO.rs);
    fprintf(fout,'paramMO.maxit       = %d [Maximum MOPSO iterations] \n',paramMO.maxit);
    fprintf(fout,'paramMO.repNoImp    = %d [Maximum iterations with no repository change] \n',paramMO.repNoImp);
    fprintf(fout,'paramMO.W           = %5.3f [MOPSO inertia weight] \n',paramMO.W);
    fprintf(fout,'paramMO.C1          = %5.2f [MOPSO cognitive parameter] \n',paramMO.C1);
    fprintf(fout,'paramMO.C2          = %5.2f [MOPSO social parameter] \n',paramMO.C2);
    fprintf(fout,'paramMO.grid        = %d [Number of grid points per dimension] \n',paramMO.grid);
    fprintf(fout,'paramMO.maxv        = %f [Maximum velocity as percentage of the search range] \n',paramMO.maxv);
    fprintf(fout,'paramMO.mutp        = %f [Percentage of uniform mutation] \n',paramMO.mutp);
    fprintf(fout,'evalStrategy        = %d [Pareto set evaluation strategy] \n',evalStrategy);
    fclose(fout);
else
    error('< ERROR > Failed to open log file %s.',file_log);
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
