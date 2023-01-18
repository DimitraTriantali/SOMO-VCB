% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%
% "SOMO-VCB: SOMO-VCB: A Matlab software for single-objective and multi-
% objective optimization for variance counterbalancing in stochastic learning."
%
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% This Matlab program implements a multi-objective optimization approach 
% of the Variance CounterBalancing (VCB) algorithm for stochastic large-
% scale learning. VCB was originally introduced in the following work:
%
% Lagari P.L., Tsoukalas, L.H., and Lagaris, I.E., 
% "Variance counterbalancing for stochastic large-scale learning",
% International Journal on Artificial Intelligence Tools, Vol. 29, No. 5,
% World Scientific Publishing Company, 2020.
%
% The implemented optimizer of the VCB algorithm is the Multi-Objective 
% Particle Swarm Optimization (MOPSO). More details about the optimizer 
% and the relevant reference can be found in the "mo_mopso.m" file that is 
% included in the present software package.
%
% The provided software is self-contained, using only basic Matlab 
% functions available in all Matlab distributions. It has been developed 
% and tested on Matlab R2018a running on a Linux system using the 
% kernel 5.8.0-48-generic. Minor modifications in non-essential parts of 
% the code may be needed to run on different platforms or systems.
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% CLEAR MEMORY AND SCREEN

% The Matlab environment is completely cleaned before execution. The screen
% is also cleaned. If external variables from the Matlab environment are
% needed for a specific application, the user shall modify the following
% commands accordingly.
clear variables;
clc;

% INITIALIZE RANDOM NUMBER GENERATOR

% The random number generator is seeded based on the current time. For
% alternatives, please check the "rng" command manual of Matlab.
rng('shuffle');

% GENERAL PARAMETERS

% screenOutput : Set to 'on' in order to write intermediate information on 
%                screen; otherwise, set to 'off' (strictly lowercase).
screenOutput = 'on';

% INPUT FILENAMES

% These are the filenames of the necessary input files:
% file_training: It contains the complete training dataset.
% file_model   : It contains the parameters of the RBF network model.
% file_vcb     : It contains the parameters of the VCB algorithm.
% file_mo      : It contains the parameters of the optimization problem 
%                and the parameters of the MOPSO optimizer.
% The corresponding files are expected to be found in the "data" directory 
% that shall be included in the current working directory. The filenames 
% follow the Linux/Unix directory naming and path conventions. The user 
% may need to modify the paths for use in different operating systems.
file_training = 'data/training_set.txt';
file_model = 'data/model_data.txt';
file_vcb = 'data/vcb_data.txt';
file_mo = 'data/mo_data.txt';

% OUTPUT FILENAMES

% These are the filenames of the output files: 
% file_log      : It is used for storing all input parameter values to 
%                 facilitate verification of their values.
% file_report   : It contains all output information (one row per experiment) 
%                 except for the actual solution vector. Each row has the 
%                 following form: 
% (Experiment ID)-(Solution value)-(VCB cycles)-(Network evaluations)-(Elapsed time)
% file_solution : It contains the actual solutions vectors (one-row per 
%                 experiment) in the form: (Experiment ID)-(Solution vector)
% The corresponding files are expected to be stored in the "results" 
% directory that shall exist in the current working directory.
file_log = 'results/mo_log';
file_report = 'results/mo_report';
file_solution = 'results/mo_solution';

% GLOBAL VARIABLES

% scrFlag : This flag is automatically set to 1 if screenOutput is 'on' 
%           otherwise, it is set to 0.
global scrFlag;
scrFlag = strcmp(screenOutput,'on');

% trainingSet : This structure contains the complete training 
% dataset. It is read from the file "data/training_set.txt" where the 
% patterns appear in rows in a (pattern vector)-(value) format.
global trainingSet; 

% neuralModel : This is a structure that contains the details of the RBF 
% network model. The data is read from the file "data/model_data.txt" where 
% the corresponding parameters of the network are given, one per line.
global neuralModel;

% paramVCB : This is a structure that contains the parameters of the VCB 
% algorithm. The data is read from the file "data/vcb_data.txt" where the
% corresponding parameters of the VCB algorithm are given, one per line.
global paramVCB;

% paramMO : This is a structure that contains parameters of the MO problem 
% and the employed MOPSO algorithm, such as the number of objective functions, 
% the number of independent experiments, the parameters of the MOPSO 
% optimizer, the size of the swarm vectors, etc.
global paramMO;

% miniBatch : This matrix contains the training pattern indices 
% of each mini-batch for each VCB cycle. Each row of the matrix corresponds
% to a mini-batch and contains the indices of its constituent training 
% patterns from the training dataset.
global miniBatch;

% runCount : This is a structure that contains the counters relevant to the 
% current run of the algorithm.
global runCount;

% evalStrategy : This variable defines the number of solutions from the 
% received Pareto set (provided by MOPSO) that will be eventually evaluated
% according to the total Mean Squared Error (MSE) over the whole training 
% set at the end of each VCB cycle. These solutions compete with the best 
% solution found so far by the algorithm in the current experiment. 
% The possible values of the strategy parameter are as follows:
% N <= 0 : All the detected Pareto set solutions are evaluated.
% N > 0  : N solutions of the Pareto set (up to its current size) are 
%          selected at random and evaluated.
% Note that evaluating many solutions can rapidly consume the available 
% computational budget in terms of single-pattern neural network 
% evaluations.
global evalStrategy;

% READ TRAINING DATASET

% trainingSet.patterns : This is the complete matrix of the training set 
%                        containing the training patterns in rows. The last 
%                        value of each row is the correct value of the 
%                        corresponding pattern vector.
% trainingSet.size     : Number of pattern vectors in the training set.
% trainingSet.dim      : Dimension of the pattern vectors.
% The maximum possible size of the training set that can be loaded is
% directly related to the specific system configuration of the user
% (available RAM memory). In the case of huge datasets, it may be impossible to
% load all training vectors. In such cases, the user may need to modify the
% source code such that each pattern vector is loaded directly at the point 
% where it is used. It is expected that such changes will significantly 
% increase the execution time due to heavy I/O file operations.
trainingSet.patterns = load(file_training);
[trainingSet.size, trainingSet.dim] = size(trainingSet.patterns(:,1:end-1));

% READ NEURAL NETWORK MODEL (RBF) PARAMETERS

% neuralModel.nn      : Number of neurons of the neural network.
% neuralModel.muRange : Range of the mu parameter (mean vector).
% neuralModel.sRange  : Range of the sigma parameter (standard deviation).
% neuralModel.wRange  : Range of the network weights.
aux = load(file_model);
if ((aux(4) <= 0) || (aux(5) <= 0))
    error('< ERROR > Negative standard deviation value is assigned to the RBF network.');
end
neuralModel.nn = aux(1);
neuralModel.muRange = [aux(2) aux(3)];
neuralModel.sRange = [aux(4) aux(5)];
neuralModel.wRange = [aux(6) aux(7)];
clear aux;

% READ VCB PARAMETERS

% paramVCB.nbat      : Number of mini-batches per cycle of the VCB algorithm.
% paramVCB.npat      : Percentage of patterns used per mini-batch set.
% paramVCB.MAXcycles : Maximum number of cycles of the VCB algorithm.
% paramVCB.MAXeval   : Maximum number of single-pattern evaluations of the 
%                      neural network model.
% The maximum number of cycles ("paramVCB.MAXcycles") is used as a flag to
% discriminate between termination condition scenarios of the optimizer.
% More specifically, if paramVCB.MAXcycles > 0, then the VCB algorithm will
% conduct exactly this number of cycles. At each cycle, the optimizer will
% run until it exceeds the specified number of network evaluations for this 
% cycle, which is determined as paramVCB.MAXeval/paramVCB.MAXcycles. On the 
% other hand, if paramVCB.MAXcycles <= 0, then the optimizer at each cycle 
% is applied until an algorithm-related termination condition holds, e.g., 
% the maximum number of iterations is exceeded.
aux = load(file_vcb);
paramVCB.nbat = aux(1);
paramVCB.npat = floor((aux(2)*trainingSet.size)/paramVCB.nbat);
paramVCB.MAXcycles = aux(3);
paramVCB.MAXeval = aux(4);
clear aux;

% READ MOPSO AND PROBLEM PARAMETERS

% paramMO.nobj     : Number of objective functions.
% paramMO.nexp     : Number of experiments (runs) to be performed using MOPSO.
% paramMO.dim      : Length of the PSO particles (dimension of the problem).
% paramMO.ss       : Swarm size.
% paramMO.rs       : Repository size.
% paramMO.maxit    : Maximum number of iterations per MOPSO call.
% paramMO.repNoImp : Maximum number of iterations with no repository
%                    change (restarting condition).
% paramMO.W/C1/C2  : MOPSO velocity parameters W, C1, C2.
% paramMO.grid     : Number of grid intervals per dimension (hypercubes).
% paramMO.maxv     : Maximum velocity as a percentage of the search range.
% paramMO.mutp     : Mutation rate.
% evalStrategy     : Pareto set evaluation strategy.
% The maximum number of iterations of the optimizer "paramSO.maxit", as 
% well as the function value improvement tolerance "paramSO.eps" and the 
% gradient norm tolerance "paramSO.geps" are used as termination conditions
% of the optimizer in case the user does not specify a maximum number of 
% VCB cycles (in "paramVCB.MAXcycles").
aux = load(file_mo);
paramMO.nobj = aux(1);
paramMO.nexp = aux(2);
paramMO.ss = aux(3);
paramMO.rs = aux(4);
paramMO.maxit = aux(5);
paramMO.repNoImp = aux(6);
paramMO.W = aux(7);
paramMO.C1 = aux(8);
paramMO.C2 = aux(9);
paramMO.grid = aux(10);
paramMO.maxv = aux(11);
paramMO.mutp = aux(12);
evalStrategy = aux(13);
clear aux;

% For each neuron, the decision variables that need to be optimized are:
% 1. The neuron weight.
% 2. The mean vector of length equal to the dimension of the pattern vectors.
% 3. The positive standard deviation.
% In addition, there is a bias weight for the network. Thus, the total
% number of decision variables and, therefore, the dimension of the 
% optimization problem is as follows:
paramMO.dim = neuralModel.nn*(2+trainingSet.dim)+1;

% PREPARE OUTPUT FILES

% All parameter values are stored in the output file determined by the 
% "file_log" filename in order to facilitate the verification of their 
% proper values.
mo_write_log(file_log);

% Prepare the solution output file.
fout = fopen(file_solution,'w');
if (fout ~= -1)
    fclose(fout);
else
    error('< ERROR > Cannot write solution file %s. \n',file_solution);
end

% Prepare the report output file.
fout = fopen(file_report,'w');
if (fout ~= -1)
    fclose(fout);
else
    error('< ERROR > Cannot write report file %s. \n',file_report);
end

% WRITE SCREEN INFORMATION

fprintf('-------------------------------------------------------\n');
fprintf('MO-VCB ALGORITHM \n');
fprintf('All data has been read and output files are ready. \n');
fprintf('Proceeding to experiments: \n\n');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% LOOP ON THE NUMBER OF EXPERIMENTS
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

for i_exp = 1:paramMO.nexp
    
    % WRITE SCREEN INFORMATION
    
    fprintf('Running Experiment %d... \n',i_exp);
    
    % INITIALIZE RUN COUNTERS
    
    % runCount.cycles : Counter of VCB cycles.
    % runCount.stop   : Stopping condition flag (1=stop / 0=continue).
    % runCount.Tstart : Start-time of the current experiment.
    % runCount.Tend   : End-time of the current experiment.
    % runCount.xbest  : Best solution vector of the current run.
    % runCount.fbest  : Best solution value of current run.
    % runCount.eval   : Counter of single-pattern network evaluations.
    runCount.cycles = 0; 
    runCount.stop = 0;
    runCount.Tstart = tic;
    runCount.Tend = 0;
    runCount.xbest = [];
    runCount.fbest = Inf;
    runCount.eval = 0;

    % Initial population for MOPSO
    iniPop = [];                                                            

    % START NEW VCB CYCLE
    
    while (runCount.stop == 0)
        
        % Update counter of VCB cycles.
        runCount.cycles = runCount.cycles + 1;
    
        % Write screen information (current cycle number and percentage).
        if (scrFlag)
            fprintf('\t Cycle %d ... ',runCount.cycles);
        end
        
        % DETERMINE MINI-BATCHES

        % The mini-batches are pre-determined for the current VCB cycle. 
        % They are stored in a matrix of "paramVCB.nbat" rows, each one
        % containing a number of "paramVCB.npat" indices of training patterns 
        % that comprise the specific mini-batch. The indices of the patterns
        % are their corresponding row indices in the "trainingSet.patterns" 
        % matrix. The training patterns of each mini-batch are selected at
        % random from the training dataset.
        miniBatch = randi(trainingSet.size,[paramVCB.nbat paramVCB.npat]);
        
        % CALL MOPSO OPTIMIZER
        
        % The MOPSO optimizer is called at this point. 
        % The output of the optimizer is the structure "ParetoSet",
        % containing the following data:
        % ParetoSet.crs  : Number of Pareto set solutions.
        % ParetoSet.sol  : Pareto set solutions.
        % ParetoSet.fsol : Pareto set values.
        ParetoSet = mo_mopso(iniPop);
        
        iniPop = ParetoSet;
        
        % Write screen information (Pareto set size).
        if (scrFlag)
            fprintf('Pareto.Size=%d ... ',ParetoSet.crs);
        end
        
        % UPDATE BEST SOLUTION
        
        % The best solution of the current run is updated according to the
        % selected strategy as determined in the global variable "evalStrategy".
        mo_update_best(ParetoSet);
        
        % Write screen information (best solution so far and evaluations).
        if (scrFlag)
            fprintf('f*=%.10f ... NNeval=%d/%d (%4.1f%%) \n',runCount.fbest,runCount.eval,paramVCB.MAXeval,100*runCount.eval/paramVCB.MAXeval);
        end
        
        % CHECK STOPPING CONDITION
        
        % Two stopping conditions are implemented (whatever comes first): 
        % 1. Reaching the maximum number of VCB cycles (determined in 
        %    parameter "paramVCB.MAXcycles"). 
        %    - OR -
        % 2. Reaching the maximum number of single-pattern evaluations of 
        %    the neural network (determined in parameter "paramVCB.MAXeval").
        if ((runCount.cycles == paramVCB.MAXcycles) || (runCount.eval >= paramVCB.MAXeval))
            runCount.stop = 1;
        end
                
    end
    
    % END OF VCB CYCLE
    
    % WRITE OUTPUT
    
    % Calculate elapsed time (in seconds).
    runCount.Tend = toc(runCount.Tstart);
    
    % Write screen information.
    fprintf('...finished (elapsed time: %.2f sec). \n\n',runCount.Tend);
    
    % Write solution file.
    fout = fopen(file_solution,'a');
    if (fout ~= -1)
        % First write the current experiment ID number.
        fprintf(fout,'%d  ',i_exp);
        % Then write the solution vector.
        for i=1:paramMO.dim
            fprintf(fout,'%15.10f ',runCount.xbest(i));
        end
        % Change line to prepare the file for the next experiment.
        fprintf(fout,'\n');
        fclose(fout);
    else
        error('< ERROR > Cannot write solution file %s. \n',file_solution);
    end
    
    % Write report file.
    fout = fopen(file_report,'a');
    if (fout ~= -1)
        % Write information in the following order:
        % (Experiment ID)-(Solution value)-(VCB cycles)-(Network evaluations)-(Pareto size)-(Elapsed time)
        fprintf(fout,'%3d %15.10f %10d %10d %10d %10.2f \n',i_exp,runCount.fbest,runCount.cycles,runCount.eval,ParetoSet.crs,runCount.Tend);
        fclose(fout);
    else
        error('< ERROR > Cannot write report file %s. \n',file_report);
    end
    
end
% END LOOP ON EXPERIMENTS

% Write closing messages on the screen.
fprintf('All %d experiments have successfully finished.\n',paramMO.nexp);
fprintf('Best solutions are stored in the "%s" file. \n',file_solution);
fprintf('Function values and run counters are stored in the "%s" file. \n',file_report);
fprintf('Program terminated. \n');
fprintf('-------------------------------------------------------\n\n');

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
