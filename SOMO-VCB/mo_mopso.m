% This function implements the MOPSO algorithm for multi-objective
% optimization, which was originally proposed in:
% C.A. Coello Coello, G.T. Pulido, M.S. Lechuga, 
% "Handling Multiple Objectives with Particle Swarm Optimization,"
% IEEE Transactions on Evolutionary Computation, 8(3), 256-279, 2004.
%
% The current file is self-contained, i.e., it includes all the
% necessary functions of the MOPSO algorithm.
%
% The initial population is the input of the algorithm, while
% as output, the algorithm returns the structure "ParetoSet," 
% containing the following information:
% ParetoSet.crs  : Number of non-dominated Pareto solutions 
%                  that have been found during the search.
% ParetoSet.sol  : Pareto set solutions.
% ParetoSet.fsol : Pareto set values.
        
function ParetoSet = mo_mopso(iniPop)

% GLOBAL PARAMETERS

% Check the main program for an explanation of each global parameter.
global trainingSet; 
global neuralModel;
global paramVCB;
global runCount;
global paramMO;
global evalStrategy;

% SET BOUNDARY VECTORS

% Xmin: This vector contains the lower bounds of all dimensions.
% Xmax: This vector contains the upper bounds of all dimensions.
Xmin = zeros(1,paramMO.dim);
Xmax = zeros(1,paramMO.dim);

% Bounds for the bias w0.
Xmin(1) = neuralModel.wRange(1);
Xmax(1) = neuralModel.wRange(2);

% Bounds for all neuron parameters.
for k=1:neuralModel.nn
    % Index offset of the k-th neuron in the parameter vector.
    os = (k-1)*(trainingSet.dim+2)+1; 
    % Lower bound of the weight w_k of the k-th neuron.
    Xmin(os+1) = neuralModel.wRange(1); 
    % Upper bound of the weight w_k of the k-th neuron.
    Xmax(os+1) = neuralModel.wRange(2); 
    % Lower bound of the standard deviation s_k of the k-th neuron.
    Xmin(os+2) = neuralModel.sRange(1);
    % Upper bound of the standard deviation s_k of the k-th neuron.
    Xmax(os+2) = neuralModel.sRange(2);
    % Lower bound of the mean vector mu_k of the k-th neuron.
    Xmin(os+3:os+2+trainingSet.dim) = neuralModel.muRange(1);
    % Upper bound of the mean vector mu_k of the k-th neuron.
    Xmax(os+3:os+2+trainingSet.dim) = neuralModel.muRange(2);
end

% Vmax: This vector contains the maximum absolute velocity for all dimensions,
% as a fraction of the search space.
Vmax = paramMO.maxv * (Xmax-Xmin);

% INITIALIZATION 

% swarm.pos: This matrix contains all particle vectors in rows.
% swarm.fpos: This matrix contains all particle objective vectors in rows.
swarm.pos = zeros(paramMO.ss,paramMO.dim);
swarm.fpos = zeros(paramMO.ss,paramMO.nobj);

% swarm.repos: This matrix contains all repository vectors in rows.
% swarm.frepos: This matrix contains all repository objective vectors in rows.
swarm.repos = zeros(paramMO.rs,paramMO.dim);
swarm.frepos = zeros(paramMO.rs,paramMO.nobj);

% swarm.vel: This matrix contains all velocity vectors in rows.
swarm.vel = zeros(paramMO.ss,paramMO.dim);

% Network evaluations so far in the current VCB run.
nfev = runCount.eval;

% Available evaluations for the current MOPSO run.
optiFev = (paramVCB.MAXeval-nfev)/(paramVCB.MAXcycles-runCount.cycles+1);

% Initialize all particles and their best positions.
for i=1:paramMO.ss
	swarm.pos(i,:) = Xmin + rand(1,paramMO.dim).*(Xmax-Xmin);
    swarm.fpos(i,:) = evalObjective(swarm.pos(i,:));
end
    
% swarm.bpos   : This matrix contains all the best positions in rows.
% swarm.fbpos  : This vector contains all best positions objective vectors in rows.
swarm.bpos = swarm.pos;
swarm.fbpos = swarm.fpos;

% UPDATE REPOSITORY

% Check for domination among the initial particles. 
% The returned vector contains a "1" at the j-th position 
% if at least another particle, dominate the j-th particle 
% otherwise, it contains a "0". 
% Thus, the positions marked by "0" in the "dominatedIndices" vector 
% indicate non-dominated particles in the initial swarm.
dominatedIndices = dominationCheck(swarm.fpos);

% Put the non-dominated particles in the repository.
nd = find(dominatedIndices == 0);
swarm.repos = swarm.pos(nd,:);
swarm.frepos = swarm.fpos(nd,:);

% Current repository size (number of vectors).
swarm.crs = length(nd);

% UPDATE GRID

% Determine the hypercubes and calculate their cardinality. The relevant
% information is stored in the "swarm" structure. 
swarm = gridManipulation(swarm);

% START MOPSO ITERATIONS

% Stopping flag.
stop = 0;
% Iterations counter.
iter = 0;
% Number of consecutive iterations without repository improvement.
noRepImp = 0;

% Main loop
while (stop == 0)
    
    % Update iteration counter.
    iter = iter + 1;
        
    % UPDATE PARTICLES
    
    % Select an attractor from the current repository.
    gbest = reposAttractor(swarm);
    
    % Update velocities and particle positions.
    for i=1:paramMO.ss
        
        % Update velocity.
        swarm.vel(i,:) = paramMO.W*swarm.vel(i,:) + ...
                         paramMO.C1*rand(1,paramMO.dim).*(swarm.bpos(i,:)-swarm.pos(i,:)) + ...
                         paramMO.C2*rand(1,paramMO.dim).*(swarm.repos(gbest,:)-swarm.pos(i,:));
        
        % Restrict velocity.
        violComp = find(swarm.vel(i,:) > Vmax);
        swarm.vel(i,violComp) = Vmax(violComp);
        violComp = find(swarm.vel(i,:) < -Vmax);
        swarm.vel(i,violComp) = -Vmax(violComp);
        
        % Update particle.
        swarm.pos(i,:) = swarm.pos(i,:) + swarm.vel(i,:);
        
        % Restrict particle.
        violComp = find(swarm.pos(i,:) > Xmax);
        swarm.pos(i,violComp) = Xmax(violComp);
        violComp = find(swarm.pos(i,:) < Xmin);
        swarm.pos(i,violComp) = Xmin(violComp);
        
    end
        
    % Mutate particles.
    swarm = particleMutation(swarm,iter,Xmin,Xmax);
        
    % EVALUATE PARTICLES AND UPDATE BEST POSITIONS
    
    % Particles evaluation.
    for i=1:paramMO.ss
        swarm.fpos(i,:) = evalObjective(swarm.pos(i,:));
    end
    
    % Update best positions.
    swarm = updateBestPos(swarm);
    
    % UPDATE REPOSITORY
    
    % Keep old repository.
    oldRep = swarm.repos;
    % Update repository.
    swarm = updateRepository(swarm);
    % Check if new information has entered the repository.
    if ((size(oldRep,1) == size(swarm.repos,1)) && (isempty(find(oldRep~=swarm.repos,1)) == 1))
        noRepImp = noRepImp + 1;
    else
        noRepImp = 0;
    end
        
    % TERMINATION CONDITION
    
    % First, we check if the number of network evaluations has already
    % exceeded its maximum value.
    if (runCount.eval >= paramVCB.MAXeval)
        stop = 1;
    % Then, we check if a fixed number of VCB cycles is determined by the
    % user. This implies a specific number of network evaluations per
    % cycle.
    elseif (paramVCB.MAXcycles > 0)
        % Check if the stopping condition is fulfilled.
        if ((evalStrategy <= 0) || (evalStrategy >= swarm.crs))
            aux = swarm.crs;
        else
            aux = evalStrategy;
        end
        if (runCount.eval-nfev >= optiFev-aux*trainingSet.size)
            % Maximum current run fevals reached and MOPSO will stop
            stop = 1;
        % Otherwise, a restart may be needed (checked later).
        else
            stop = 2;
        end
    % If no fixed number of VCB cycles has been set by the user, the
    % algorithm terminates on its stopping conditions, i.e., it has
    % exceeded a specific number of "paramMO.maxit" iterations or the
    % repository has not been changed for "paramMO.repNoImp" consecutive
    % iterations.
    else
        % Check termination condition.
        if (iter >= paramMO.maxit)
            % Maximum current run iters reached and MOPSO will stop 
            stop = 1;
        elseif (noRepImp >= paramMO.repNoImp)
            % There was no improvement in repository for noRepImp iterations and MOPSO will stop
            stop = 1;
        end
    end
    
    % CHECK RESTART CONDITION
    
    if (stop == 2)
        % Check the restarting condition (no improvement of the repository).
        if (noRepImp >= paramMO.repNoImp)
            % MOPSO will restart because repository non-improving iterations are noRepImp
            % Initialize all particles and their best positions.
            for i=1:paramMO.ss
                swarm.pos(i,:) = Xmin + rand(1,paramMO.dim).*(Xmax-Xmin);
                swarm.fpos(i,:) = evalObjective(swarm.pos(i,:));
            end
            swarm.bpos = swarm.pos;
            swarm.fbpos = swarm.fpos;
            % Reset velocities.
            swarm.vel = zeros(paramMO.ss,paramMO.dim);
            % Update repository and grid.
            swarm = updateRepository(swarm);
            swarm = gridManipulation(swarm);
            % Reset counter of non-improving iterations.
            noRepImp = 0;
        end
        % MOPSO will not restart
        % Reset stopping flag.
        if ((evalStrategy <= 0) || (evalStrategy >= swarm.crs))
            aux = swarm.crs;
        else
            aux = evalStrategy;
        end
        if ((runCount.eval-nfev >= optiFev-aux*trainingSet.size) || (runCount.eval >= paramVCB.MAXeval))
            % MOPSO stops
            stop = 1;
        else
            % MOPSO will continue to new iteration
            stop = 0;
        end
    end
    
end

% SET OUTPUT MATRICES

ParetoSet.sol = swarm.repos;
ParetoSet.fsol = swarm.frepos;
ParetoSet.crs = swarm.crs;

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function calls the implemented neural network to evaluate the
% current mini-batches set. 
function fvec = evalObjective(x)

fvec = mo_RBF_set(x);

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function updates the best positions of the particles after their move
% to new positions. A new position replaces a current best position if
% its corresponding objective vector dominates that of the best position. 
% If the best positions objective vector dominates that of the current
% particle, it remains unchanged. If the two objective vectors are
% non-dominated, one is selected randomly.
function swarm = updateBestPos(swarm)

global paramMO;

for i=1:paramMO.ss
    % Check if the current particle objective vector dominates the best position 
    % objective vector.
    dom = vectorDomination(swarm.fpos(i,:),swarm.fbpos(i,:));
    % Current particle objective vector dominates its best position objective 
    % vector, and the particle replaces the best position.
    if (dom == 1)
        swarm.fbpos(i,:) = swarm.fpos(i,:);
        swarm.bpos(i,:) = swarm.pos(i,:);
    else
        % Check if the best position objective vector dominates the current particle
        % objective vector. In this case, no action is taken.
        dom = vectorDomination(swarm.fbpos(i,:),swarm.fpos(i,:));
        % The current particle and best position objective vectors are 
        % non-dominated. One of them is selected at random.
        if (dom == 0)
            if (rand <= 0.5)
                swarm.fbpos(i,:) = swarm.fpos(i,:);
                swarm.bpos(i,:) = swarm.pos(i,:);
            end
        end
    end
end
    
end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function updates the repository with new non-dominated particles. In
% order to retain the limited size of the repository, particles in less
% crowded positions in the objective space are preferred.
function swarm = updateRepository(swarm)

global paramMO;

% Merge the current swarm and repository in an augmented repository.
augmentRep = [swarm.repos; swarm.pos];
augmentFRep = [swarm.frepos; swarm.fpos];

% Find the dominated vectors in the augmented repository.
dominatedIndices = dominationCheck(augmentFRep);

% Retain only the non-dominated vectors in the actual repository.
swarm.repos = augmentRep(~dominatedIndices,:);
swarm.frepos = augmentFRep(~dominatedIndices,:);
swarm.crs = size(swarm.repos,1);

% Update the hypercubes.
swarm = gridManipulation(swarm);

% Constrain the size of the new repository by removing vectors from the most
% crowded hypercubes.
while (swarm.crs > paramMO.rs)
    % Find the most crowded hypercube.
    [freq,hcub] = hist(swarm.repuni,unique(swarm.repuni));
    % ID of the most crowded hypercube.
    [~,maxid] = max(freq);
    id = hcub(maxid);
    % Find objective vectors belonging in the most crowded hypercube.
    q = find(swarm.repuni == id);
    % Select at random one of the vectors.
    qi = randi(length(q));
    % Remove the selected vector from the repository.
    ii = ones(swarm.crs,1);
    ii(q(qi)) = 0;
    swarm.repos = swarm.repos(ii==1,:);
    swarm.frepos = swarm.frepos(ii==1,:);
    swarm.crs = size(swarm.repos,1);
    % Update the hypercubes.
    swarm = gridManipulation(swarm);
end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function compares two vectors "fv1" and "fv2" and returns a "1" if
% "fv1" dominates "fv2" or a "0" otherwise.
function dom = vectorDomination(fv1,fv2)

global paramMO;

% The vector "fv1" dominates "fv2" if all its components are smaller or 
% equal to the corresponding components of "fv2", and at least one of its
% components is strictly smaller than the corresponding component of "fv2".
if ((sum(fv1<=fv2) == paramMO.nobj) && (sum(fv1<fv2) > 0))
    dom = 1;
else
    dom = 0;
end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function checks the rows of the input matrix "fvec" for domination.
% Each row of the matrix is supposed to be the objective vector of 
% a particle. The output is the vector "dominatedIndices" which contains a 
% "1" at the i-th position if the i-th row of "fvec" is dominated by 
% another row; otherwise, it contains a "0".
function dominatedIndices = dominationCheck(fvec)

global paramMO;

% Initialize domination vector.
dominatedIndices = zeros(1,paramMO.ss);

% Loop on particles.
for i=1:paramMO.ss-1
    for j=i+1:paramMO.ss
        % Check if the i-th objective vector dominates the j-th objective 
        % vector.
        d = vectorDomination(fvec(i,:),fvec(j,:));
        if (d == 1)
            % The j-th objective vector is marked as dominated.
            dominatedIndices(j) = 1;
        else
            % Check if the j-th objective vector dominates the i-th 
            % objective vector.
            d = vectorDomination(fvec(j,:),fvec(i,:));
            % The i-th objective vector is marked as dominated.
            if (d == 1)
                dominatedIndices(i) = 1;
            end
        end
    end
end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function updates hypercubes maintained by MOPSO in order to
% identify crowded regions of the objective space.
function swarm = gridManipulation(swarm)

global paramMO;

% OBJECTIVE SPACE DIVISION

% Divide each objective range in "paramMO.grid" intervals.
for i=1:paramMO.nobj
    mini = min(swarm.frepos(:,i));
    maxi = max(swarm.frepos(:,i));
    step = (maxi-mini)/paramMO.grid;
    swarm.hcubeBounds(i,:) = mini + step*(0:paramMO.grid);
end

% ASSIGN REPOSITORY VECTORS TO HYPERCUBES

% Reset matrices.
swarm.repbox = zeros(swarm.crs,2);
swarm.repuni = zeros(swarm.crs,1);

% Loop on the repository vectors.
for i=1:swarm.crs
    % Find hypercube coordinates.
    aux1 = find(swarm.frepos(i,1)>=swarm.hcubeBounds(1,:),1,'last');
    if (aux1 == paramMO.grid+1)
        aux1 = paramMO.grid;
    end
    aux2 = find(swarm.frepos(i,2)>=swarm.hcubeBounds(2,:),1,'last');
    if (aux2 == paramMO.grid+1)
        aux2 = paramMO.grid;
    end
    % Hypercube position where current repository objective vector lies.
    boxID = [aux1 aux2];
    swarm.repbox(i,:) = boxID;
    % Calculate a unique box ID for this hypercube.
    swarm.repuni(i) = (aux1-1)*paramMO.grid + aux2;
end

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function selects an attractor vector from the repository, which will
% be used to update the velocity of each particle. The selection is
% probabilistic and based on a roulette-wheel procedure. The fitness
% of each hypercube of the grid is determined by its cardinality such that
% less crowded hypercubes receive higher selection probability.
function attractorID = reposAttractor(swarm)
    
% Find non-empty hypercube IDs.
hID = unique(swarm.repuni);

% Calculate the selection probability of each non-empty hypercube.
selP = zeros(1,length(hID));
for i=1:length(hID)
    % Get a non-empty hypercube ID.
    id = hID(i);
    % Get the number of vectors in this hypercube.
    card = sum(swarm.repuni == id);
    % Calculate the selection probability (it follows the form proposed in
    % the original MOPSO reference [1]).
    selP(i) = 10.0/card;
end
selP = selP./sum(selP);

% ROULETTE WHEEL

% Initialize the cumulative probability.
s = 0;

% Get a random number.
R = rand;

% Apply roulette wheel selection.
for i=1:length(selP)
    s = s + selP(i);
    if (R <= s)
        % Select the current hypercube.
        hcubeID = i;
        break;
    end
end

% GET THE ATTRACTOR 

% Selected hypercube's ID.
id = hID(hcubeID);

% Find repository objective vectors that belong to this hypercube.
q = find(swarm.repuni == id);

% Select one of these vectors at random.
attractorID = q(randi(length(q)));

end

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% This function mutates the particles. The mutation aims to increase the
% diversity of the swarm and takes into consideration the current
% optimization phase (iteration) and the user-defined mutation
% parameter according to the original MOPSO reference (Scheme 1). 
% An alternative mutation probability scheme (Scheme 2) that decreases 
% the probability less rapidly is also given.
function swarm = particleMutation(swarm,iter,Xmin,Xmax)

global paramMO;

% Calculate mutation probability.
% Scheme 1 (original MOPSO). Uncomment below to use.
mutp = (1-iter/paramMO.maxit)^(5/paramMO.mutp);
% Scheme 2 (slower decrease). Uncomment below to use.
%mutp = (1-iter/paramMO.maxit)^4;

% Loop on particles.
for i=1:paramMO.ss
    % The i-th particle is selected for mutation.
    if (rand <= mutp)
        % Select at random the dimensions to mutate.
        rdim = randperm(paramMO.dim);
        ndim = randi(length(rdim));
        rdim = rdim(1:ndim);
        for j=1:ndim
            % Mutate the k-th dimension.
            k = rdim(j);
            mutrange = (Xmax(k)-Xmin(k))*mutp;
            swarm.pos(i,k) = swarm.pos(i,k) + mutrange*(2*rand-1);
            if (swarm.pos(i,k) > Xmax(k))
                swarm.pos(i,k) = Xmax(k);
            elseif (swarm.pos(i,k) < Xmin(k))
                swarm.pos(i,k) = Xmin(k);
            end
        end
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
