> ## **SOMO-VCB: A Matlab software for single-objective and multi-objective optimization for variance counterbalancing in stochastic learning.**
- Dimitra G. Triantali (d.triantali@uoi.gr)
- Konstantinos E. Parsopoulos (kostasp@uoi.gr)
- Isaac E. Lagaris (lagaris@uoi.gr)

#### Keywords
---
- Variance counterbalancing
- Stochastic learning 
- Neural networks
- Single-objective optimization
- Multi-objective optimization
- Matlab

#### Variance counterbalancing
---
Artificial neural networks have been placed in a salient position among machine learning methods for solving challenging problems in science and engineering. The stochastic gradient descent (SGD) methods, which constitute the main optimization artillery for training neural networks, are often characterized by slow convergence rates, especially in large datasets. This is a consequence of considering the mean squared error (MSE) of the network over individual mini-batches randomly selected from the training dataset.

The VCB method was proposed [[1]](#1) as an alternative approach that employs random sets of mini-batches and concurrently minimizes the average and the variance of the network’s MSE. Thus, if $w$ is the parameter vector of the neural network (i.e., synapses weights and other possible parameters of its activation functions), then the average, $\bar{E}(w)$, and the variance, $\bar{\sigma}^2(w)$, of the network’s MSE over randomly selected sets of mini-batches are used to form the objective function for the network’s training. The main gain of this approach is the alleviation of slow convergence through the use of more efficient optimizers, such as quasi-Newton methods, as well as the enhanced generalization capability of the network.

In its early variants, VCB followed a single-objective (SO) optimization approach [[1]](#1). Thus, the optimization problem was formulated through the penalty function:

$$\min_{w \in W} \bar{F}(w,\lambda_{c}) = \bar{E}(w) + \lambda_{c}  \text{   } \bar{\sigma}^2(w),$$

where $λc$ stands for the penalty coefficient that needs to be properly set. The problem was solved using the BFGS algorithm with strong Wolfe-Powell line search conditions [[3]](#3). Recently, the inherent bi-objective nature of VCB has motivated a multi-objective (MO) formulation of the problem [[5]](#5) based on the minimization of the vectorial objective function:

$$\bar{F}(w) = \left[ \bar{E}(w) \text{            } \text{            } \bar{\sigma}^2(w) \right]^T.$$

whose Pareto optimal solutions can be detected through state-of-the-art metaheuristics, such as the multiobjective particle swarm optimization (MOPSO) method [[4]](#4).

The proposed SOMO-VCB software implements both the single-objective and the multi-objective VCB approaches, especially designed for function approximation tasks using RBF neural networks [[2]](#2). The recently published proof-of-concept results in [[5]](#5) verify the feasibility of both approaches as well as their potential in diverse regression tasks.

#### Software's guide
---

The proposed software is placed in the [SOMO-VCB](https://github.com/DimitraTriantali/SOMO-VCB/tree/main/SOMO-VCB) folder. The folder contains two groups of m-files. The first group implements the single-objective approach and comprises the files with the "so" prefix. The second group implements the multi-objective strategy and contains the files with the "mo" prefix. The details of the software's files and folders are in the included [README](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/SOMO-VCB/README.pdf) file. A simple application example that comprehensively describes the workings of the proposed software and can be used as a guide for applying the VCB algorithm to more complicated problems is offered on [Example](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/Example.pdf).

#### Licence
---

[MIT License](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/MIT%20License.txt)

#### Acknowledgements
---

We acknowledge support of this work by the project “Dioni: Computing Infrastructure for Big-Data Processing and Analysis.” (MIS No. 5047222) which is implemented under the Action “Reinforcement of the Research and Innovation Infrastructure”, funded by the Operational Programme "Competitiveness, Entrepreneurship and Innovation" (NSRF 2014-2020) and co-financed by Greece and the European Union (European Regional Development Fund).

#### References
---
<a id="1">[1]</a> Lagari, P. L., Tsoukalas, L. H., & Lagaris, I. E. (2020). Variance Counterbalancing for Stochastic Large-scale Learning. In International Journal on Artificial Intelligence Tools (Vol. 29, Issue 05, p. 2050010). World Scientific Pub Co Pte Lt. https://doi.org/10.1142/s0218213020500104

<a id="2">[2]</a> Broomhead, David & Lowe, David. (1988). Radial basis functions, multi-variable functional interpolation and adaptive networks. ROYAL SIGNALS AND RADAR ESTABLISHMENT MALVERN (UNITED KINGDOM). RSRE-MEMO-4148. 

<a id="3">[3]</a> Fletcher R. (1987). Practical methods of optimization (2nd ed.). Wiley.

<a id="4">[4]</a> C. A. C. Coello, G. T. Pulido and M. S. Lechuga, "Handling multiple objectives with particle swarm optimization," in IEEE Transactions on Evolutionary Computation, vol. 8, no. 3, pp. 256-279, June 2004, doi: 10.1109/TEVC.2004.826067.

<a id="5">[5]</a> Triantali, D. G., Parsopoulos, K. E., & Lagaris, I. E. (2023). Single-objective and multi-objective optimization for variance counterbalancing in stochastic learning. In Applied Soft Computing (Vol. 142, p. 110331). Elsevier BV. https://doi.org/10.1016/j.asoc.2023.110331
