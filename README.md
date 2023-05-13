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
Artificial neural networks are becoming increasingly important in engineering as they have proven helpful in various demanding applications. The Variance counterbalancing (VCB) method was proposed as a neural network training approach employing randomly selected mini-batches to optimize the network's parameters $w \in W$, where $W$ is an appropriate domain [[1]](#1). This method is based on the minimization of the average mean squared error $\bar{E}(w)$ of the network along with the error variance $\bar{\sigma}^2(w)$ over the mini-batches. The VCB method helps eliminate the need to reduce the size of the steps and promotes the use of efficient optimizers, making it a remedy for slow convergence in stochastic learning. 

The optimization problem underlying this technique can be handled as a single-objective or multi-objective. The proposed software implements both VCB approaches in function approximation tasks using RBF networks [[2]](#2).

The first approach (SO) employs a penalty function $\lambda_{c}$ and the BFGS algorithm with strong Wolfe-Powell line search conditions [[3]](#3) to solve the problem:

$$\min_{w \in W} \bar{F}(w,\lambda_{c}) = \bar{E}(w) + \lambda_{c}  \text{   } \bar{\sigma}^2(w),$$

The second approach (MO) utilizes the inherent bi-objective form of the problem. It uses the state-of-the-art multi-objective particle swarm optimization method [[4]](#4) to optimize the vectorial objective function:

$$\bar{F}(w) = \left[ \bar{E}(w) \text{            } \text{            } \bar{\sigma}^2(w) \right]^T.$$

The SOMO-VCB software has been validated by the research of Triantali et al. [[5]](#5), which demonstrated the promising performance of VCB approaches compared to the established Adam method. 

#### Software's guide
---

The proposed software is placed in the [SOMO-VCB](https://github.com/DimitraTriantali/SOMO-VCB/tree/main/SOMO-VCB) folder. The folder contains two groups of m-files. The first group implements the single-objective approach and comprises the files with the "so" prefix. The second group implements the multi-objective strategy and contains the files with the "mo" prefix. The details of the software's files and folders are in the included [README](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/SOMO-VCB/README.pdf) file. An additional file named [Example](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/Example.pdf) is provided to make it easier for the user to understand the software through a step-by-step example.

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
