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

#### Introduction
---

Variance counterbalancing (VCB) was proposed as a neural network training method employing randomly selected mini-batches [[1]](#1). Having assumed that the network's parameters we want to optimize are collectively included in a vector $w \in W$, where $W$ is an appropriate domain, the VCB approach is based on the minimization of the average mean squared error $\bar{E}(w)$ of the network along with the error variance $\bar{\sigma}^2(w)$ over random sets of mini-batches. The VCB is a remedy for the slow convergence in stochastic learning, eliminating the need to reduce the steps' size and promoting the employment of efficient optimizers. 

The underlying optimization problem can be handled either as a single-objective or multi-objective. The proposed software implements both VCB approaches. The first approach employs a penalty function $\lambda_{c}$ and solves the problem 

$$\min_{w \in W} \bar{F}(w,\lambda_{c}) = \bar{E}(w) + \lambda_{c}  \text{   } \bar{\sigma}^2(w),$$ 

With the BFGS algorithm with strong Wolfe-Powell line search conditions [[2]](#2). The second approach utilizes the inherent bi-objective form of the problem. It uses the state-of-the-art multi-objective particle swarm optimization (MOPSO) method [[3]](#3) to optimize the vectorial objective function:

$$\bar{F}(w) = \left[ \bar{E}(w) \text{            } \text{            } \bar{\sigma}^2(w) \right]^T.$$

Both VCB approaches are demonstrated in function approximation tasks using RBF networks [[4]](#4).

#### Software's guide
---

The proposed software is placed in the [SOMO-VCB](https://github.com/DimitraTriantali/SOMO-VCB/tree/main/SOMO-VCB) folder. The folder contains two groups of m-files. The first group implements the single-objective approach and comprises the files with the "so" prefix. The second group implements the multi-objective strategy and contains the files with the "mo" prefix. The details of the software's files and folders are comprised in the included [README](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/SOMO-VCB/README.pdf) file. An additional file named [Example](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/Example.pdf) is provided to make it easier for the user to understand the software through an illustrative example.

#### Licence
---

[MIT License](https://github.com/DimitraTriantali/SOMO-VCB/blob/main/LICENSE.txt)

#### Acknowledgements
---

This research was supported by the project "Dioni: Computing Infrastructure for Big-Data Processing and Analysis" (MIS No. 5047222), co-funded by European Union (ERDF) and Greece through Operational Program "Competitiveness, Entrepreneurship and Innovation", NSRF 2014-2020.

#### References
---
<a id="1">[1]</a> Lagari, P. L., Tsoukalas, L. H., & Lagaris, I. E. (2020). Variance Counterbalancing for Stochastic Large-scale Learning. In International Journal on Artificial Intelligence Tools (Vol. 29, Issue 05, p. 2050010). World Scientific Pub Co Pte Lt. https://doi.org/10.1142/s0218213020500104

<a id="2">[2]</a> Fletcher R. (1987). Practical methods of optimization (2nd ed.). Wiley.

<a id="3">[3]</a> C. A. C. Coello, G. T. Pulido and M. S. Lechuga, "Handling multiple objectives with particle swarm optimization," in IEEE Transactions on Evolutionary Computation, vol. 8, no. 3, pp. 256-279, June 2004, doi: 10.1109/TEVC.2004.826067.

<a id="4">[4]</a> Broomhead, David & Lowe, David. (1988). Radial basis functions, multi-variable functional interpolation and adaptive networks. ROYAL SIGNALS AND RADAR ESTABLISHMENT MALVERN (UNITED KINGDOM). RSRE-MEMO-4148. 
