> **SOMO-VCB: A Matlab software for single-objective and multi-objective optimization for variance counterbalancing in stochastic learning.**
- Dimitra G. Triantali (d.triantali@uoi.gr)
- Konstantinos E. Parsopoulos (kostasp@uoi.gr)
- Isaac E. Lagaris (lagaris@uoi.gr)

##### Keywords
---
- Variance counterbalancing
- Stochastic learning 
- Neural networks
- Single-objective optimization
- Multi-objective optimization
- Matlab

##### Introduction
---

Variance counterbalancing (VCB) was proposed as a neural network training method employing randomly selected mini-batches. Having assumed that the network's parameters we want to optimize are collectively included in a vector $$w \in W$$, where $$W$$ is an appropriate domain, the VCB approach is based on the minimization of the average mean squared error $$\bar{E}(w)$$ of the network along with the error variance $$\bar{\sigma}^2(w)$$ over random sets of mini-batches. The VCB is a remedy for the slow convergence in stochastic learning, eliminating the need to reduce the steps' size and promoting the employment of efficient optimizers. 

The underlying optimization problem can be handled either as a single-objective or multi-objective. The proposed software implements both VCB approaches. The first approach employs a penalty function $$\lambda_{c}$$ and solves the problem 

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$$\min_{w \in W} \bar{F}(w,\lambda_{c}) = \bar{E}(w) + \lambda_{c} \,\, \bar{\sigma}^2(w),$$ &emsp;&emsp; (1) 

With the BFGS algorithm with strong Wolfe-Powell line search conditions. The second approach utilizes the inherent bi-objective form of the problem. It uses the state-of-the-art multi-objective particle swarm optimization (MOPSO) method to optimize the vectorial objective function:

&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;&emsp;$$\bar{F}(w) = \left[ \bar{E}(w) \,\,\, \bar{\sigma}^2(w) \right]^T.\,$$ &emsp;&emsp;&emsp;&emsp;&emsp; (2)

Both VCB approaches are demonstrated in function approximation tasks using RBF networks.

##### Software
---

The proposed software is placed in the [SOMO-VCB](https://app.grammarly.com/ddocs/1896222669) folder. The folder contains two groups of m-files. The first group implements the single-objective approach and comprises the files with the "so" prefix. The second group implements the multi-objective strategy and contains the files with the "mo" prefix. The details of the software's files and folders are comprised in the included [README.pdf](https://app.grammarly.com/ddocs/1896222669) file. An additional folder named [Example](https://app.grammarly.com/ddocs/1896222669) is provided to make it easier for the user to understand the software through an illustrative example.

##### Licence
---

[MIT License](https://app.grammarly.com/ddocs/1896222669)