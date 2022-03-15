## AA 228: Decision Making Under Uncertainty
A Julia implementation of Particle Swarm Optimization for Bayesian network structure learning. 

A brief write-up available here: [AA288_Project1.pdf](./docs/aa228_project1.pdf)

## Project 1: Bayesian Network Structure Learning

### Approach

Learning the optimal Bayesian network structure from a set of unordered node is an NP-hard problem. Rather than searching the space of possible graph structures, the complexity may be reduced by searching the *ordering space* of the nodes. This code utilizes a Particle Swarm Optimization (PSO) based global optimization routine [3] to determine the preferred node order, and the traditional K2 algorithm [1, 2] for learning the optimal structure of the Bayesian network given a specific ordering.

### Project 1 Description

Learn the structure of a Bayesian network that best represents the given data sets. There are 3 data sets:
- small.csv, 8 vars
- medium.csv, 12 vars
- large.csv, 50 vars

The results are stored in the `output` folder:
- Bayesian scores
- .gph graph structure files
- .png graph visualization files

References:

[1] Cooper, G. F., & Herskovits, E. (1992). A Bayesian Method for the Induction of Probabilistic Networks from Data. Machine Learning, 9(4), 309–347. https://doi.org/10.1023/A:1022649401552

[2] Ruiz, C. (2005). Illustration of the K2 algorithm for learning Bayes net structures. WPI Department of Computer Science, 1–7. http://web.cs.wpi.edu/~cs539/s07/Projects/k2_algorithm.pdf

[3] Aouay, S., Jamoussi, S., & Ayed, Y. Ben. (2013). Particle swarm optimization based method for Bayesian Network structure learning. 2013 5th International Conference on Modeling, Simulation and Applied Optimization, ICMSAO 2013, 0–5. https://doi.org/10.1109/ICMSAO.2013.6552569
