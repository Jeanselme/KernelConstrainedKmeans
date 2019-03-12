# KernelConstrainedKmeans
Simple implementation of a kernel ckmeans

## Project
### Initialization
Contains a farthest first initialization from the heuristic proposed in [Farthest-Point Heuristic based Initialization Methods for K-Modes Clustering](https://arxiv.org/pdf/cs/0610043.pdf) adapted for a kernel metric.

### Kckmeans
Kernel constrained kmeans which takes the initial assignation and returns the best assignation repecting the constrained.  
This idea is inspired by [Constrained K-means Clusetering with Background Knowledge](https://pdfs.semanticscholar.org/0bac/ca0993a3f51649a6bb8dbb093fc8d8481ad4.pdf) with a kernel instead of an euclidean distance. The code has been optimized to not recompute the whole distance but just the impact of moving one point from one cluster to another.

## Remarks
### Constraints
One important point is that we represent constraints as a n by n matrix of values between -1 and 1, where -1 is a must not link and +1 is a must link constraint. All values between is allowed in order to represent the uncertainty of the user.

### Initialization
In this process the initialization is crucial: no cannot link constraint can be broken otherwise the algorithm will returns an error. In the same way if there is no clustering verifying the constraint, the current initialization will return an error.

### Possible improvement
Integrate this code in an active learning framework in order to limit the number of constraints necessary for the convergence of the algorithm.

## Dependencies
Code tested with python 3.5 with numpy and scipy.  
Sklearn and matplotlib necessary for the example.

## Example
A simple example is given in the notebook `Example.ipynb` which explore the constraint clustering with an rbf kernel on [MNIST](http://yann.lecun.com/exdb/mnist/) dataset.
