"""
    Files containing all the procedures in order
    to initialize the cluster assigment at the start
"""
import numpy as np
from scipy.sparse.csgraph import connected_components

class Initialization:
    """
        This object precompute the main components implied by constraints
    """

    def __init__(self, k, constraint):
        """
            Precompute connected components
            Arguments:
                k {int} -- Number of cluster
                constraint {Array n * n} -- Constraint matrix with value in (-1, 1)
                    Positive values are must link constraints
                    Negative values are must not link constraints
        """
        assert constraint is not None, "Farthest initialization cannot be used with no constraint"
        # Computes the most important components and order by importance
        positive = np.where(constraint > 0, constraint, np.zeros_like(constraint))
        self.number, components = connected_components(positive, directed=False)
        unique, count = np.unique(components, return_counts = True)
        order = np.argsort(count)[::-1]
        self.components = np.argsort(unique[order])[components] 
        self.constraint = constraint
        assert self.number >= k, "Constraint too important for number of cluster"
        self.k = k
        
    def farthest_initialization(self, kernel):
        """
            Farthest points that verify constraint

            Arguments:
                kernel {Array n * n} -- Kernel matrix (n * n)
                k {Int} --  Number cluster
                constraint {Array n * n} -- Constraint matrix
        """
        components = self.components.copy()

        # Precompute center distances
        assignation_cluster, intra_distance, intra_number = {}, {}, {}
        for c in range(self.k):
            assignation_cluster[c] = (components == c).reshape((-1,1))
            intra_distance[c] = np.matmul(kernel, assignation_cluster[c])
            intra_number[c] = np.sum(assignation_cluster[c])

        # Merge components respecting constraint until # = k
        for i in range(self.k, self.number):
            # Computes intra distance
            assignation_cluster[i] = (components == i).reshape((-1,1))
            intra_distance[i] = np.matmul(kernel, assignation_cluster[i])
            intra_number[i] = np.sum(assignation_cluster[i])

            # Computes distances to all other cluster 
            # We ignore the last part which depends on the intravariance of the cluster i
            distance = [float(np.dot(assignation_cluster[c].T, intra_distance[c])/(intra_number[c]**2) 
                - 2 * np.dot(assignation_cluster[i].T, intra_distance[c])/(intra_number[c] * intra_number[i]))
                for c in range(self.k)]

            # Closest verifying constraint
            order = np.argsort(distance)

            # If no constraint is positive => Too much constraint
            broken_constraint = self.constraint[:, assignation_cluster[i].flatten()]
            closest = min(order, key=lambda o: np.sum(broken_constraint[(components == o),:] < 0))
            components[assignation_cluster[i].flatten()] = closest

            # Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return components
