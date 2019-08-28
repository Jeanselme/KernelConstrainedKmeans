"""
    Files containing all the procedures in order
    to initialize the cluster assigment at the start
"""
import numpy as np
from scipy.sparse import find, coo_matrix
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
        rows, columns, values =  find(constraint)
        selection = values > 0
        rows, columns, values = rows[selection], columns[selection], values[selection]
        positive = coo_matrix((values, (rows, columns)), shape = constraint.shape)

        self.number, components = connected_components(positive, directed=False)
        unique, count = np.unique(components, return_counts = True)
        order = np.argsort(count)[::-1]
        self.components = np.argsort(unique[order])[components]
        self.constraint = constraint
        assert self.number >= k, "Constraint too important for number of cluster"

        if (count > 1).sum() >= k:
            print("Constraints do not allow to find enough connected components for farthest first")

        self.k = k

    def _finalize_assignation(self, ):
        # TODO: Clean assignation of the remaining connected components
        return
        
    def farthest_initialization(self, kernel):
        """
            Farthest points that verify constraint

            Arguments:
                kernel {Array n * n} -- Kernel matrix (n * n)
        """
        # Compute the farthest centers
        assignations = np.full_like(self.components, np.nan)

        # Precompute center distances
        assignation_cluster, intra_distance, intra_number = {}, {}, {}
        for c in range(self.number):
            assignation_cluster[c] = (self.components == c).reshape((-1,1))
            intra_distance[c] = np.matmul(kernel, assignation_cluster[c])
            intra_number[c] = np.sum(assignation_cluster[c])

        ## First assignation is the largest component
        assignations[assignation_cluster[0].flatten()] = 0

        ## Save characteristics of assigned components
        assigned = (assignations == 0).reshape((-1,1))
        assigned_intra_distance = np.matmul(kernel, assigned)
        assigned_intra_number = np.sum(assigned)

        remaining = set(range(1, self.number))

        ## Compute iteratively the farthest given all other
        match = {}
        for i in range(1, self.k):
            # Computes distances to all remaining NON TRIVIAL connected components
            # We ignore the last part which depends on the intravariance of the past clusters
            distance = {c: float(np.dot(assignation_cluster[c].T, intra_distance[c])/(intra_number[c]**2) 
                - 2 * np.dot(assigned.T, intra_distance[c])/(intra_number[c] * assigned_intra_number))
                for c in remaining if intra_number[c] > 1}

            farthest = max(distance, key = lambda x: distance[x])
            assignations[assignation_cluster[farthest].flatten()] = i
            match[farthest] = i

            # Update characteristics of assigned
            assigned += assignation_cluster[farthest]
            assigned_intra_distance += intra_distance[farthest]
            assigned_intra_number += intra_number[farthest]

            # Remove components
            remaining.remove(farthest)

        # Assign each unassigned components
        for i in remaining:
            ## Computes distances to all other cluster 
            ## We ignore the last part which depends on the intravariance of the cluster i
            distance = {c: float(np.dot(assignation_cluster[c].T, intra_distance[c])/(intra_number[c]**2) 
                - 2 * np.dot(assignation_cluster[i].T, intra_distance[c])/(intra_number[c] * intra_number[i]))
                for c in match}

            ## Closest verifying constraint
            order = sorted(distance.keys(), key = lambda k: distance[k])

            ## If no constraint is positive => Too much constraint
            broken_constraint = self.constraint[:, assignation_cluster[i].flatten()]
            closest = min(order, key=lambda o: np.sum(broken_constraint[(assignations == o),:] < 0))
            assignations[assignation_cluster[i].flatten()] = match[closest]

            ## Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return assignations

    def kmeanspp_initialization(self, kernel):
        """
            Kmeans ++ that verify constraint

            Arguments:
                kernel {Array n * n} -- Kernel matrix (n * n)
        """
        # Compute the farthest centers
        assignations = np.full_like(self.components, np.nan)

        return assignations

    def random_initialization(self, kernel = None):
        """
            Random Assignation
        """
        return np.random.choice(self.k, size = len(self.components))
