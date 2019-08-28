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

        if (count > 1).sum() < k:
            print("Constraints do not allow to find enough connected components for farthest first ({} for {} classes) => Random forced".format((count > 1).sum(), k))
            self.farthest_initialization = lambda x: None

        self.k = k

    def farthest_initialization(self, kernel):
        """
            Farthest points that verifies constraint

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
            closest = min(distance.keys(), key = lambda k: distance[k])
            assignations[assignation_cluster[i].flatten()] = match[closest]

            ## Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return assignations

    def random_initialization(self):
        """
            Random Assignation
        """
        return np.random.choice(self.k, size = len(self.components))

class Euclidean_Initialization(Initialization):

    @classmethod
    def compute_center(cls, data, assignation):
        """
            Computes euclidean centers given an assignation
            
            Arguments:
                data {[type]} -- [description]
                assignation {[type]} -- [description]
        """
        centers = []
        for i in np.unique(assignation):
            centers.append(data[assignation == i].mean(0))
        return np.vstack(centers)

    def farthest_initialization(self, data):
        """
            Farthest points that verifies constraint

            Arguments:
                data {Array n * d} -- Data
        """
        # Compute the farthest centers
        assignations = np.full_like(self.components, np.nan)

        # Precompute center distances
        assignation_cluster, center_cluster, intra_number = {}, {}, {}
        for c in range(self.number):
            assignation_cluster[c] = (self.components == c).reshape((-1,1))
            center_cluster[c] = data[self.components == c].mean(0)
            intra_number[c] = assignation_cluster[c].sum()

        ## First assignation is the largest component
        assignations[self.components == 0] = 0

        ## Save characteristics of assigned components
        assigned_center = data[self.components == 0].mean(0)
        remaining = set(range(1, self.number))

        ## Compute iteratively the farthest given all other
        match = {}
        for i in range(1, self.k):
            # Computes distances to all remaining NON TRIVIAL connected components
            # We ignore the last part which depends on the intravariance of the past clusters
            distance = {c: np.linalg.norm(assigned_center - center_cluster[c])
                for c in remaining if intra_number[c] > 1}

            farthest = max(distance, key = lambda x: distance[x])
            assignations[assignation_cluster[farthest].flatten()] = i
            match[farthest] = i

            # Update characteristics of assigned
            assigned_center = data[assignations >= 0].mean(0)

            # Remove components
            remaining.remove(farthest)

        # Assign each unassigned components
        for i in remaining:
            ## Computes distances to all other cluster 
            ## We ignore the last part which depends on the intravariance of the cluster i
            distance = {c: np.linalg.norm(center_cluster[i] - center_cluster[c])
                for c in match}

            ## Closest verifying constraint
            closest = min(distance.keys(), key = lambda k: distance[k])
            assignations[assignation_cluster[i].flatten()] = match[closest]

            ## Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            center_cluster[closest] = data[assignation_cluster[closest]].mean(0)

        return assignations
