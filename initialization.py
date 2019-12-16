"""
    Files containing all the procedures in order
    to initialize the cluster assigment at the start
"""
import numpy as np
from scipy.sparse import find, coo_matrix, issparse
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
        self.number, components = connected_components(constraint > 0, directed=False)
        unique, count = np.unique(components, return_counts = True)
        order = np.argsort(count)[::-1]
        self.components = np.argsort(unique[order])[components]
        self.constraint = constraint
        assert self.number >= k, "Constraint too important for number of cluster"

        if (count > 1).sum() < k:
            print("Constraints do not allow to find enough connected components for farthest first ({} for {} classes) => Approximation".format((count > 1).sum(), k))
            self.farthest_initialization = lambda x: self.back_up(x)

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
        constraint = self.constraint
        if issparse(constraint):
            constraint = constraint.todense()

        for i in remaining:
            ## Computes distances to all other cluster 
            ## We ignore the last part which depends on the intravariance of the cluster i
            distance = {c: float(np.dot(assignation_cluster[c].T, intra_distance[c])/(intra_number[c]**2) 
                - 2 * np.dot(assignation_cluster[i].T, intra_distance[c])/(intra_number[c] * intra_number[i]))
                for c in match}

            ## Closest verifying constraint
            order = sorted(distance.keys(), key = lambda k: distance[k])

            ## If no constraint is positive => Too much constraint
            broken_constraint = constraint[:, assignation_cluster[i].flatten()]
            closest = min(order, key=lambda o: np.sum(broken_constraint[(assignations == o),:] < 0))
            assignations[assignation_cluster[i].flatten()] = match[closest]

            ## Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return assignations

    def back_up(self, kernel):
        """
            Farthest points that verify constraint
            Withour constraint on non trivial cluster
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

        # Assign each unassigned components
        constraint = self.constraint
        if issparse(constraint):
            constraint = constraint.todense()

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
            broken_constraint = constraint[:, assignation_cluster[i].flatten()]
            closest = min(order, key=lambda o: np.sum(broken_constraint[(components == o),:] < 0))
            components[assignation_cluster[i].flatten()] = closest

            # Update assignation closest
            assignation_cluster[closest] += assignation_cluster[i]
            intra_distance[closest] += intra_distance[i]
            intra_number[closest] += intra_number[i]

        return components

    def random_initialization(self):
        """
            Random Assignation
        """
        return np.random.choice(self.k, size = len(self.components))

class Euclidean_Initialization(Initialization):

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
        match, centers = {}, [center_cluster[0]]
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
            centers.append(center_cluster[farthest])

        # Save centers if no assignation needed
        self.centers = np.vstack(centers)

        # Assign each unassigned components
        for i in remaining:
            ## Computes distances to all other cluster 
            ## We ignore the last part which depends on the intravariance of the cluster i
            distance = {c: np.linalg.norm(center_cluster[i] - center_cluster[c])
                for c in match}

            ## Closest verifying constraint
            # TODO: Verify constraint
            closest = min(distance.keys(), key = lambda k: distance[k])
            assignations[assignation_cluster[i].flatten()] = match[closest]

            ## Update assignation closest
            assignation_cluster[closest] = assignation_cluster[closest] + assignation_cluster[i]
            center_cluster[closest] = data[assignation_cluster[closest].flatten()].mean(0)

        return assignations

class InitializationScale:
    """
        Farthest first initialization with precomputation of connected components
    """

    def __init__(self, k, constraintmatrix):
        """
            Precompute connected components
            Arguments:
                k {int} -- Number of cluster
                constraintmatrix {sparse matrix  n * n} -- Constraint matrix with value in (-1, 1)
                    Positive values are must link constraints
                    Negative values are must not link constraints
        """
        assert constraintmatrix is not None, "Farthest initialization cannot be used with no constraint"
        # Computes the most important components and order by importance
        number, components = connected_components(constraintmatrix > 0, directed=False)
        assert number >= k, "Constraint too noisy"
        self.k = k
        bincount = np.bincount(components)
        largest = bincount.argmax()
        self.components = components
        self.components_subset = np.where(bincount>1)[0]

        if len(self.components_subset) < k:
            print("Constraints do not allow to find enough connected components for farthest first ({} for {} classes) => Random forced".format(len(self.components_subset), k))
            self.farthest_initialization = lambda x: None
        self.largestidx = np.where(self.components_subset==largest)[0][0]

    def farthest_initialization(self, X):
        """
            Farthest points that verify constraint
            Arguments:
                X {Array n * d} -- data
        """
        centers = np.vstack([X[self.components==i].mean(0) for i in self.components_subset])
        distances = np.linalg.norm(centers[self.largestidx]-centers,axis =1)
        farthest = np.argmax(distances)
        clusters = set([self.largestidx,farthest])
        for i in range(2,self.k):
            distances = np.zeros(len(centers))
            for j in clusters:
                distances += np.linalg.norm(centers[j]-centers,axis =1)
            for farthest in np.argsort(-distances):
                if not farthest in clusters:
                    clusters.add(farthest)
                    break

        cluster_centers = [X[self.components==self.components_subset[i]].mean(0) for i in clusters]

        return np.vstack(cluster_centers)