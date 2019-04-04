"""
    This file contains an implementation of a kernel constraint kmeans
    with soft penalty to allow to break some of the constraints
"""
import numpy as np


def weightedKernelSoftConstrainedKmeans(kernel, assignation, constraints = None, penalty = 0.5, weights = None, max_iteration = 100, threshold_certainty = 0):
    """
        Compute kernel constrained kmeans with a penalty for broken constraints

        Arguments:
            kernel {Array n*n} -- Kernel
            assignation {Array n} -- Initial assignation 

        Keyword Arguments:
            constraints {Array n*n} -- Constraints matrix with value between -1 and 1 
                (default: {None} -- No constraint => Simpel kernel kmeans)
            weights {Array n} -- Initial Weights for the different points (default: {None} -- Equal weights)
            max_iteration {int} -- Maximum iteration (default: {100})
            threshold_certainty {float} -- Level under which we consider to break 
                    a cannot link constraint (take negative value of it)
                    a must link (take positive)
                    (default: {0} Any constraint is considered)

        Returns:
            Assignation - Array n
    """
    intra_distance, number, base_distance = {}, {}, {}
    index = np.arange(len(assignation))
    clusters = np.unique(assignation)
    iteration, change = 0, True

    if weights is None:
        weights = np.ones_like(assignation)

    max_distance = np.max([kernel[i, i] + kernel[j, j] - 2 * kernel[i, j] for i in index for j in index[:i]])

    for _ in range(max_iteration):
        # Update cluster centers
        for k in clusters:
            assignation_cluster = np.multiply((assignation == k), weights).reshape((-1,1))
            intra_distance[k] = np.matmul(kernel, assignation_cluster)
            number[k] = np.sum(assignation_cluster)
            base_distance[k] = np.dot(assignation_cluster.T, intra_distance[k])/(number[k]**2)

        assignation_previous = assignation.copy()
        # Double loop : centers are fixed but assignation is updated to compute broken constraints
        while change and iteration < max_iteration:
            np.random.shuffle(index)
            for i in index:
                previous = assignation[i]

                distance = {k: float(base_distance[k]) for k in clusters}
                for k in clusters:
                    # Only this term implies a change if center unupdated
                    distance[k] += kernel[i,i] - 2*intra_distance[k][i]/number[k]

                    # Also add the penalty of putting this points in this cluster
                    if constraints is not None:
                        # Computes broken constraints
                        assignation_cluster = np.multiply((assignation == k), weights).reshape((-1,1))
                        not_assigned_cluster = np.multiply((assignation != k), weights).reshape((-1,1))
                        broken_must_link = np.multiply(not_assigned_cluster.T, constraints[i] > threshold_certainty)
                        broken_cannot_link = np.multiply(assignation_cluster.T, constraints[i] < -threshold_certainty)

                        # Computes penalty
                        penalty_ml = np.dot(broken_must_link, kernel.diagonal()) \
                            - 2 * np.dot(broken_must_link, kernel[i, :]) \
                            + np.sum(broken_must_link) * kernel[i,i]
                        penalty_cl = np.sum(broken_cannot_link) * max_distance \
                            - np.dot(broken_cannot_link, kernel.diagonal()) \
                            + 2 * np.dot(broken_cannot_link, kernel[i, :]) \
                            - np.sum(broken_cannot_link) * kernel[i,i]

                        distance[k] += penalty * (penalty_cl + penalty_ml)

                assignation[i] = min(distance, key=lambda d: float(distance[d]))
                if previous != assignation[i]:
                    change = True

            iteration += 1

        # Stops if no change
        if np.array_equal(assignation, assignation_previous):
            break
            
    return assignation