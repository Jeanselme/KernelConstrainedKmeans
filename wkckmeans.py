"""
    This file contains an implementation of a kernel constrained kmeans
"""
import numpy as np

def weightedKernelConstrainedKmeans(kernel, assignation, constraints = None, weights = None, max_iteration = 100, threshold_certainty = 0):
    """
        Compute kernel constrained kmeans

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
    assignation_cluster, intra_distance, number, base_distance = {}, {}, {}, {}
    index = np.arange(len(assignation))
    clusters = np.unique(assignation)
    iteration, change = 0, True

    if weights is None:
        weights = np.ones_like(assignation)

    while change and iteration < max_iteration:
        change = False
        np.random.shuffle(index)

        # Update cluster centers
        for k in clusters:
            assignation_cluster[k] = np.multiply((assignation == k), weights).reshape((-1,1))
            intra_distance[k] = np.matmul(kernel, assignation_cluster[k])
            number[k] = np.sum(assignation_cluster[k])
            base_distance[k] = np.dot(assignation_cluster[k].T, intra_distance[k])/(number[k]**2)

        for i in index:
            previous = assignation[i]

            if constraints is None:
                possibleClusters = clusters
            else:
                # Computes possible cluster for the point that does not break any constraint
                possibleClusters = [c for c in clusters if 
                                    (c not in np.unique(assignation[constraints[i, :] < -threshold_certainty])) and # Cannot link constraint
                                    ((c in np.unique(assignation[constraints[i, :] > threshold_certainty])) or      # Must link constraint
                                    (len(assignation[constraints[i, :] > threshold_certainty]) == 0))]             # In case no constraint

                assert len(possibleClusters) > 0, "No cluster respecting constraint"

            distance = {k: float(base_distance[k]) - 2*intra_distance[k][i]/number[k] for k in possibleClusters}
            assignation[i] = min(distance, key=lambda d: float(distance[d]))
            if previous != assignation[i]:
                change = True

        iteration += 1

    return assignation