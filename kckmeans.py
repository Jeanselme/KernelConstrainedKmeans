"""
    This file contains an implementation of a kernel constrained kmeans
"""
import numpy as np

def kernelConstrainedKmeans(kernel, assignation, constraints, max_iteration = 100, threshold_certainty = 0):
    """
        Compute kernel constrained kmeans
        
        Arguments:
            kernel {Array n*n} -- Kernel
            assignation {Array n} -- Initial assignation 
            constraints {Array n*n} -- Constraints matrix with value between -1 and 1
        
        Keyword Arguments:
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

    while change and iteration < max_iteration:
        change, approx_error = False, 0.
        np.random.shuffle(index)

        # Update cluster centers
        for k in clusters:
            assignation_cluster[k] = (assignation == k).reshape((-1,1))
            intra_distance[k] = np.matmul(kernel, assignation_cluster[k])
            number[k] = np.sum(assignation_cluster[k])
            base_distance[k] = np.dot(assignation_cluster[k].T, intra_distance[k])/(number[k]**2)

        for i in index:
            previous = assignation[i]

            # Computes possible cluster for the point that does not break any constraint
            possibleClusters = [c for c in clusters if 
                                (c not in np.unique(assignation[constraints[i, :] < -threshold_certainty])) and # Cannot link constraint
                                ((c in np.unique(assignation[constraints[i, :] > threshold_certainty])) or      # Must link constraint
                                 (len(assignation[constraints[i, :] > threshold_certainty]) == 0))]             # In case no constraint

            distance = {k: float(base_distance[k]) for k in possibleClusters}
            for k in possibleClusters:
                # Only this term implies a change if center unupdated
                distance[k] += kernel[i,i] - 2*intra_distance[k][i]/number[k]
                assignation[i] = k

            assert len(possibleClusters) > 0, "No cluster respecting constraint"

            assignation[i] = min(distance, key=lambda d: float(distance[d]))
            if previous != assignation[i]:
                change = True
            approx_error += float(distance[assignation[i]])

        iteration += 1

    return assignation