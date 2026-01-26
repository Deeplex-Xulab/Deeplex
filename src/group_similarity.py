"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Model performance evaluation indicators.
"""

#################### Model performance evaluation indicators ###################
##### When importing packages, avoid circular imports. #####
import numpy as np
import itertools


def group_seq(matrix, group_label, cluster):
    """
    Input data and arrays of each cluster to obtain the common pattern of that cluster (ignoring 0).

    :param matrix: Cell-SNP matrix;
    :param group_label: The cluster labels of each sample (predicted results);
    :param cluster: Number of cell clusters;
    :return common: the common pattern of clusters.
    """
    snp = matrix.shape[1]
    counts = []
    for g in range(1, cluster+1):
        per_snp_count = np.zeros((snp, 3), dtype=np.int_)
        temp_matrix = matrix[group_label == g,:]
        for s in range(snp):
            uniq, temp_counts = np.unique(temp_matrix[:,s], return_counts=True)
            # Uniq may be [0 1 2 3], [0 1 2], [0 1 3]
            count = np.zeros(3, dtype=np.int_)
            for i in range(len(uniq)): # 1 2 3
                if uniq[i] != 0:
                    count[uniq[i]-1] = temp_counts[i]
            per_snp_count[s,:] = count
        counts.append(per_snp_count)
    common_count = np.stack(counts)
    common = np.argmax(common_count, axis=2) + 1
    return common

def group_combinations(cluster):
    """
    Get pairwise combinations.

    :param cluster: Number of cell cluster;
    :return combinations: Pairwise combinations of cell clusters.
    """
    return np.array(list(itertools.combinations(list(range(cluster)), 2)))

def group_similarity(group_seqs, combination):
    """
    Calculate the average similarity between each cluster based on group_seq() (pairwise calculation, then average)

    :param common_seq: the common pattern of clusters;
    :param combination: pairwise combinations;
    :return avg_simil: the average similarity between each cluster.
    """
    simil = []
    for c in combination:
        # print(c)
        group1_seq = group_seqs[c[0]]
        group2_seq = group_seqs[c[1]]
        # print(np.sum(group1_seq == group2_seq)/len(group1_seq))
        simil.append(np.sum(group1_seq == group2_seq)/len(group1_seq))
    return np.mean(np.array(simil))

