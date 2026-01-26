"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Function set 2.
"""

################### Function set 2 ###################
##### When importing packages, avoid circular imports. #####
import numpy as np


def common_seq(data, group_label):
    """
    Input data and arrays of each cluster to obtain the common pattern of that cluster (ignoring 0).

    :param data: Cell-SNP matrix;
    :param group_label: The cluster labels of each sample (predicted results);
    :return common_count: Genotyping counting matrix for each cell cluster;
    :return common: the common pattern of clusters.
    """
    cluster = len(np.unique(group_label))
    snp = data.shape[1]
    counts = []
    for g in range(1, cluster+1):
        per_snp_count = np.zeros((snp, 3), dtype=np.int_)
        temp_data = data[group_label == g,:]
        for s in range(snp):
            uniq, temp_counts = np.unique(temp_data[:,s], return_counts=True)
            # uniq like [0 1 2 3]、[0 1 2]、[0 1 3]
            count = np.zeros(3, dtype=np.int_)
            for i in range(len(uniq)): # 1 2 3
                if uniq[i] != 0:
                    count[uniq[i]-1] = temp_counts[i]
            per_snp_count[s,:] = count
        counts.append(per_snp_count)
    common_count = np.stack(counts)
    common = np.argmax(common_count, axis=2) + 1
    return common_count, common

def get_group_pred_plus(data, group_label, n=2):
    """
    Reclassify the classification results obtained earlier and obtain the predicted labels for clustering.

    :param data: Cell-SNP matrix;
    :param group_label: The cluster labels of each sample (predicted results);
    :param n: Number of reclassify;
    :return group_label: The cluster new labels of each sample (predicted results).
    """
    cluster = len(np.unique(group_label))
    cell_num = data.shape[0]
    for _ in range(n):
        _, common = common_seq(data, group_label)
        similarity = np.zeros((cell_num, cluster), dtype=np.float32)
        for i in range(cell_num):
            cell = data[i,:]
            for g in range(cluster):
                group_common = common[g,:]
                similarity[i,g] = calc_similarity(cell, group_common)
        group_label = np.argmax(similarity, axis=1) + 1
    return group_label

def calc_similarity(celli, cellj):
    """
    Calculate the similarity between two sequences (cells).

    :param celli: cell1;
    :param cellj: cell2;
    :return similarity: Similarity between cells: 
    """
    # Non zero length (two cells simultaneously)
    non_zero = (celli != 0) & (cellj != 0)
    com_non_num = np.sum(non_zero)
    # Non zero, same number of SNPs
    com_snp_num = 0
    com_snp_num = np.sum(non_zero & (celli == cellj)) \
        + np.sum((celli == cellj) & (celli == 2)) \
        + np.sum((celli == cellj) & (celli == 3))*2
    
    temp = 0 if com_non_num == 0 else com_snp_num/com_non_num
    # print(temp, com_snp_num, com_non_num)
    return temp
