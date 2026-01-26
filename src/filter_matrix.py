"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Input a dense matrix, retain SNPs with high variability, and remove SNPs with too much missing data.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Input a dense matrix, retain SNPs with high variability, and remove SNPs with too much missing data.
    """)
parser.add_argument("-m", "--matrix", default="matrix.txt", help="Dense matrix")
parser.add_argument("-p", "--percent", default="0.99", help="Missing value ratio, default 0.99")
parser.add_argument("-c", "--common", default="0.95", help="Maximum proportion of identical SNP genotypes, default 0.95")
parser.add_argument("-o", "--out", default="matrix_filter.txt", help="Filtered matrix")
parser.add_argument("-t", "--test", default="NO", help="Whether to test only (check the number of SNPs after filtering without outputting the matrix)")
args = parser.parse_args()


################ parameter/arguments ################
percent = float(args.percent)
common = float(args.common)
mtx_file = args.matrix
out_file = args.out
test = args.test


################ import packages ################
import numpy as np
import pandas as pd


# read matrix
matrix = pd.read_csv(mtx_file, sep='\s+', header = 0)
# row is barcode, column is SNP
data = matrix
index = np.mean(data.values == 0, axis=0) >= percent
dropSNPs = data.columns[index]
# deep copy to avoid modifying the original data
temp = data.copy()
temp.drop(dropSNPs,  axis=1, inplace=True)

snp_num = len(temp.columns)
per_index = np.zeros(snp_num, dtype=np.bool_)
n = 0
for i in range(snp_num):
    snp = temp.iloc[:,i]    # No. i SNP
    non_zero = np.sum(snp.values != 0)  # none-zero values
    non_snp = snp[snp.values != 0]
    pers = (non_snp.value_counts()/non_zero).values     # percentage of each genotype
    is_diff = pers >= common
    if np.sum(is_diff) >= 1:
        n += 1
        per_index[i] = True
        # print(pers)

dropSNPs = temp.columns[per_index]
# deep copy to avoid modifying the original data
per_temp = temp.copy()
per_temp.drop(dropSNPs,  axis=1, inplace=True)


################ print filter info ################
print("Original number of SNPs: ", len(data.columns))
print("Number of SNPs removed due to excessive missing data: ", np.sum(index))
print("Number of SNPs remaining: ", np.sum(~index))
print("Number of SNPs removed due to high genotype similarity: ", np.sum(per_index))
print("Final number of SNPs retained: ", np.sum(~per_index))

if(test.upper() == "YES"):
    pass
else:
    per_temp.to_csv(out_file, sep="\t", index=True, header=True)   
