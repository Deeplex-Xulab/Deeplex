"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Input a sparse matrix vartrix.mtx and convert it into a dense (normal) matrix.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Input a sparse matrix vartrix.mtx and convert it into a dense (normal) matrix.
        The output matrix should have barcodes as rows and SNPs as columns.
    """)
parser.add_argument("-m", "--matrix", default="vartrix.mtx", help="Sparse matrix")
parser.add_argument("-s", "--snp", default="snp.list", help="SNPs list")
parser.add_argument("-b", "--barcode", default="barcode.tsv", help=" Barcode list")
parser.add_argument("-t", "--transposition", default="yes", help="Whether to transpose")
parser.add_argument("-o", "--out", default="matrix.txt", help="Output file")
args = parser.parse_args()


################ parameter/arguments ################
mtx_file = args.matrix
snp_file = args.snp
barcode_file = args.barcode
transposition = args.transposition
is_t = False if transposition.lower().startswith("n") else True
out_file = args.out


################ import packages ################
from scipy.io import mmread
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# read sparse matrix
sparse_matrix = mmread(mtx_file)

# convert to dense matrix
dense_matrix = sparse_matrix.toarray().astype(int)

# read SNPs and barcodes
snp = np.loadtxt(snp_file, comments="#", dtype=str)
barcode = np.loadtxt(barcode_file, comments="#", dtype=str)

df = pd.DataFrame(dense_matrix, index=snp, columns=barcode)
if is_t:
    df = df.T

df.to_csv(out_file, sep="\t", index=True, header=True)
