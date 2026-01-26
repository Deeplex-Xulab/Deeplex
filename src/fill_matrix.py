"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Input a dense matrix, fill missing entries in the matrix.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Input a dense matrix, fill missing entries in the matrix.
    """)
parser.add_argument("-m", "--matrix", default="matrix.txt", help="Dense matrix")
parser.add_argument("-n", "--most_num", default="20", help="Number of most similar cells to use for filling missing values, default 20")
parser.add_argument("-o", "--out", default="matrix_final.txt", help="Output filled matrix")
args = parser.parse_args()


################ parameter/arguments ################
mtx_file = args.matrix
most_num = int(args.most_num)
out_file = args.out


################ import packages ################
import numpy as np
import pandas as pd


# read matrix
raw_data = pd.read_csv(mtx_file, sep="\s+", header=0, index_col=0)
data = raw_data.values
# barcode
cell_barcodes = raw_data.index
# SNP
snp_names = raw_data.columns
cell_num = data.shape[0]
snp_num = data.shape[1]

class CELL:
  """
  storage class for cell information
  """
  def __init__(self, snps, cell_barcode=""):
    """init method"""
    self.snps = np.copy(snps)  # deep copy of snps
    self.nozero_length = np.sum(self.snps != 0) # non-zero length
    self.cell_barcode = cell_barcode  # barcode of the cell
  def __str__(self):
    """print method"""
    fmt = "snps:{0}, nozero_length:{1}, barcode:{2}".format(
        self.snps, self.nozero_length, self.cell_barcode)
    return fmt

def deep_copy(cell_obj:CELL):
  """
  return a deep copy of the cell object
  """
  return CELL(np.copy(cell_obj.snps), cell_obj.cell_barcode)

# convert the data into CELL objects
all_cells = [] # according to the cell_barcode
for i in range(cell_num): # No. i cell
  snps = data[i]
  cell_barcode = cell_barcodes[i]
  cell_obj = CELL(snps, cell_barcode)
  all_cells.append(cell_obj)


def fill_deletion(cell_obj, most_index):
  """
  used to fill missing values in the cell object
  most_index : the list of indices of the most similar cells
  """
  # deep copy the current cell to avoid modifying the original data
  temp_cell = deep_copy(cell_obj)
  for idx in most_index:
    cell = all_cells[idx]
    # replace the missing values (0) in the current cell with the corresponding values from the most similar cells
    temp_cell.snps[temp_cell.snps == 0] = cell.snps[temp_cell.snps == 0]
  return CELL(temp_cell.snps, temp_cell.cell_barcode)

# Store the similarity between cells as a matrix
# Calculate the similarity between cells by assigning weights of 1, 2, and 3 to genotypes 1, 2, and 3 respectively.
with open(out_file, 'w') as out:
  # output the header
  out.write("\t" + "\t".join(snp_names) + "\n")
  similarity = np.zeros((cell_num,cell_num), dtype=np.int_)
  for i in range(cell_num):
    # if i % 200 == 0:
    #   print("calculated", i , "/", cell_num, "cells")
    celli = all_cells[i]
    for j in range(i+1, cell_num):
      cellj = all_cells[j]
      # non-zero length (two cell)
      non_zero = (celli.snps != 0) & (cellj.snps != 0)
      com_non_num = np.sum(non_zero)
      # non-zero length of common SNPs (two cell)
      com_snp_num = 0
      com_snp_num = np.sum(non_zero & (celli.snps == cellj.snps)) \
                + np.sum(non_zero & (celli.snps == cellj.snps) & (celli.snps == 2)) \
                + np.sum(non_zero & (celli.snps == cellj.snps) & (celli.snps == 3))*2
      temp = 0 if com_non_num == 0 else com_snp_num/com_non_num
      similarity[i,j] = temp
      similarity[j,i] = temp
    curr_cell = similarity[i,:]
    most_index = np.argpartition(curr_cell, -most_num)[-most_num:]
    # fill the missing values in the current cell using the most similar cells
    fill_cell = fill_deletion(celli, most_index)
    # output the filled cell information
    line = fill_cell.cell_barcode + "\t" + "\t".join(str(snp) for snp in fill_cell.snps)
    out.write(line + "\n")
