"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Model prediction.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Doublet cell prediction.
        Based on the previous step's prediction results for each cell, further predict which of them are doublets.
    """)
parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
parser.add_argument("-s", "--single_pred", default="not_provided", help="tSNE_pred.txt, single cell prediction result file")
parser.add_argument("-c", "--coordinate", default="not_provided", help="tSNE_result_data.txt, tSNE/UMAP result data file")
parser.add_argument("-f", "--fraction", default="not_provided", help="Is the percentage of doublet cells known in advance, default False")
parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")
parser.add_argument("-o", "--out", default=".", help="Output Directory")
parser.add_argument("-p", "--prefix", default="vae_", help="Output prefix, default vae_")
args = parser.parse_args()


################ parameter/arguments ################
mtx_file = args.mtx
pred_file = args.single_pred
coordinate_file = args.coordinate
doublet_percent = float(args.fraction) if args.fraction != "not_provided" else "not_provided"
img_type = args.img_type
out_path = args.out
prefix = args.prefix

if pred_file == "not_provided":
    pred_file = f"{out_path}/__prediction__/results/vae_{img_type}_pred.txt"
if coordinate_file == "not_provided":
    coordinate_file = f"{out_path}/__prediction__/results/vae_{img_type}_result_data.txt"

################ import packages ################
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import gc
import subprocess
import datetime

#### others
from others import display_info
#### function
from functions4 import *


log_file = f"{out_path}/deeplex_doublet.log"
def log_message(message):
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
    print(message)
subprocess.run(f">{log_file}", shell=True)

# create directory
if out_path == "." or out_path == "./":
    result = subprocess.run("pwd", shell=True, capture_output=True, text=True)
    out_path = result.stdout.strip()

subprocess.run(f"mkdir -p {out_path}/__doublet__", shell=True)
subprocess.run(f"mkdir -p {out_path}/__doublet__/imgs", shell=True)
subprocess.run(f"mkdir -p {out_path}/__doublet__/results", shell=True)


current_time = datetime.datetime.now()
out_disp = display_info("start "+str(current_time).split(".")[0])
log_message(out_disp)
################ import data ################
# read cell-snp matrix
mtx = pd.read_csv(mtx_file, sep='\s+', header = 0, index_col=0)
# tSNE/UMAP map
map = pd.read_csv(coordinate_file, sep='\s+', header = None, index_col=None)
result_data = map.values
# barcode and group prediction
pred = pd.read_csv(pred_file, sep='\t', header=0, index_col=0)
barcode = np.array(pred.index)
group_pred = pred['prediction'].values
cluster = len(np.unique(group_pred))
data = mtx.values

gc.collect()

log_message("data:")
log_message(f"  Cell-SNP Matrix dim: {mtx.shape}")
log_message(f"  Barcode number: {len(barcode)}")
log_message(f"  Cluster number: {cluster}")


################ print parameter/arguments ################
log_message("parameter:")
log_message(f"  Cell-SNP Matrix: {mtx_file}")
log_message(f"  Single cell prediction result file: {pred_file}")
log_message(f"  tSNE/UMAP result data file: {coordinate_file}")
log_message(f"  Percentage of doublet cells: {doublet_percent}")
log_message(f"  Image type: {img_type}")


################ doublet prediction ################
# get combination of clusters
arr = np.array(range(cluster))
comb = np.array(np.meshgrid(arr, arr)).T.reshape(-1, 2)
combination = comb[comb[:, 0] != comb[:, 1]]
combination = np.unique([tuple(sorted(item)) for item in combination], axis=0)

# Doublet Identification Log
doublet_log = f"{out_path}/__doublet__/doublet.log"
# Calculate the number of double cells
if doublet_percent == "not_provided":
    is_percent = False
    doublet_num = None
else:
    is_percent = True
    doublet_num = len(barcode) * doublet_percent

# doublet prediction
result(data, group_pred, doublet_log, combination, barcode, 
       result_data, prefix, out_path+"/__doublet__/", is_percent, img_type, 
       doublet_num, saveInfo=True)

current_time = datetime.datetime.now()
out_disp = display_info("end "+str(current_time).split(".")[0])
log_message(out_disp)
log_message("Doublet prediction completed successfully.")
