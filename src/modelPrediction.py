"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Model prediction.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Model prediction.
        Using a trained model for prediction.
    """)
parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
parser.add_argument("-vm", "--vae_model", default="not_provided", help="Trained VAE model")
parser.add_argument("-ld", "--latent_dim", default=4, help="Latent dimension, default 4")
parser.add_argument("-c", "--cluster", default=3, help="Number of clusters, default 3")
parser.add_argument("-si", "--show_info", default="FALSE", help="Whether to display detailed information, default True")
parser.add_argument("-s", "--seed", default=42, help="Random seed, default 42")
parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")
parser.add_argument("-o", "--out", default=".", help="Output Directory")
parser.add_argument("-p", "--prefix", default="vae_", help="Output prefix, default vae_")
parser.add_argument("-pc", "--plot_center", default="False", help="Do you want to draw cluster centers, default False")
args = parser.parse_args()


################ parameter/arguments ################
mtx_file = args.mtx
vae_file = args.vae_model
latent_dim = int(args.latent_dim)
cluster = int(args.cluster)
showInfo = False if args.show_info.upper() == "FALSE" else True
seed = int(args.seed)
img_type = args.img_type
out_path = args.out
prefix = args.prefix
plot_center = False if args.plot_center.upper() == "FALSE" else True


################ import packages ################
import numpy as np
import pandas as pd
import gc
import subprocess
import datetime
import os

#### others
from others import display_info
#### model
from model import *
from functions3 import *

################ Check if training has been completed ################
log_file = f"{out_path}/deeplex_prediction.log"
def log_message(message):
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
    print(message)
subprocess.run(f">{log_file}", shell=True)

# create directory
if out_path == "." or out_path == "./":
    result_dir = subprocess.run("pwd", shell=True, capture_output=True, text=True)
    out_path = result_dir.stdout.strip()

subprocess.run(f"mkdir -p {out_path}/__prediction__", shell=True)
subprocess.run(f"mkdir -p {out_path}/__prediction__/imgs", shell=True)
subprocess.run(f"mkdir -p {out_path}/__prediction__/results", shell=True)

# Check model file
if vae_file == "not_provided":
    train_completed = f"{out_path}/__train__/train.completed"
    if os.path.exists(train_completed):
        train_log = f"{out_path}/__train__/deeplex_train.log"
        with open(train_log, "r") as log:
            lines = log.readlines()
            for line in lines:
                if line.startswith("best epoch: "):
                    best_num = line.split(": ")[1].split("\t")[0].strip()
                    vae_file = out_path + "/__train__/models/model_{:0>3}.weights.h5".format(best_num)

current_time = datetime.datetime.now()
out_disp = display_info("start "+str(current_time).split(".")[0])
log_message(out_disp)
################ import data ################
# read cell-snp matrix
mtx = pd.read_csv(mtx_file, sep='\s+', header = 0, index_col=0)
# Store encoded data
data = np.zeros((mtx.shape[0], mtx.shape[1], 2), dtype=np.float64)
# code
conv = {0:np.array([0,0]), 1:np.array([1,0]), 2:np.array([0,1]), 3:np.array([0.5,0.5])}
barcode = np.array(mtx.index.tolist())

# Encode
for r in range(len(mtx)):
  row = mtx.values[r,]      # No.r cell
  for c in range(len(row)):
    data[r,c] = conv[row[c]]

gc.collect()

log_message("data:")
log_message(f"  Cell-SNP Matrix dim: {mtx.shape}")
log_message(f"  Encode data dim: {data.shape}")
log_message(f"  Barcode number: {len(barcode)}")


################ print parameter/arguments ################
log_message("parameter:")
log_message(f"  Model path: {vae_file}")
log_message(f"  Latent dimension: {latent_dim}")
log_message(f"  Number of clusters: {cluster}")
log_message(f"  Seed: {seed}")
log_message(f"  Image type: {img_type}")
log_message(f"  File prefix: {prefix}")
log_message(f"  Plot center: {plot_center}")

snp_dim = data.shape[1]
cat = 2
n_subset = 1500
n_common = 200


################ prediction ################
# load model
vae = load_model(vae_file, snp_dim, cat, latent_dim)

group_pred, group_pred_plus = result(vae = vae, data = data, barcode = barcode,
                                        matrix = mtx.values, cluster = cluster,
                                        out_path = out_path+"/__prediction__/", prefix = prefix, 
                                        num = n_subset, n = n_common,
                                        img_type = img_type, saveImg = True,
                                        plot_center = plot_center, saveZ = True, 
                                        showInfo = showInfo)


current_time = datetime.datetime.now()
out_disp = display_info("end "+str(current_time).split(".")[0])
log_message(out_disp)
log_message("Prediction completed successfully.")
