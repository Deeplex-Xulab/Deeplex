"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Model training.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Model training.
        Input the required data to train the model.
    """)
parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
parser.add_argument("-bn", "--batch_num", default=16, help="Number of batch, default 16")
parser.add_argument("-en", "--epoch_num", default=25, help="Number of epoch, default 25")
parser.add_argument("-ld", "--latent_dim", default=4, help="latent dimension, default 4")
parser.add_argument("-c", "--cluster", default=3, help="Number of clusters, default 3")
parser.add_argument("-si", "--show_info", default="True", help="Whether to display detailed information, default True")
parser.add_argument("-tr", "--test_radio", default=0.2, help="Ratio of test set, default 0.2")
parser.add_argument("-vr", "--valid_radio", default=0.2, help="Ratio of validation set, default 0.2")
parser.add_argument("-s", "--seed", default=42, help="Random seed, default 42")
parser.add_argument("-lr", "--learning_rate", default=0.0001, help="Learning rate, default 0.0001")
parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")
parser.add_argument("-o", "--out", default=".", help="Output Directory")
args = parser.parse_args()


################ parameter/arguments ################
mtx_file = args.mtx
n_batch = int(args.batch_num)
n_epoch = int(args.epoch_num)
latent_dim = int(args.latent_dim)
cluster = int(args.cluster)
showInfo = False if args.show_info.upper() == "FALSE" else True
test_radio = float(args.test_radio)
valid_radio = float(args.valid_radio)
seed = int(args.seed)
learning_rate = float(args.learning_rate)
img_type = args.img_type
out_path = args.out


if n_epoch < 15:
    print("The number of epochs should be at least 15.")
    import sys
    sys.exit(1)


################ Check if training has been completed ################
import os
import subprocess
log_file = f"{out_path}/deeplex_training.log"
def log_message(message):
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
    print(message)
subprocess.run(f">{log_file}", shell=True)

# create directory
if out_path == "." or out_path == "./":
    result = subprocess.run("pwd", shell=True, capture_output=True, text=True)
    out_path = result.stdout.strip()

subprocess.run(f"mkdir -p {out_path}/__train__", shell=True)
subprocess.run(f"mkdir -p {out_path}/__train__/models", shell=True)
subprocess.run(f"mkdir -p {out_path}/__train__/loss", shell=True)

train_completed = f"{out_path}/__train__/train.completed"
# Check if training has been completed
if os.path.exists(train_completed):
    log_message("Training has already been completed. Skipping...")
    exit(0)


################ import packages ################
import numpy as np
import pandas as pd
import gc
import subprocess
import datetime
import os
from tensorflow import keras

#### others
from others import display_info
#### split data
from train_test_valid_data import *
#### group similarity
from group_similarity import *
#### function
from functions1 import *
# from functions2 import *
#### model
from model import *



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
log_message(f"  Encode data dim: {data.shape}", )
log_message(f"  Barcode number: {len(barcode)}")


################ print parameter/arguments ################
log_message("parameter:")
log_message(f"  Number of batch: {n_batch}")
log_message(f"  Number of eopch: {n_epoch}")
log_message(f"  Latent dimension: {latent_dim}")
log_message(f"  Number of clusters: {cluster}")
log_message(f"  Seed: {seed}")
log_message(f"  Learning rate: {learning_rate}")
log_message(f"  Image type: {img_type}")

snp_dim = data.shape[1]
cat = 2
n_subset = 1500
n_common = 200

################ train, valid, test set ################
(train_data,train_barcode,train_mtx),(valid_data,valid_barcode,valid_mtx),(test_data,test_barcode,test_mtx) = get_train_test_valid_data(data, barcode, mtx.values, test_radio=test_radio, valid_radio=valid_radio, seed=seed, showInfo=showInfo)

# save train, valid, test set barcode list
np.savetxt(out_path + "/__train__/train_barcodes.list", train_barcode, fmt='%s')
np.savetxt(out_path + "/__train__/valid_barcodes.list", valid_barcode, fmt='%s')
np.savetxt(out_path + "/__train__/test_barcodes.list", test_barcode, fmt='%s')


################ started training ################
train_log = out_path + "/__train__/deeplex_train.log"
with open(train_log,"w") as log: log.write("Validation set\n")
# built model
model = VAE(snp_dim, cat, latent_dim)
model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate))     # learning_rate:0.0001
history = LossHistory()
combination = group_combinations(cluster)
custom_callback = CustomCallback(valid_data, valid_mtx, cluster, combination, out_path+"/__train__/models/", train_log, img_type)
# model training
model.fit(train_data, shuffle=True, epochs= n_epoch, batch_size= n_batch, callbacks=[history, custom_callback])

log_message("training results:")
log_message(f"  best epoch: {custom_callback.epoch}")
log_message(f"  best similarity: {custom_callback.simil}")

with open(train_log, "a") as log: log.write("best epoch: {}\tbest similarity: {:.6f}\n".format(custom_callback.epoch, custom_callback.simil))


current_time = datetime.datetime.now()
out_disp = display_info("training completed "+str(current_time).split(".")[0])
log_message(out_disp)
################ training completed ################
# plot and save loss
print_loss(history, out_path + "/__train__/loss/", saveLoss=True)
plot_loss(history, out_path + "/__train__/loss/", saveImg=True)

# load the best model
model_path = out_path + "/__train__/models/model_{:0>3}.weights.h5".format(custom_callback.epoch)
vae = load_model(model_path, snp_dim, cat, latent_dim)

log_message("load the best model:")
log_message(f"  path: {model_path}")

# Calculate the similarity of train/test/all set prediction results
train_simil = data_predict_similarity(vae, train_data, train_mtx, cluster, img_type, combination)
test_simil = data_predict_similarity(vae, test_data, test_mtx, cluster, img_type, combination)
all_simil = data_predict_similarity(vae, data, mtx.values, cluster, img_type, combination)

with open(train_log, "a") as log: log.write("train similarity: {:.6f}\ntest similarity: {:.6f}\nall similarity: {:.6f}\n".format(
   train_simil, test_simil, all_simil
))

current_time = datetime.datetime.now()
out_disp = display_info("end "+str(current_time).split(".")[0])
log_message(out_disp)
log_message("Training completed successfully.")

# complete the training
with open(train_completed, "w") as f:
    f.write("Training completed successfully.\n")
