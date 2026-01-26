"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Preparation of training data.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        In Deeplex's pipeline mode, the steps are executed sequentially in the following order: prepare.py, deeplex.py, prediction.py, and doublet.py. You can use the -H option to get the help message for each step.
    """)
parser.add_argument("-H", "--HELP", action="store_true", help="Help message for each step")
parser.add_argument("-g", "--genome", default="reference.fasta", help="Reference genome")
parser.add_argument("-b", "--bam", default="possorted_genome_bam.bam", help="BAM file")
parser.add_argument("-c", "--barcode", default="barcodes.txt", help="Barcode list")
parser.add_argument("-o", "--out", default=".", help="Output Directory")

# parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
parser.add_argument("-bn", "--batch_num", default=16, help="Number of batch, default 16")
parser.add_argument("-en", "--epoch_num", default=25, help="Number of epoch, default 25")
parser.add_argument("-ld", "--latent_dim", default=4, help="latent dimension, default 4")
parser.add_argument("-cn", "--cluster_num", default=3, help="Number of clusters, default 3") ##
parser.add_argument("-si", "--show_info", default="True", help="Whether to display detailed information, default True")
parser.add_argument("-tr", "--test_radio", default=0.2, help="Ratio of test set, default 0.2")
parser.add_argument("-vr", "--valid_radio", default=0.2, help="Ratio of validation set, default 0.2")
parser.add_argument("-s", "--seed", default=42, help="Random seed, default 42")
parser.add_argument("-lr", "--learning_rate", default=0.0001, help="Learning rate, default 0.0001")
parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")

# parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
# parser.add_argument("-vm", "--vae_model", default="not_provided", help="Trained VAE model")
# parser.add_argument("-ld", "--latent_dim", default=4, help="Latent dimension, default 4")
# parser.add_argument("-c", "--cluster", default=3, help="Number of clusters, default 3")
# parser.add_argument("-si", "--show_info", default="FALSE", help="Whether to display detailed information, default True")
# parser.add_argument("-s", "--seed", default=42, help="Random seed, default 42")
# parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")
# parser.add_argument("-o", "--out", default=".", help="Output Directory")
parser.add_argument("-p", "--prefix", default="vae_", help="Output prefix, default vae_")
parser.add_argument("-pc", "--plot_center", default="False", help="Do you want to draw cluster centers, default False")

# parser.add_argument("-m", "--mtx", default="matrix.txt", help="Matrix file")
# parser.add_argument("-s", "--single_pred", default="not_provided", help="tSNE_pred.txt, single cell prediction result file")
# parser.add_argument("-c", "--coordinate", default="not_provided", help="tSNE_result_data.txt, tSNE/UMAP result data file")
parser.add_argument("-f", "--fraction", default="not_provided", help="Is the percentage of doublet cells known in advance, default False")
# parser.add_argument("-it", "--img_type", default="tSNE", help="Image type (tSNE/UMAP), default tSNE")
# parser.add_argument("-o", "--out", default=".", help="Output Directory")
# parser.add_argument("-p", "--prefix", default="vae_", help="Output prefix, default vae_")

args = parser.parse_args()


################ parameter/arguments ################
genome_file = args.genome
bam_file = args.bam
barcode_file = args.barcode
out_path = args.out

# mtx_file = args.mtx
n_batch = args.batch_num
n_epoch = args.epoch_num
latent_dim = args.latent_dim
cluster = args.cluster_num ##
showInfo = args.show_info
test_radio = args.test_radio
valid_radio = args.valid_radio
seed = args.seed
learning_rate = args.learning_rate
img_type = args.img_type

prefix = args.prefix
plot_center = args.plot_center

doublet_percent = args.fraction

################ import packages ################
import sys
import subprocess
import datetime
from others import print_hash,display_info


# create directory
if out_path == "." or out_path == "./":
    result = subprocess.run("pwd", shell=True, capture_output=True, text=True)
    out_path = result.stdout.strip()
subprocess.run(f"mkdir -p {out_path}", shell=True)

log_file = f"{out_path}/deeplex_pipeline.log"
def log_message(message):
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
    print(message)
subprocess.run(f">{log_file}", shell=True)


################ help message ################
l = 100
if args.HELP:
    # prepare.py
    line = display_info("prepare.py", l)
    print_hash(len(line))
    print(line)
    print_hash(len(line))
    cmd = f"prepare.py -h"
    subprocess.run(cmd, shell=True)
    # deeplex.py
    line = display_info("deeplex.py", l)
    print_hash(len(line))
    print(line)
    print_hash(len(line))
    cmd = f"deeplex.py -h"
    subprocess.run(cmd, shell=True)
    # prediction.py
    line = display_info("prediction.py", l)
    print_hash(len(line))
    print(line)
    print_hash(len(line))
    cmd = f"prediction.py -h"
    subprocess.run(cmd, shell=True)
    # doublet.py
    line = display_info("doublet.py", l)
    print_hash(len(line))
    print(line)
    print_hash(len(line))
    cmd = f"doublet.py -h"
    subprocess.run(cmd, shell=True)

    sys.exit()


current_time = datetime.datetime.now()
out_disp = display_info("start "+str(current_time).split(".")[0], l)
log_message(out_disp + "\n")

################ Step1 execute prepare.py ################
cmd = f"prepare.py -g {genome_file} -b {bam_file} -c {barcode_file} -o {out_path}"
log_message(display_info("Step1 execute prepare.py", l) + "\n")
log_message(" Step1 : " + cmd + "\n")
subprocess.run(cmd, shell=True)

################ Step2 execute deeplex.py ################
mtx_file = f"{out_path}/__prepare__/vartrix/matrix_final.txt"
cmd = f"deeplex.py -m {mtx_file} -bn {n_batch} -en {n_epoch} -ld {latent_dim} -c {cluster} -si {showInfo} -tr {test_radio} -vr {valid_radio} -s {seed} -lr {learning_rate} -it {img_type} -o {out_path}"
log_message(display_info("Step2 execute deeplex.py", l) + "\n")
log_message(" Step2 : " + cmd + "\n")
subprocess.run(cmd, shell=True)

################ Step3 execute prediction.py ################
cmd = f"prediction.py -m {mtx_file} -ld {latent_dim} -c {cluster} -si {showInfo} -s {seed} -it {img_type} -o {out_path} -p {prefix} -pc {plot_center}"
log_message(display_info("Step3 execute prediction.py", l) + "\n")
log_message(" Step3 : " + cmd + "\n")
subprocess.run(cmd, shell=True)

################ Step4 execute doublet.py ################
cmd = f"doublet.py -m {mtx_file} -f {doublet_percent} -it {img_type} -o {out_path} -p {prefix}"
log_message(display_info("Step4 execute doublet.py", l) + "\n")
log_message(" Step4 : " + cmd + "\n")
subprocess.run(cmd, shell=True)

current_time = datetime.datetime.now()
out_disp = display_info("end "+str(current_time).split(".")[0], l)
log_message(out_disp)
