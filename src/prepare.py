"""
Author: Ogia Wei
Date: 2025-07-05
Version: 1.0
Description: Preparation of training data.
"""

import argparse
parser = argparse.ArgumentParser(description=
    """
        Preparation of training data.
        Provide a BAM file, a barcode list, and a reference genome as input to prepare data for model training.
    """)
parser.add_argument("-g", "--genome", default="reference.fasta", help="Reference genome")
parser.add_argument("-b", "--bam", default="possorted_genome_bam.bam", help="BAM file")
parser.add_argument("-c", "--barcode", default="barcodes.txt", help="Barcode list")
parser.add_argument("-o", "--out", default=".", help="Output Directory")
args = parser.parse_args()


################ parameter/arguments ################
genome_file = args.genome
bam_file = args.bam
barcode_file = args.barcode
out_path = args.out


################ import packages ################
import subprocess
import os

log_file = f"{out_path}/deeplex_prepare.log"
def log_message(message):
    with open(log_file, "a") as log:
        log.write(f"{message}\n")
    print(message)
subprocess.run(f">{log_file}", shell=True)

################ print parameter/arguments ################
log_message("parameter:")
log_message(f"  Reference genome: {genome_file}")
log_message(f"  BAM file: {bam_file}")
log_message(f"  Barcode list: {barcode_file}")
log_message(f"  Output Directory: {out_path}")


# create directory
if out_path == "." or out_path == "./":
    result = subprocess.run("pwd", shell=True, capture_output=True, text=True)
    out_path = result.stdout.strip()

subprocess.run(f"mkdir -p {out_path}/__prepare__", shell=True)
subprocess.run(f"mkdir -p {out_path}/__prepare__/freebayes", shell=True)
subprocess.run(f"mkdir -p {out_path}/__prepare__/vartrix", shell=True)

freebayes_completed = f"{out_path}/__prepare__/freebayes/freebayes.completed"
vcftools_completed = f"{out_path}/__prepare__/freebayes/vcftools.completed"
vartrix_completed = f"{out_path}/__prepare__/vartrix/vartrix.completed"
# Check if freebayes has been run
if os.path.exists(freebayes_completed):
    log_message("Freebayes has already been run. Skipping...")
else:
    # Run freebayes
    log_message("Running freebayes...")
    cmd = f"freebayes -f {genome_file} {bam_file} -C 2 -q 30 -n 3 -E 1 -m 30 --min-coverage 40 --pooled-continuous > {out_path}/__prepare__/freebayes/freebayes.vcf"
    subprocess.run(cmd, shell=True)
    log_message(f"Command: {cmd}")
    with open(freebayes_completed, "w") as f:
        f.write("Freebayes completed successfully.\n")
    log_message("Freebayes completed.")

# Check if vcftools has been run
if os.path.exists(vcftools_completed):
    log_message("VCFtools has already been run. Skipping...")
else:
    # Run vcftools
    log_message("Running VCFtools...")
    cmd = f"vcftools --vcf {out_path}/__prepare__/freebayes/freebayes.vcf --remove-indels --recode --recode-INFO-all --out {out_path}/__prepare__/freebayes/SNPs_rmindel"
    subprocess.run(cmd, shell=True)
    log_message(f"Command: {cmd}")
    with open(vcftools_completed, "w") as f:
        f.write("VCFtools completed successfully.\n")
    log_message("VCFtools completed.")

# Check if vartrix has been run
if os.path.exists(vartrix_completed):
    log_message("Vartrix has already been run. Skipping...")
else:
    # Run vartrix
    log_message("Running Vartrix...")
    cmd = f"vartrix_linux --scoring-method consensus --vcf {out_path}/__prepare__/freebayes/SNPs_rmindel.recode.vcf --bam {bam_file} --fasta {genome_file} --cell-barcodes {barcode_file} --threads 10 --out-matrix {out_path}/__prepare__/vartrix/vartrix.mtx"
    subprocess.run(cmd, shell=True)
    log_message(f"Command: {cmd}")
    with open(vartrix_completed, "w") as f:
        f.write("Vartrix completed successfully.\n")
    log_message("Vartrix completed.")

matrix_completed = f"{out_path}/__prepare__/vartrix/matrix.completed"
# Check if mtx2matrix has been run
if os.path.exists(matrix_completed):
    log_message("Matrix conversion has already been run. Skipping...")
else:
    # Convert mtx to matrix
    log_message("Converting mtx to matrix...")
    # Prepare SNPs loci file
    cmd = f"awk '{{print $1,$2}}' {out_path}/__prepare__/freebayes/SNPs_rmindel.recode.vcf > {out_path}/__prepare__/vartrix/SNPs.loci.txt"
    subprocess.run(cmd, shell=True)
    cmd = f"sed -i 's/\\s/:/g' {out_path}/__prepare__/vartrix/SNPs.loci.txt"
    subprocess.run(cmd, shell=True)
    cmd = f"/program/Deeplex/miniconda3/bin/python /program/Deeplex/bin/mtx2matrix.py -m {out_path}/__prepare__/vartrix/vartrix.mtx -s {out_path}/__prepare__/vartrix/SNPs.loci.txt -b {barcode_file} -o {out_path}/__prepare__/vartrix/matrix.txt"
    subprocess.run(cmd, shell=True)
    with open(matrix_completed, "w") as f:
        f.write("Matrix conversion completed successfully.\n")
    log_message("Matrix conversion completed.")


filter_completed = f"{out_path}/__prepare__/vartrix/filter.completed"
# Check if filter has been run
if os.path.exists(filter_completed):
    log_message("Matrix filtering has already been run. Skipping...")
else:
    # Filter matrix
    log_message("Filtering matrix...")
    cmd = f"/program/Deeplex/miniconda3/bin/python /program/Deeplex/bin/filter_matrix.py -m {out_path}/__prepare__/vartrix/matrix.txt -o {out_path}/__prepare__/vartrix/matrix_filter.txt"
    subprocess.run(cmd, shell=True)
    cmd = f"/program/Deeplex/miniconda3/bin/python /program/Deeplex/bin/fill_matrix.py -m {out_path}/__prepare__/vartrix/matrix_filter.txt -o {out_path}/__prepare__/vartrix/matrix_final.txt"
    subprocess.run(cmd, shell=True)
    with open(filter_completed, "w") as f:
        f.write("Matrix filtering completed successfully.\n")
    log_message("Matrix filtering completed.")

log_message("Preparation completed successfully.")
