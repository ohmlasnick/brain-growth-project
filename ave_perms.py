#!/usr/bin/env python3
#SBATCH --partition=SkylakePriority 
#SBATCH --account=roh17004
#SBATCH --qos=roh17004sky
#SBATCH --mail-type=END
#SBATCH --mail-type=ERROR
#SBATCH --mail-user=oliver.lasnick@uconn.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --time=12:00:00
#SBATCH -e error_bgp_svr_%A_%a.txt
#SBATCH -o output_bgp_svr_%A_%a.txt
#SBATCH --job-name=bgp_svr
#SBATCH --mem=16gb
##### END OF JOB DESCRIPTION #####

# %%
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
import math
import os
import re
from datetime import datetime

global DATE
global NUM_PERMUTATIONS

now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
DATE = year + '_' + month + '_' + day 

# Determine which model we're training (cutoff points for ROIs)
INPUT_KEY = int(sys.argv[1])
assert(INPUT_KEY >= 0)
assert(INPUT_KEY <= 6)

# Count the number of permutations in the current input dir
curr_dir = "./output/Cutoff_" + str(INPUT_KEY)
all_subdirs = os.listdir(curr_dir)
perm_dirs = [d for d in all_subdirs if ("Perm_" in d and os.path.isdir(curr_dir + '/' + d))]
NUM_PERMUTATIONS = len(perm_dirs)

# Load all phenotypic data and initialize dicts - each sub
# has its own dict with a x-val (age) and y-vals (preds)
rd_pheno = pd.read_csv(curr_dir + "/all_rd_pheno.csv")
td_pheno = pd.read_csv(curr_dir + "/all_td_pheno.csv")
sr_pheno = pd.read_csv(curr_dir + "/all_sr_pheno.csv")
rd_dict = []
td_dict = []
sr_dict = []

for d in perm_dirs:
    p_dir = curr_dir + '/' + d
    for root,dirs,files in os.walk(p_dir):
        model_pred_fles = [os.path.join(root,f) for f in files if "model_preds_demo" in f]
    if len(model_pred_fles) == 0:
        continue

    # For each perm file, map x-val and y-vals to subject ID
    for f in model_pred_fles:
        # Read files
        cols = ['Mapping_2', 'Age', 'Sex_F', 'Handedness_L', 'WISC_BD_Scaled',
       'WISC_FSIQ', 'TOWRE_PDE_Scaled', 'TOWRE_SWE_Scaled', 'SES',
       'Predicted_Age', 'Actual_Age']
        f_df = pd.read_csv(f, dtype=object, usecols=cols)
        if 'RD_' in f:
            group_dict = rd_dict
            group = 'RD'
        elif 'TD_' in f:
            group_dict = td_dict
            group = 'TD'
        elif 'SR_' in f:
            group_dict = sr_dict
            group = 'SR'

        sub_ids = f_df['Mapping_2']
        x_vals = f_df['Actual_Age']
        y_vals = f_df['Predicted_Age']
        ses, sex, dom_hand = f_df['SES'], f_df['Sex_F'], f_df['Handedness_L']
        wisc_bd, wisc_fsiq, pde, swe = f_df['WISC_BD_Scaled'], f_df['WISC_FSIQ'], f_df['TOWRE_PDE_Scaled'], f_df['TOWRE_SWE_Scaled']
        for i in range(len(sub_ids)):
            group_dict += [np.array([group, sub_ids[i], ses[i], sex[i], dom_hand[i], wisc_bd[i], wisc_fsiq[i], pde[i], swe[i], x_vals[i], y_vals[i]])]
    print("")
    print(np.shape(np.array(rd_dict)))
    print(np.shape(np.array(td_dict)))
    print(np.shape(np.array(sr_dict)))

# Save all perm data for a group in single file
rd_df = pd.DataFrame(np.array(rd_dict))
td_df = pd.DataFrame(np.array(td_dict))
sr_df = pd.DataFrame(np.array(sr_dict))
all_df = pd.concat([rd_df, td_df, sr_df])
all_df.to_csv(curr_dir + "/combined_perms_ALL.csv")

# Average preds by sub, then store in single file
print("Averaging all model predictions...")
rd_mean = []
for sub in list(rd_pheno['Mapping_2']):
    print("")
    print(sub)
    reduced_sub = [i for i in rd_dict if sub in i]
    arr_data = np.array(reduced_sub)
    num_data = arr_data[:,-1]
    mean_arr = [np.mean(num_data.astype(float))]
    rd_mean += [np.array(list(reduced_sub[0][:len(reduced_sub[0])-1]) + mean_arr)]

td_mean = []
for sub in list(td_pheno['Mapping_2']):
    print("")
    print(sub)
    reduced_sub = [i for i in td_dict if sub in i]
    arr_data = np.array(reduced_sub)
    num_data = arr_data[:,-1]
    mean_arr = [np.mean(num_data.astype(float))]
    td_mean += [np.array(list(reduced_sub[0][:len(reduced_sub[0])-1]) + mean_arr)]

sr_mean = []
for sub in list(sr_pheno['Mapping_2']):
    print("")
    print(sub)
    reduced_sub = [i for i in sr_dict if sub in i]
    arr_data = np.array(reduced_sub)
    num_data = arr_data[:,-1]
    mean_arr = [np.mean(num_data.astype(float))]
    sr_mean += [np.array(list(reduced_sub[0][:len(reduced_sub[0])-1]) + mean_arr)]

print("")
print(np.array(rd_mean))
print(np.shape(np.array(rd_mean)))
averaged_rd_df = pd.DataFrame(np.array(rd_mean))

print("")
print(np.array(td_mean))
print(np.shape(np.array(td_mean)))
averaged_td_df = pd.DataFrame(np.array(td_mean))

print("")
print(np.array(sr_mean))
print(np.shape(np.array(sr_mean)))
averaged_sr_df = pd.DataFrame(np.array(sr_mean))

all_averaged_df = pd.concat([averaged_rd_df, averaged_td_df, averaged_sr_df])
all_averaged_df.to_csv(curr_dir + "/averaged_perms_ALL.csv")

print("Done processing {} permutations".format(NUM_PERMUTATIONS))

