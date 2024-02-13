#!/usr/bin/env python3
#SBATCH --partition=SkylakePriority 
#SBATCH --account=roh17004
#SBATCH --qos=roh17004sky
#SBATCH --mail-type=END
#SBATCH --mail-type=ERROR
#SBATCH --mail-user=oliver.lasnick@uconn.edu
#SBATCH --nodes=1
#SBATCH --ntasks=24
#SBATCH --time=7-00:00:00
#SBATCH -e error_bgp_svr_%A_%a.txt
#SBATCH -o output_bgp_svr_%A_%a.txt
#SBATCH --job-name=bgp_svr
#SBATCH --mem=96gb
##### END OF JOB DESCRIPTION #####

# %%
import sys
import joblib
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import math
import os
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime

global DATE
global NUM_PERMUTATIONS

now = datetime.now()
year = now.strftime("%Y")
month = now.strftime("%m")
day = now.strftime("%d")
DATE = year + '_' + month + '_' + day 


# Read in all data and transpose so cols = feats, rows = subs
rd_data = pd.read_csv('./data/pairwise_corrs_RD_all.csv', low_memory=False)
rd_data = rd_data.set_index('Unnamed: 0').T
sr_data = pd.read_csv('./data/pairwise_corrs_SR_all.csv', low_memory=False)
sr_data = sr_data.set_index('Unnamed: 0').T
td_data = pd.read_csv('./data/pairwise_corrs_TD_all.csv', low_memory=False)
td_data = td_data.set_index('Unnamed: 0').T

# Determine which model we're training (cutoff points for ROIs)
INPUT_KEY = int(sys.argv[1])
NUM_PERMUTATIONS = abs(int(sys.argv[2]))
assert(NUM_PERMUTATIONS > 0)
assert(INPUT_KEY >= 0)
assert(INPUT_KEY <= 6)

roi_sets = dict()
roi_sets['0'] = [str(z) for z in range(1,401)]
roi_sets['1'] = ['1', '2', '3', '6', '7', '8', '9', '12', '14', '16', '17', '19', '21', '22', '23', '25', '32', '33', '37', '38', '40', '41', '43', '45', '47', '50', '52', '56', '69', '70', '71', '72', '74', '75', '77', '78', '79', '81', '82', '86', '90', '91', '92', '93', '95', '96', '97', '99', '101', '103', '104', '105', '110', '117', '123', '124', '129', '131', '132', '133', '136', '140', '141', '143', '147', '148', '151', '153', '154', '155', '156', '157', '158', '159', '160', '162', '163', '164', '166', '167', '169', '172', '173', '175', '177', '178', '181', '182', '184', '185', '187', '189', '192', '197', '200', '204', '208', '209', '218', '230', '232', '233', '246', '248', '249', '255', '262', '269', '271', '272', '274', '280', '281', '284', '293', '296', '297', '299', '306', '309', '310', '311', '312', '314', '330', '335', '337', '339', '340', '344', '345', '346', '347', '350', '355', '360', '361', '362', '363', '367', '370', '372', '373', '374', '376', '378', '382', '388', '395', '397', '399', '400']
roi_sets['2'] = ['2', '3', '6', '8', '9', '33', '37', '38', '43', '47', '50', '69', '70', '71', '72', '75', '77', '78', '90', '91', '93', '96', '99', '101', '103', '105', '110', '129', '131', '132', '133', '136', '140', '141', '148', '153', '154', '156', '157', '158', '159', '160', '162', '164', '167', '169', '172', '175', '181', '184', '185', '187', '189', '197', '204', '232', '246', '248', '274', '280', '293', '296', '297', '306', '311', '312', '314', '335', '337', '339', '340', '346', '347', '350', '370', '372', '374', '378']
roi_sets['3'] = ['3', '8', '33', '37', '38', '47', '69', '70', '71', '77', '78', '90', '96', '99', '105', '110', '129', '132', '133', '136', '140', '153', '156', '157', '158', '159', '160', '162', '164', '172', '175', '189', '197', '204', '232', '246', '248', '297', '314', '340', '346', '347', '374']
roi_sets['4'] = ['3', '8', '33', '37', '47', '69', '70', '71', '90', '96', '105', '110', '129', '133', '136', '140', '153', '156', '157', '158', '159', '160', '172', '175', '197', '314', '347']
roi_sets['5'] = ['3', '8', '37', '47', '69', '70', '71', '90', '96', '105', '133', '140', '156', '157', '158', '172', '175']
roi_sets['6'] = ['8', '47', '69', '70', '71', '90', '96', '133', '140', '157', '158', '172', '175']

def invert_PCA(pca_model):
    # number of components
    n_pcs = pca_model.components_.shape[0]

    # get the index of the most important feature on EACH component
    most_important = [np.abs(pca_model.components_[i]).argmax() for i in range(n_pcs)]

    # get the names
    initial_feature_names = list(pd.read_csv("./data/top_" + str(len(roi_sets[str(INPUT_KEY)])) + "_feats_ordered.csv")['features'])
    most_important_names = [initial_feature_names[most_important[i]] for i in range(n_pcs)]

    # Create dictionary with all numbered PCs and their names
    dic = {'PC{}'.format(i): most_important_names[i] for i in range(n_pcs)}

    # build the dataframe
    df = pd.DataFrame(dic, index=list(range(len(dic.items()))))
    df.to_csv(PERM_FOLDER + "/most_sig_PCs_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv")

    feat_weight_df = pd.DataFrame(np.transpose(pca_model.components_), columns=["PC-{}".format(i) for i in range(n_pcs)], index=list(initial_feature_names))
    feat_weight_df.to_csv(PERM_FOLDER + "/PC_feat_coeffs_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv")

# Map ROIs to string names in Schaefer parcellation
a_priori_roi_ind = roi_sets[str(INPUT_KEY)]
roi_ind_fle = open('./data/roi_ind_map.txt', 'r')
roi_ind_map = roi_ind_fle.readlines()
roi_dict = dict()
for mapping in roi_ind_map:
    ind = mapping.split()[0]
    name_prts = mapping.split()[1].split("_")[1:]
    if len(name_prts) == 3:
        str_name = "400p_7NTW2m." + name_prts[1] + "." + name_prts[2] + "\\t(" + name_prts[0][0] + ")"
    else:
        str_name = "400p_7NTW2m." + name_prts[1] + "." + name_prts[2] + "." + name_prts[3] + "\\t(" + name_prts[0][0] + ")"
    roi_dict[ind] = str_name

# Pheno / demographic dataframe
pheno_df = pd.read_csv('./data/ALL_SUBS_PHENO_SES.csv')
pheno_df_identifiers = ['sub-' + i.split(',')[0] for i in pheno_df['Identifiers']]

# Filter all data by whether it's included in phenotypic dataframe
rd_data = rd_data[rd_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])
sr_data = sr_data[sr_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])
td_data = td_data[td_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])

# Set columns using pheno data - for each ID, get the corresponding pheno
# data from pheno_df and add as a new column to [rd/td/sr]_data
demo_vars = ['Age', 'Sex_F', 'Handedness_L', 'WISC_BD_Scaled', 'WISC_FSIQ', 'TOWRE_PDE_Scaled', 'TOWRE_SWE_Scaled', 'SES']
for var in demo_vars:
    rd_var, sr_var, td_var = [], [], []
    for sub in rd_data['Mapping_2']:
        new_str = sub.split('-')[1].strip() + ',assessment'
        sub_var_col = list(pheno_df[pheno_df['Identifiers'] == new_str][var])
        if len(sub_var_col) > 0:
            sub_var = sub_var_col[0]
        else:
            sub_var = '.'
        rd_var += [sub_var]
    for sub in sr_data['Mapping_2']:
        new_str = sub.split('-')[1].strip() + ',assessment'
        sub_var_col = list(pheno_df[pheno_df['Identifiers'] == new_str][var])
        if len(sub_var_col) > 0:
            sub_var = sub_var_col[0]
        else:
            sub_var = '.'
        sr_var += [sub_var]
    for sub in td_data['Mapping_2']:
        new_str = sub.split('-')[1].strip() + ',assessment'
        sub_var_col = list(pheno_df[pheno_df['Identifiers'] == new_str][var])
        if len(sub_var_col) > 0:
            sub_var = sub_var_col[0]
        else:
            sub_var = '.'
        td_var += [sub_var]
    rd_data[var] = rd_var
    sr_data[var] = sr_var
    td_data[var] = td_var

# Get full list of features for the given ROI cutoff
num_feats = 0
feat_lst = []
a_priori_rois = [roi_dict[i] for i in a_priori_roi_ind]
for col in rd_data.columns:
    if '--' not in col:
        continue
    roi_pair = col.split('--')
    roi_1 = (roi_pair[0].split(')')[0] + ')')[2:]
    roi_2 = (roi_pair[1].split(')')[0] + ')')[2:]
    if roi_1 in a_priori_rois and roi_2 in a_priori_rois:
        num_feats += 1
        feat_lst += [col]

# Filter all dataframes by the included features
ids_feats_lst = ['Mapping_2'] + feat_lst
ids_demos_lst = ['Mapping_2'] + demo_vars

a_priori_rd = rd_data[ids_feats_lst]
a_priori_td = td_data[ids_feats_lst]
a_priori_sr = sr_data[ids_feats_lst]

a_priori_rd.to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_rd_features_only.csv")
a_priori_td.to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_td_features_only.csv")
a_priori_sr.to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_sr_features_only.csv")

rd_data[ids_demos_lst].to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_rd_pheno.csv")
td_data[ids_demos_lst].to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_td_pheno.csv")
sr_data[ids_demos_lst].to_csv("./output/Cutoff_" + str(INPUT_KEY) + "/all_sr_pheno.csv")

### Scale all data, then do PCA on all data (training + testing) ###

# Combine features (with IDs) for all groups into single df
# Separate IDs and data
all_X = pd.concat([rd_data[ids_feats_lst + demo_vars],
    td_data[ids_feats_lst + demo_vars], 
    sr_data[ids_feats_lst + demo_vars]])
all_X_groups = ['RD']*len(rd_data) + ['TD']*len(td_data) + ['SR']*len(sr_data)
all_X_ids = all_X['Mapping_2']
all_X_age = all_X['Age']
all_X_sex = all_X['Sex_F']
all_X_hand = all_X['Handedness_L']
all_X_bd = all_X['WISC_BD_Scaled']
all_X_fsiq = all_X['WISC_FSIQ']
all_X_pde = all_X['TOWRE_PDE_Scaled']
all_X_swe = all_X['TOWRE_SWE_Scaled']
all_X_ses = all_X['SES']
all_X_no_ids = all_X[feat_lst]
print("Number of raw features: ", len(all_X_no_ids.columns))

all_X_PCA = all_X
all_X_PCA.columns = all_X_PCA.columns.map(str)
all_X_PCA['Group'] = all_X_groups

### EVERYTHING BELOW THIS LINE GETS PERMUTED IN A LOOP ###

for p_num in range(NUM_PERMUTATIONS):
    print("PERMUTATION {}".format(p_num))

    PERM_FOLDER = "./output/Cutoff_" + str(INPUT_KEY) + "/Perm_" + str(p_num)

    if not os.path.isdir(PERM_FOLDER):
        os.mkdir(PERM_FOLDER)
    else:
        print("Already exists, skipping Permutation {}".format(p_num))
        continue

    # Grid search parameters
    C_range = np.logspace(-2, 4, 7)
    parameters = {'kernel': ['linear'], 'C': C_range}

    ### CONSTRUCT SVR MODEL WITH LINEAR KERNEL ###
    ### TRAIN ON CORRELATION MATRIX DATA TO PREDICT AGE ###

    # Create empty model
    svr_lin = SVR(epsilon=0.1, max_iter=100000)
        
    # Perform grid search with above set parameters
    grid_svr_lin = GridSearchCV(svr_lin, param_grid=parameters, cv=5, return_train_score=True)

    # Create training and testing data from TDs
    non_feat_cols = ['Mapping_2','Group'] + demo_vars

    td_filt = all_X_PCA[all_X_PCA['Group'] == 'TD'].dropna(subset=['Age'])
    td_X = td_filt
    td_Y = td_filt['Age']
    td_train_X, td_test_X, td_train_Y, td_test_Y = train_test_split(td_X, td_Y,
                                                                    test_size=0.33,
                                                                    random_state=p_num)

    # Create training and testing data from RDs
    rd_filt = all_X_PCA[all_X_PCA['Group'] == 'RD'].dropna(subset=['Age'])
    rd_X = rd_filt
    rd_Y = rd_filt['Age']
    rd_train_X, rd_test_X, rd_train_Y, rd_test_Y = train_test_split(rd_X, rd_Y,
                                                                    test_size=0.33,
                                                                    random_state=p_num)

    # Create training and testing data from SRs
    sr_filt = all_X_PCA[all_X_PCA['Group'] == 'SR'].dropna(subset=['Age'])
    sr_X = sr_filt
    sr_Y = sr_filt['Age']
    sr_train_X, sr_test_X, sr_train_Y, sr_test_Y = train_test_split(sr_X, sr_Y,
                                                                    test_size=0.33,
                                                                    random_state=p_num)

    # Recombine data from all groups after proportional splitting
    train_X = pd.concat([td_train_X, rd_train_X, sr_train_X])
    train_Y = list(td_train_Y) + list(rd_train_Y) + list(sr_train_Y)

    # Save split training + test data separately (with demos)
    train_X.to_csv(PERM_FOLDER + "/training_data_pre_pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    td_test_X.to_csv(PERM_FOLDER + "/TD_test_data_pre_pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    sr_test_X.to_csv(PERM_FOLDER + "/SR_test_data_pre_pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    rd_test_X.to_csv(PERM_FOLDER + "/RD_test_data_pre_pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    print("")
    print("Split training and testing data. All data saved. Example:")
    print("")

    # Separate demographic from training data
    train_demo = train_X[ids_demos_lst]
    td_test_demo = td_test_X[ids_demos_lst]
    rd_test_demo = rd_test_X[ids_demos_lst]
    sr_test_demo = sr_test_X[ids_demos_lst]

    train_X = train_X[feat_lst]
    td_test_X = td_test_X[feat_lst]
    rd_test_X = rd_test_X[feat_lst]
    sr_test_X = sr_test_X[feat_lst]

    print(train_demo.head())
    print(train_X.head())

    # Create PCA model and extract principal componenents
    pca = PCA(.95)
    pca.fit(train_X)
    print("Done extracting components.")
    train_X_PCA = pd.DataFrame(pca.transform(train_X))
    print("Done with transform of training data: ")
    print(train_X_PCA.head())

    # Scale all data
    scaler = StandardScaler()
    scaler.fit(train_X_PCA)
    train_X_PCA = pd.DataFrame(scaler.transform(train_X_PCA))
    print("Done rescaling: ")
    print("")
    print(train_X_PCA.head())
    
    # Save all PCA transform files
    pd.DataFrame(train_X_PCA).to_csv(PERM_FOLDER + "/training_data_POST-pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    #file_pi = open(PERM_FOLDER + "/pca_pickle" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".pkl", 'wb')
    #pickle.dump(pca, file_pi)
    #joblib.dump(pca, PERM_FOLDER + "/pca_jobdump_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".joblib")
    invert_PCA(pca)
    print("Saved PCA transform info.")

    # Train model
    print("")
    print("### TRAINING MODEL... ###")
    print("")
    fitted_model = grid_svr_lin.fit(train_X_PCA, train_Y)
    print("Done training. Running model on test data.")

    pca_test_sr = pd.DataFrame(scaler.transform(pd.DataFrame(pca.transform(sr_test_X))))
    pca_test_rd = pd.DataFrame(scaler.transform(pd.DataFrame(pca.transform(rd_test_X))))
    pca_test_td = pd.DataFrame(scaler.transform(pd.DataFrame(pca.transform(td_test_X))))
    pca_test_sr.to_csv(PERM_FOLDER + "/SR_test_data_POST-pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    pca_test_rd.to_csv(PERM_FOLDER + "/RD_test_data_POST-pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    pca_test_td.to_csv(PERM_FOLDER + "/TD_test_data_POST-pca_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)

    predicted_td = fitted_model.predict(pca_test_td)
    predicted_sr = fitted_model.predict(pca_test_sr)
    predicted_rd = fitted_model.predict(pca_test_rd)

    # Calculate z-score for predictions then multiply by std of actual age, then add mean
    print("Done. Calculating z-scores and standardizing predictions.")
    both_test_Y = np.array(list(sr_test_Y) + list(rd_test_Y) + list(td_test_Y))
    sd_actual, mean_actual = np.std(both_test_Y), np.mean(both_test_Y)
    both_z_score = stats.zscore(list(predicted_sr) + list(predicted_rd) + list(predicted_td))

    sr_std_ages = both_z_score[:len(predicted_sr)] * sd_actual + mean_actual
    rd_std_ages = both_z_score[len(predicted_sr):len(predicted_sr) + len(predicted_rd)] * sd_actual + mean_actual
    td_std_ages = both_z_score[len(predicted_sr) + len(predicted_rd):] * sd_actual + mean_actual

    ### Save coefficients and other information on SVR + PCA
    coef_dict = dict()
    for coef, feat in zip(fitted_model.best_estimator_.coef_, list(train_X_PCA.columns)):
        coef_dict[str(feat)] = coef
    coef_df = pd.DataFrame.from_dict(coef_dict)
    coef_df.to_csv(PERM_FOLDER + '/svr_model_coeff_' + DATE + '_Cutoff-' + str(INPUT_KEY) + '.csv', index=False)
    print("Done. Saved model coefficients.")

    td_test_demo['Predicted_Age'] = td_std_ages
    rd_test_demo['Predicted_Age'] = rd_std_ages
    sr_test_demo['Predicted_Age'] = sr_std_ages

    td_test_demo['Actual_Age'] = td_test_Y
    rd_test_demo['Actual_Age'] = rd_test_Y
    sr_test_demo['Actual_Age'] = sr_test_Y

    td_test_demo.to_csv(PERM_FOLDER + "/TD_model_preds_demo_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    rd_test_demo.to_csv(PERM_FOLDER + "/RD_model_preds_demo_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    sr_test_demo.to_csv(PERM_FOLDER + "/SR_model_preds_demo_" + DATE + "_Cutoff-" + str(INPUT_KEY) + ".csv", index=False)
    print("Saved model output predictions. Ending Permutation {} and saving plots.".format(p_num))

    fig = plt.subplots(figsize=(10, 10))

    # Generate plots
    plt.plot(td_test_Y, td_std_ages, 'b.', label='Normal readers (TD)')
    plt.plot(np.unique(td_test_Y),
            np.poly1d(np.polyfit(td_test_Y, td_std_ages, 1))(np.unique(td_test_Y)),
            'b')
    plt.plot(sr_test_Y, sr_std_ages, 'y.', label='Excellent readers (SR)')
    plt.plot(np.unique(sr_test_Y),
            np.poly1d(np.polyfit(sr_test_Y, sr_std_ages, 1))(np.unique(sr_test_Y)),
            'y')
    plt.plot(rd_test_Y, rd_std_ages, 'r.', label='Poor readers (RD)')
    plt.plot(np.unique(rd_test_Y),
             np.poly1d(np.polyfit(rd_test_Y, rd_std_ages, 1))(np.unique(rd_test_Y)),
             'r')

    # Plot linear line
    plt.plot([6, 17], [6, 17], color = 'k', linestyle = '--', label='Perfect prediction')

    plt.xlabel('Actual Age (Years)', fontname='Arial', fontsize=18)
    plt.ylabel('Standardized Model Prediction in Years (Z-score x $\sigma$ + $\mu$)', fontname='Arial', fontsize=18)
    plt.legend(loc='lower right', prop={'family': 'Arial', 'size': 12})
    plt.xticks(fontname='Arial', fontsize=18)
    plt.yticks(fontname='Arial', fontsize=18)

    plt.savefig(PERM_FOLDER + "/model_pred_plot_" + DATE + "_Cutoff-" + str(INPUT_KEY) + "_Perm_" + str(p_num) + ".png")
    plt.close()
    print("Saved figure.")



