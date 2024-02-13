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
#SBATCH --mem=48gb
##### END OF JOB DESCRIPTION #####

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms
import scipy.stats as stats
import math
import os
import glob
from scipy.optimize import curve_fit
from sklearn.svm import SVR
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from datetime import datetime
from collections import OrderedDict
from itertools import islice


# Set all necessary hardcoded variables (date, ROI lists)
 
INPUT_KEY = str(sys.argv[1])
DO_PERMS = bool(int(sys.argv[2]))
GET_FEATS = bool(int(sys.argv[3]))
CONCAT_FEATS = bool(int(sys.argv[4]))
CREATE_MATS = bool(int(sys.argv[5]))

roi_sets = dict()
roi_sets['0'] = [str(z) for z in range(1,401)]
roi_sets['1'] = ['1', '2', '3', '6', '7', '8', '9', '12', '14', '16', '17', '19', '21', '22', '23', '25', '32', 
                 '33', '37', '38', '40', '41', '43', '45', '47', '50', '52', '56', '69', '70', '71', '72', '74', 
                 '75', '77', '78', '79', '81', '82', '86', '90', '91', '92', '93', '95', '96', '97', '99', '101', 
                 '103', '104', '105', '110', '117', '123', '124', '129', '131', '132', '133', '136', '140', '141',
                 '143', '147', '148', '151', '153', '154', '155', '156', '157', '158', '159', '160', '162', '163', 
                 '164', '166', '167', '169', '172', '173', '175', '177', '178', '181', '182', '184', '185', '187', 
                 '189', '192', '197', '200', '204', '208', '209', '218', '230', '232', '233', '246', '248', '249', 
                 '255', '262', '269', '271', '272', '274', '280', '281', '284', '293', '296', '297', '299', '306', 
                 '309', '310', '311', '312', '314', '330', '335', '337', '339', '340', '344', '345', '346', '347', 
                 '350', '355', '360', '361', '362', '363', '367', '370', '372', '373', '374', '376', '378', '382', 
                 '388', '395', '397', '399', '400']
roi_sets['2'] = ['2', '3', '6', '8', '9', '33', '37', '38', '43', '47', '50', '69', '70', '71', '72', '75', '77', 
                 '78', '90', '91', '93', '96', '99', '101', '103', '105', '110', '129', '131', '132', '133', '136',
                 '140', '141', '148', '153', '154', '156', '157', '158', '159', '160', '162', '164', '167', '169', 
                 '172', '175', '181', '184', '185', '187', '189', '197', '204', '232', '246', '248', '274', '280', 
                 '293', '296', '297', '306', '311', '312', '314', '335', '337', '339', '340', '346', '347', '350', 
                 '370', '372', '374', '378']
roi_sets['3'] = ['3', '8', '33', '37', '38', '47', '69', '70', '71', '77', '78', '90', '96', '99', '105', '110', 
                 '129', '132', '133', '136', '140', '153', '156', '157', '158', '159', '160', '162', '164', '172', 
                 '175', '189', '197', '204', '232', '246', '248', '297', '314', '340', '346', '347', '374']
roi_sets['4'] = ['3', '8', '33', '37', '47', '69', '70', '71', '90', '96', '105', '110', '129', '133', '136', '140',
                 '153', '156', '157', '158', '159', '160', '172', '175', '197', '314', '347']
roi_sets['5'] = ['3', '8', '37', '47', '69', '70', '71', '90', '96', '105', '133', '140', '156', '157', '158', 
                 '172', '175']
roi_sets['6'] = ['8', '47', '69', '70', '71', '90', '96', '133', '140', '157', '158', '172', '175']

### BEGIN HELPER FUNCTIONS ###

def clean_string(s):
    """Takes in string s and reformats it with
    certain characters removed."""
    return s.replace('.','_').replace("['400p_7NTW2m","7Networks").replace("\\t","").replace("']","").replace('(','_').replace(')','')

class StringMatrix:
    """ 2D Array with strings as row, col indices."""

    def __init__(self, rows, cols):
        """Indices are dictionaries."""
        self.rows = rows
        self.cols = cols
        self.data = np.zeros((len(self.rows), len(self.cols)))
        self.row_index = dict([(w, i) for i, w in enumerate(self.rows)])
        self.col_index = dict([(w, i) for i, w in enumerate(self.cols)])

    def _parsekey(self, key):
        """Convert index key to indices."""
        w0, w1 = key[0], key[1]
        return self.row_index[w0], self.col_index[w1]

    def __getitem__(self, key):
        """Overload [] operator - get() method."""
        i0, i1 = self._parsekey(key)
        return self.data[i0][i1]

    def __setitem__(self, key, value):
        """Overload [] operator -  set() method."""
        i0, i1 = self._parsekey(key)
        self.data[i0][i1] = value
        
def inverse_binomial(M):
    """Returns n given M, the value of n choose 2."""
    M_root = int(math.ceil(math.sqrt(2 * M))) # whole int > sqrt(2M)
    n = M_root
    index, limit = 0, 100
    while not ((n**2 - n) == (2 * M)):
        if index >= limit:
            return "no solution, something's wrong: M = " + str(M)
        n += 1
        index += 1
    return n

def get_individual_feats(df):
    """Create and return list of strings with
    each element a feature name (clean string)."""
    individ_feats = []
    end = inverse_binomial(len(df)) - 1
    for f in list(df['features'])[:end]:
        split_feats = clean_string(f).split('--')[1]
        individ_feats += [split_feats]
    return [clean_string(list(df['features'])[0]).split('--')[0]] + individ_feats

def populate_matrix(df, fts):
    """Create an ROI x ROI StringMatrix object, populate it
    with each corr, and return it."""
    matrix = StringMatrix(fts, fts)
    features_col, coeff_col = list(df['features']), list(df['coeff'])
    for i in range(len(features_col)):
        row, col = clean_string(features_col[i]).split('--')
        cf = coeff_col[i]
        matrix[row, col] = cf
        matrix[col, row] = cf
    return matrix

### END HELPER FUNCTIONS ###

# Count the number of permutations in the current input dir
curr_dir = "./output/Cutoff_" + str(INPUT_KEY)
all_subdirs = os.listdir(curr_dir)
perm_dirs = [d for d in all_subdirs if ("Perm_" in d and os.path.isdir(curr_dir + '/' + d))]
NUM_PERMUTATIONS = len(perm_dirs)
outlier = lambda a: (a < (np.mean(a) - (3 * np.std(a)))) or (a > (np.mean(a) + (3 * np.std(a))))

if DO_PERMS:

### EVERYTHING BELOW THIS LINE DONE FOR EACH PERMUTATION ###

    for d in perm_dirs:
        print("")
        print(d)
        p_dir = curr_dir + '/' + d
        files = sorted(glob.glob(p_dir + "/*.csv"))
        print(files[0], files[-3])
        svr_coef = np.array(pd.read_csv(files[-3]).iloc[:, 0])
        feat_weight_arr = np.array(pd.read_csv(files[0]))[:,1:]
        initial_feature_names = list(pd.read_csv("./data/top_" + str(len(roi_sets[INPUT_KEY])) + "_feats_ordered.csv")['features'])
        n_pcs = len(svr_coef)

        print(svr_coef, n_pcs)
        print(feat_weight_arr, feat_weight_arr.shape)

        # Multiply all feats in feat_weight_arr (1 row = 1 PC, all feats) by that PC's SVR coeff
        multiplied_arr = (svr_coef * feat_weight_arr).T
        multiplied_df = pd.DataFrame(multiplied_arr.T, columns=["PC-{}".format(i) for i in range(n_pcs)], index=list(initial_feature_names))

        # Average coeffs for each feature across PCs (df transposed so 1 col = 1 feat)
        multiplied_df_average = multiplied_df.transpose().mean().to_frame().T
        feat_names = multiplied_df_average.columns
        coeffs = list(multiplied_df_average.transpose().iloc[:, 0])

        # Create and save transposed df with feats + coeffs
        feat_dict = {'features': list(feat_names), 'coeff': coeffs}
        feat_df = pd.DataFrame(feat_dict)
        feat_df.to_csv(p_dir + '/feat_names_coeffs_Cutoff-' + str(INPUT_KEY) + '.csv')

        # Add absolute values and list in rank order
        feat_df['abs_coeff'] = abs(feat_df['coeff'])
        ranked_feats = feat_df.sort_values(by=['abs_coeff'])
        ranked_feats.to_csv(p_dir + '/ranked_feats_Cutoff-' + str(INPUT_KEY) + '.csv')

        # Convert to 2d matrix
        feats = get_individual_feats(feat_df)
        matrix = populate_matrix(feat_df, feats)
        feats_matrix = pd.DataFrame(data=matrix.data, index=feats, columns=feats)
        feats_matrix.to_csv(p_dir + '/2d_feats_coeffs_Cutoff-' + str(INPUT_KEY) + '.csv')

else:
    print("Skipping per-permutation matrix generation.")

### BEGIN HELPER FUNCTIONS ###

def get_ave():
    """Returns average values of raw features
    (PCA_coef * SVR_coef) across permutations."""
    feat_arrs, feat_mats = [], []
    print("")
    print("Averaging features from permutations...")
    for d in perm_dirs:
        p_dir = curr_dir + '/' + d
        feat_arr_df = pd.read_csv(p_dir + '/feat_names_coeffs_Cutoff-' + str(INPUT_KEY) + '.csv')
        feat_mat_df = pd.read_csv(p_dir + '/2d_feats_coeffs_Cutoff-' + str(INPUT_KEY) + '.csv')
        print("")
        print("Processing Perm {}".format(d))
        feat_arr_cols = list(feat_arr_df['features'])
        feat_arrs += [np.array(feat_arr_df['coeff'])]
        feat_mats += [feat_mat_df.values.T[1:]]
        print("")
        print(feat_mat_df.values.T[1:])
        print(feat_mat_df.values.T[1:].shape)
    ave_feat_arr = np.mean(feat_arrs, axis=0)
    print("")
    print("Averages:")
    print("")
    print(np.array(ave_feat_arr))
    print(np.array(ave_feat_arr).shape)
    print("")
    print(np.array(feat_mats))
    print(np.array(feat_mats).shape)
    ave_feat_mat = np.mean(np.array(feat_mats), axis=0)
    ave_feat_arr_df = pd.DataFrame({'features': feat_arr_cols, 'coeffs': list(ave_feat_arr), 'abs_coeff': np.absolute(ave_feat_arr)})
    print(feat_mat_df.columns)
    ave_feat_mat_df = pd.DataFrame(ave_feat_mat, columns=list(feat_mat_df.columns[1:]), index=list(feat_mat_df.columns[1:]))
    return ave_feat_arr_df, ave_feat_mat_df

def combine_feat_coeff():
    feat_files = glob.glob(curr_dir + "/*/feat_names_coeffs_Cutoff-" + str(INPUT_KEY) + ".csv")
    pca_files = glob.glob(curr_dir + "/*/coeff_by_pca_feat.csv")
    print(feat_files)
    print(pca_files)
    feat_dfs = [pd.read_csv(f) for f in feat_files]
    pca_dfs = [pd.read_csv(f) for f in pca_files]
    feat_concat = pd.concat(feat_dfs)
    pca_concat = pd.concat(pca_dfs)
    feat_concat.to_csv(curr_dir + '/feat_coeffs_ALL_PERMS.csv')
    pca_concat.to_csv(curr_dir + '/pca_coeffs_ALL_PERMS.csv')

def get_group_corr_matrices():

    # Read in all data and transpose so cols = feats, rows = subs
    rd_data = pd.read_csv('./data/pairwise_corrs_RD_all.csv', low_memory=False)
    rd_data = rd_data.set_index('Unnamed: 0').T
    sr_data = pd.read_csv('./data/pairwise_corrs_SR_all.csv', low_memory=False)
    sr_data = sr_data.set_index('Unnamed: 0').T
    td_data = pd.read_csv('./data/pairwise_corrs_TD_all.csv', low_memory=False)
    td_data = td_data.set_index('Unnamed: 0').T

    # Pheno / demographic dataframe
    pheno_df = pd.read_csv('./data/ALL_SUBS_PHENO_SES.csv')
    pheno_df_identifiers = ['sub-' + i.split(',')[0] for i in pheno_df['Identifiers']]
    pheno_df['Identifiers'] = pheno_df_identifiers
    pheno_df = pheno_df.sort_values(by=['Identifiers'])
    rd_pheno = pheno_df[pheno_df['Group'] == 'RD']
    sr_pheno = pheno_df[pheno_df['Group'] == 'SR']
    td_pheno = pheno_df[pheno_df['Group'] == 'TD']

    # Filter all data by whether it's included in phenotypic dataframe
    rd_data = rd_data[rd_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])
    sr_data = sr_data[sr_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])
    td_data = td_data[td_data['Mapping_2'].isin(pheno_df_identifiers)].sort_values(by=['Mapping_2'])

    assert(list(rd_pheno['Identifiers']) == list(rd_data['Mapping_2']))
    assert(list(sr_pheno['Identifiers']) == list(sr_data['Mapping_2']))
    assert(list(td_pheno['Identifiers']) == list(td_data['Mapping_2']))

    print(rd_pheno)
    print(td_pheno)
    print(sr_pheno)

    rd_data['age_bin'] = list(rd_pheno['Age_Bin'])
    sr_data['age_bin'] = list(sr_pheno['Age_Bin'])
    td_data['age_bin'] = list(td_pheno['Age_Bin'])

    print(rd_data)
    print(td_data)
    print(sr_data)

    discard = ['Subject #', 'Mapping_1', 'Mapping_2', 'age_bin']

    # First, make 3 matrices for each group, collapsed across age
    rd_arr = np.array(rd_data.drop(discard, axis=1))
    print(rd_arr)
    rd_df = pd.DataFrame(rd_arr.transpose(), columns=['features', 'coeff'])
    print(rd_df)

     # Convert to 2d matrix
    feats = get_individual_feats(rd_df)
    matrix = populate_matrix(rd_df, feats)
    feats_matrix = pd.DataFrame(data=matrix.data, index=feats, columns=feats)
    print(feats_matrix)

    # Next, make 6 matrices for each age bin, collapsed across group

    # Finally, make 3 x 6 = 18 matrices for each Group-Bin combination

def get_top_rois(df):
    """Returns ranked ROIs based on presence
    in top-ranked features. Takes in dataframe
    df of all raw features with average coefs as
    defined in get_ave()."""
    ranked = df.sort_values(by=['abs_coeff'])
    print("")
    num_top = int(math.ceil(len(ranked) / 10.))
    print("Number of features in top 10%: {}".format(num_top))
    top_feat = ranked['features'][:num_top]
    top_rois = [clean_string(k) for k in flatten_lst([f.split('--') for f in top_feat])]
    print("")
    roi_hist = make_freq_dict(top_rois)
    return roi_hist

def flatten_lst(lst):
    flat_lst = []
    for i in lst:
        if type(i) == list:
            flat_lst += flatten_lst(i)
        else:
            flat_lst += [i]
    return flat_lst

def make_freq_dict(top):
    roi_freq = {}
    for r in top:
        if r in roi_freq:
            roi_freq[r] += 1
        else:
            roi_freq[r] = 1
    for k,v in roi_freq.items():
        print("".join(k.split("_")[1:-1]), k.split("_")[-1].split(','), ':', v)
    roi_hist = OrderedDict(sorted(roi_freq.items(), reverse=True, key=lambda x: x[1]))
    return roi_hist

def make_node_lst(roi_dict):
    node_labels = np.array(["".join(lab.split("_")[1:-1]) for lab in np.array(list(roi_dict.items()))[:,0]])
    coords = np.array([lab.split("_")[-1].split(',') for lab in np.array(list(roi_dict.items()))[:,0]])
    node_size = np.array(list(roi_dict.items()))[:,1]
    node_color = np.array([1]*len(node_labels))
    node_data = []
    for i in range(len(node_labels)):
        row = [list(coords[i]) + [node_color[i]] + [node_size[i]] + [node_labels[i]]]
        node_data += row
    node_fle = pd.DataFrame(node_data)
    node_fle.to_csv(curr_dir + '/nodes_brainNet_Cutoff-' + str(INPUT_KEY) + '.csv')

### END HELPER FUNCTIONS ###

# Get averages for pairwise feats + 2d feature matrix,
# then use to get list of ROIs with frequencies.

if GET_FEATS:

### EVERYTHING BELOW THIS LINE DONE FOR EACH PERMUTATION ###

    for d in perm_dirs:
        print("")
        print(d)
        p_dir = curr_dir + '/' + d
        files = sorted(glob.glob(p_dir + "/*.csv"))
        print(files[-3], files[-5])
        svr_coef = pd.read_csv(files[-3])
        print(svr_coef)
        most_sig_feat_PC = pd.read_csv(files[-5]).transpose()
        print(most_sig_feat_PC, most_sig_feat_PC.index)
        new_index = [int(s.split('PC')[1]) if 'PC' in s else -1 for s in most_sig_feat_PC.index]
        most_sig_feat_PC['new_index'] = new_index
        most_sig_feat_PC = most_sig_feat_PC.set_index('new_index').sort_index(ascending=True)
        print(most_sig_feat_PC)
        svr_coef['Top_feature'] = list(most_sig_feat_PC.iloc[:,1])[1:]
        svr_coef.to_csv(p_dir + '/coeff_by_pca_feat.csv')

if CONCAT_FEATS:
    combine_feat_coeff()

if CREATE_MATS:
    get_group_corr_matrices()

if not os.path.exists(curr_dir + '/AVE_feat_coeffs.csv'):
    ave_feat_arr_df, ave_feat_mat_df = get_ave()
    ave_feat_arr_df.to_csv(curr_dir + '/AVE_feat_coeffs.csv')
    ave_feat_mat_df.to_csv(curr_dir + '/AVE_coeff_matrix.csv')
else:
    ave_feat_arr_df, ave_feat_mat_df = pd.read_csv(curr_dir + '/AVE_feat_coeffs.csv'), pd.read_csv(curr_dir + '/AVE_coeff_matrix.csv')
    
make_node_lst(get_top_rois(ave_feat_arr_df))
