import argparse
import os
from scipy.stats import ttest_ind_from_stats
from utils import *

ALL_DIDS = [11, 14, 15, 16, 18, 22,
            23, 29, 31, 37, 50, 54, 
            188, 458, 469, 1049, 1050, 1063, 
            1068, 1510, 1494, 1480, 1462, 1464, 
            6332, 23381, 40966, 40982, 40994, 40975]

if __name__ == "__main__":
    latex_table = ""
    for did in ALL_DIDS:
        if not os.path.isdir(os.path.join(RESULT_DIR, f"DID_{did}")):
            print(f"Dataset {did} results not available! Skipped.")
            continue
        accurs, accurs_avg, accurs_std = {}, {}, {}
        for method in ALL_METHODS:
            vals = np.load(os.path.join(RESULT_DIR, f"DID_{did}", f"{method}_accuracies.npy"))
            accurs[method] = vals
            accurs_avg[method] = np.mean(vals)
            accurs_std[method] = np.std(vals)

        # firstly, write down the mean and std results
        latex_table += f"\multirow{{3}}{{*}}{{ ({did}) }} & Random & " 
        # add in no pretrain results
        latex_table += f"\multirow{{3}}{{*}}{{ ${accurs_avg['no_pretrain']:.2f}\pm {accurs_std['no_pretrain']:.2f}$ }} & "
        # add in results under random features corruption
        latex_table += f"${accurs_avg['rand_corr-rand_feats']:.2f}\pm {accurs_std['rand_corr-rand_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['cls_corr-rand_feats']:.2f}\pm {accurs_std['cls_corr-rand_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['orc_corr-rand_feats']:.2f}\pm {accurs_std['orc_corr-rand_feats']:.2f}$ \\\\ \n"
        # add in results under least-correlated features corruption
        latex_table += f"&Least-Corr & & ${accurs_avg['rand_corr-leastCorr_feats']:.2f}\pm {accurs_std['rand_corr-leastCorr_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['cls_corr-leastCorr_feats']:.2f}\pm {accurs_std['cls_corr-leastCorr_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['orc_corr-leastCorr_feats']:.2f}\pm {accurs_std['orc_corr-leastCorr_feats']:.2f}$ \\\\ \n"
        # add in results under most-correlated features corruption
        latex_table += f"&Most-Corr & & ${accurs_avg['rand_corr-mostCorr_feats']:.2f}\pm {accurs_std['rand_corr-mostCorr_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['cls_corr-mostCorr_feats']:.2f}\pm {accurs_std['cls_corr-mostCorr_feats']:.2f}$ & "
        latex_table += f"${accurs_avg['orc_corr-mostCorr_feats']:.2f}\pm {accurs_std['orc_corr-mostCorr_feats']:.2f}$ \\\\ \n"
        # finishing the dataset
        latex_table += "\hline \n"


    # write the results to a file
    with open(os.path.join(RESULT_DIR, "accuracies.tex"), "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {os.path.join(RESULT_DIR, 'accuracies.tex')} file!")