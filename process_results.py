import matplotlib.pyplot as plt
import os
from scipy.stats import ttest_ind
from utils import *
from tqdm import tqdm

ALL_DIDS = [11, 14, 15, 16, 18, 22,
            23, 29, 31, 37, 50, 54, 
            188, 458, 469, 1049, 1050, 1063, 
            1068, 1510, 1494, 1480, 1462, 1464, 
            6332, 23381, 40966, 40982, 40994, 40975]



if __name__ == "__main__":
    latex_table = "\hline \n Datasets (DID) & {} No-PreTrain & Random & Class & Oracle \\\\ \n\hline \n".format(
                    "Corrupted Features &" if 'cls_corr-leastCorr_feats' in ALL_METHODS else ""
    )
    win_mat = np.zeros(shape=[len(ALL_METHODS), len(ALL_METHODS)])
    datasets_list = openml.datasets.list_datasets(ALL_DIDS, output_format='dataframe')
    for did in ALL_DIDS:
        if not os.path.isdir(os.path.join(RESULT_DIR, f"DID_{did}")):
            print(f"Dataset {did} results not available! Skipped.")
            continue
        res_vals, res_avg, res_std = {}, {}, {}
        for method in ALL_METHODS:
            if METRIC == "accuracy":
                res_vals[method] = np.load(os.path.join(RESULT_DIR, f"DID_{did}", f"{method}_accuracies.npy"))
            else:
                res_vals[method] = np.load(os.path.join(RESULT_DIR, f"DID_{did}", f"{method}_aurocs.npy"))
            assert len(res_vals[method]) == len(SEEDS)
            res_avg[method] = np.mean(res_vals[method])
            res_std[method] = np.std(res_vals[method]) / np.sqrt(len(SEEDS))

        # Update the win matrix
        for i in range(len(ALL_METHODS)):
            for j in range(i+1, len(ALL_METHODS)):
                method_1, method_2 = ALL_METHODS[i], ALL_METHODS[j]
                # Accuracies
                # conduct Welch's t-test with unequal variances
                if res_avg[method_1] < res_avg[method_2]:
                    # Null hypothesis to be rejected: method_1 has higher mean
                    t_stat, p_val = ttest_ind(a=res_vals[method_1], 
                                              b=res_vals[method_2], 
                                              equal_var=False, 
                                              alternative='less')
                    if p_val < P_VAL_SIGNIFICANCE:
                        # Null hypothesis rejected, method_2 has higher mean
                        win_mat[j][i] += 1
                else:
                    # Null hypothesis to be rejected: method_2 has higher mean
                    t_stat, p_val = ttest_ind(a=res_vals[method_2], 
                                              b=res_vals[method_1], 
                                              equal_var=False, 
                                              alternative='less')
                    if p_val < P_VAL_SIGNIFICANCE:
                        # Null hypothesis rejected, method_1 has higher mean
                        win_mat[i][j] += 1

        # Write avg and std statistics in latex
        ds_name = datasets_list[datasets_list.did==did].name.item()
        ds_name = ds_name.replace("_", "-")
        if 'cls_corr-leastCorr_feats' not in ALL_METHODS:
            # Table only including comparison between methods on how to corruption 
            latex_table += f"{ds_name} ({did}) & " 
            # add in no pretrain results
            latex_table += f"${res_avg['no_pretrain']:.2f}\pm {res_std['no_pretrain']:.2f}$ & "
            # add in results under random corruption
            latex_table += f"${res_avg['rand_corr-rand_feats']:.2f}\pm {res_std['rand_corr-rand_feats']:.2f}$ & "
            # add in results under class-conditioned corruption
            latex_table += f"${res_avg['cls_corr-rand_feats']:.2f}\pm {res_std['cls_corr-rand_feats']:.2f}$ & "
            # add in results under oracle corruption
            latex_table += f"${res_avg['orc_corr-rand_feats']:.2f}\pm {res_std['orc_corr-rand_feats']:.2f}$ \\\\ \n"
        else:
            # Full table also including comparison to methods with feature correlations
            latex_table += f"\multirow{{3}}{{*}}{{ {ds_name} ({did}) }} & Random & " 
            # add in no pretrain results
            latex_table += f"\multirow{{3}}{{*}}{{ ${res_avg['no_pretrain']:.2f}\pm {res_std['no_pretrain']:.2f}$ }} & "
            # add in results under random features corruption
            latex_table += f"${res_avg['rand_corr-rand_feats']:.2f}\pm {res_std['rand_corr-rand_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-rand_feats']:.2f}\pm {res_std['cls_corr-rand_feats']:.2f}$ & "
            latex_table += f"${res_avg['orc_corr-rand_feats']:.2f}\pm {res_std['orc_corr-rand_feats']:.2f}$ \\\\ \n"
            # add in results under least-correlated features corruption
            latex_table += f"&Least-Corr & & ${res_avg['rand_corr-leastCorr_feats']:.2f}\pm {res_std['rand_corr-leastCorr_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-leastCorr_feats']:.2f}\pm {res_std['cls_corr-leastCorr_feats']:.2f}$ & "
            latex_table += f"${res_avg['orc_corr-leastCorr_feats']:.2f}\pm {res_std['orc_corr-leastCorr_feats']:.2f}$ \\\\ \n"
            # add in results under most-correlated features corruption
            latex_table += f"&Most-Corr & & ${res_avg['rand_corr-mostCorr_feats']:.2f}\pm {res_std['rand_corr-mostCorr_feats']:.2f}$ & "
            latex_table += f"${res_avg['cls_corr-mostCorr_feats']:.2f}\pm {res_std['cls_corr-mostCorr_feats']:.2f}$ & "
            latex_table += f"${res_avg['orc_corr-mostCorr_feats']:.2f}\pm {res_std['orc_corr-mostCorr_feats']:.2f}$ \\\\ \n"
      

    # process win matrices
    print(f"Win matrix for {METRIC}: \n", win_mat)
    win_mat_divisor = win_mat + np.transpose(win_mat) + np.eye(len(ALL_METHODS))
    win_mat = np.divide(win_mat, win_mat_divisor)

    fig, ax = plt.subplots()
    im = ax.imshow(win_mat)

    ax.set_xticks(np.arange(len(ALL_METHODS)))
    ax.set_yticks(np.arange(len(ALL_METHODS)))
    ax.set_xticklabels(['No Pre-Train', 'Conventional', 'Class-Conditioned', 'Oracle'])
    ax.set_yticklabels(['No Pre-Train', 'Conventional', 'Class-Conditioned', 'Oracle'])

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
            rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(len(ALL_METHODS)):
        for j in range(len(ALL_METHODS)):
            text = ax.text(j, i, f"{win_mat[i, j]*100:.1f}%",
                        ha="center", va="center", color="w")

    fig.tight_layout()
    plt.show()

    # write the results to a file
    # finishing the table by an underline
    latex_table += "\hline \n"  
    latex_table_filename = os.path.join(RESULT_DIR, 
                                        f"{METRIC}_table{'_full' if 'cls_corr-leastCorr_feats' in ALL_METHODS else ''}.tex")

    with open(latex_table_filename, "w") as f:
        f.write(latex_table)
    
    print(f"Latex table generated and saved to {latex_table_filename} file!")

    print("Script finished!")