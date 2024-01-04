import argparse
from scipy.stats import ttest_ind_from_stats
from utils import *

COMPUTE_FOR_CORRECTED_STD = True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--avg1", type=float)
    parser.add_argument("--std1", type=float)
    parser.add_argument("--avg2", type=float)
    parser.add_argument("--std2", type=float)
    args = parser.parse_args()

    avgval1 = args.avg1
    stdval1 = args.std1
    avgval2 = args.avg2
    stdval2 = args.std2

    N = len(SEEDS)

    if COMPUTE_FOR_CORRECTED_STD:
        print("correcting for std estimators...")
        stdval1 = np.sqrt(N/(N-1)*(stdval1**2))
        stdval2 = np.sqrt(N/(N-1)*(stdval2**2))
    
    stats_res, pval_res = ttest_ind_from_stats(mean1=avgval1, std1=stdval1, nobs1=N,
                                                mean2=avgval2, std2=stdval2, nobs2=N, 
                                                equal_var=False, alternative='greater')

    print("P value for the first mean being higher: ", pval_res)