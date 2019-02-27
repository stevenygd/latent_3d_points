import argparse
import os
import torch
import sys
import numpy as np
import random
sys.path.insert(0, '/home/gy46/')

############################################################################################
# Arguments: inference from file
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--ref_filename', type=str, default='scripts/ref_pcls.npy',
                    help='File containing both the reference')
parser.add_argument('--smp_filename', type=str, default='scripts/smp_pcls.npy',
                    help='File containing both the sample')
parser.add_argument('--one_to_one', action='store_true',
                    help="Whether using one-to-one evaluation.")
parser.add_argument('--use_fast_metric', action='store_true',
                    help="Whether using faster version.")
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
print(args)
############################################################################################

sample_pcl = np.load(args.smp_filename)
print("Sample size:" + str(sample_pcl.shape))

ref_pcl = np.load(args.ref_filename)
print("Reference size:" + str(ref_pcl.shape))

if args.use_fast_metric:
    from latent_3d_points.src.evaluation_metrics_fast import MMD_COV_EMD_CD
else:
    from latent_3d_points.src.evaluation_metrics import MMD_COV_EMD_CD
mmd_emd, mmd_cd, cov_emd, cov_cd = MMD_COV_EMD_CD(
        sample_pcl, ref_pcl, args.batch_size, verbose=True)
print("MMD-EMD:%s"%mmd_emd)
print("MMD-CD:%s"%mmd_cd)
print("COV-EMD:%s"%cov_emd)
print("COV-CD:%s"%cov_cd)

