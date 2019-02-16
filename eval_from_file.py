import argparse
import os
import torch

import sys
sys.path.insert(0, '/home/gy46/MVP')
import numpy as np
import random

# For evaluation
from pprint import pprint
from metrics.evaluation_metrics_pytorch import MMD_COV_EMD_CD

if __name__ == '__main__':

    ############################################################################################
    # Arguments: inference from file
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--ref_filename', type=str, default='scripts/ref_pcls.npy',
                        help='File containing both the reference')
    parser.add_argument('--smp_filename', type=str, default='scripts/smp_pcls.npy',
                        help='File containing both the sample')
    parser.add_argument('--one_to_one', action='store_true',
                        help="Whether using one-to-one evaluation.")
    parser.add_argument('--batch_size', type=int, default=100)
    args = parser.parse_args()
    print(args)
    ############################################################################################

    sample_pcl = torch.from_numpy(np.load(args.smp_filename)).cuda()
    print("Sample size:" + str(sample_pcl.shape))

    ref_pcl = torch.from_numpy(np.load(args.ref_filename)).cuda()
    print("Reference size:" + str(ref_pcl.shape))

    results = MMD_COV_EMD_CD(sample_pcl, ref_pcl, args.batch_size,
            one_to_one=args.one_to_one, accelerated_cd=True, verbose=True)
    pprint(results)
    results = {k:v.cpu().detach().item() for k,v in results.items()}
    pprint(results)

