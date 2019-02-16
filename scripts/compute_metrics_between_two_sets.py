import sys
sys.path.insert(0, "/home/gy46/")
import numpy as np
import os.path as osp
from latent_3d_points.src.in_out import snc_category_to_synth_id,                                        load_all_point_clouds_under_folder

###############################################################################################
# Arguments
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('class_name', type=str, choices=['chair', 'car', 'airplane'],
                    help='Category for which we used to train AE. (right now only chair, car, airplane)')
parser.add_argument('--ref_outfname', type=str, default="ref_pcls_random_%s.npy")
parser.add_argument('--smp_outfname', type=str, default="smp_pcls_random_%s.npy")
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--normalize_shape', action='store_true',
                    help="Whether we normalize shape.")
parser.add_argument('--batch_size', type=int, default=100)
args = parser.parse_args()
print(args)
###############################################################################################

###############################################################################################
normalize_shape=args.normalize_shape
ae_loss = 'chamfer'  # Which distance to use for the matchings.
batch_size = 100     # Find appropriate number that fits in GPU.
normalize = True # Matched distances are divided by the number of points of thepoint-clouds.
top_in_dir = args.dataset_dir
class_name = args.class_name
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'val')
###############################################################################################

all_pc_data = load_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending='.npy', verbose=True, normalize=normalize_shape)

all_ids = np.arange(all_pc_data.num_examples)
print("Lane of all data ids:%d"%len(all_ids))

pidxs = np.random.choice(range(15000), 2048*2, replace=False)
tr_idxs = pidxs[:2048]
te_idxs = pidxs[2048:]
ref_pcs = all_pc_data.point_clouds[:,tr_idxs,:]
sample_pcs = all_pc_data.point_clouds[:,te_idxs,:]

print("Dump the output so that we can use other codes to evaluate it :(")
np.save(args.ref_outfname%args.class_name, ref_pcs)
np.save(args.smp_outfname%args.class_name, sample_pcs)

print("Class Name:%s"%class_name)
print("Use L3DP metrics:")
from latent_3d_points.src.evaluation_metrics_fast import MMD_COV_EMD_CD
mmd_emd, mmd_cd, cov_emd, cov_cd = MMD_COV_EMD_CD(
    sample_pcs, ref_pcs, args.batch_size, normalize=normalize, verbose=True)

print("MMD-EMD:%s"%mmd_emd)
print("MMD-CD:%s"%mmd_cd)
print("COV-EMD:%s"%cov_emd)
print("COV-CD:%s"%cov_cd)
print(args)

