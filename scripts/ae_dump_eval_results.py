# coding: utf-8

# ## This notebook will help you train a vanilla Point-Cloud AE with the basic architecture we used in our paper.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

import os
import sys
sys.path.insert(0, "/home/gy46/")

import numpy as np
import os.path as osp
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir
from latent_3d_points.src.in_out import PointCloudDataSet,load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

# Arguments
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('class_name', type=str, choices=['chair', 'car', 'airplane'],
                    help='Category for which we used to train AE. (right now only chair, car, airplane)')
parser.add_argument('train_dir', type=str, default=None,
                    help='Training directory (where we stored the model)')
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--ref_outfname', type=str, default="ref_pcls.npy")
parser.add_argument('--smp_outfname', type=str, default="smp_pcls.npy")
parser.add_argument('--expr_prefix', type=str, default='shapenetcorev2',
                    help='Prefix for the experiment.')
parser.add_argument('--ae_loss', type=str, default='chamfer', choices=['chamfer', 'emd'],
                    help='Loss to optimize for ([emd] or [chamfer]).')
parser.add_argument('--normalize_shape', action='store_true',
                    help="Whether normalizing shape.")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Training epochs.")
args = parser.parse_args()
print(args)

top_in_dir = args.dataset_dir
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
ae_loss = args.ae_loss
class_name = args.class_name
restore_epoch = args.epochs

print("Build model")
train_params = default_train_params()
train_params['training_epochs'] = args.epochs
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = args.train_dir
assert train_dir is not None
print("Train dir:%s"%train_dir)

print("Load model configuration:")
conf_path = os.path.join(train_dir, 'configuration')
conf = Conf.load(conf_path)
print(conf)
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(conf.train_dir, epoch=restore_epoch)

print("Build tensorflow graph")
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

# Load Validation set
print("Load validation set")
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'val')
print(syn_id)
print(class_dir)
all_pc_data = load_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending='.npy', max_num_points=2048, verbose=True, normalize=args.normalize_shape)

feed_pc, _, _ = all_pc_data.full_epoch_data()
feed_pc_tr_all = feed_pc[:, :n_pc_points]
feed_pc_te_all = feed_pc[:, -n_pc_points:]
print(feed_pc_tr_all.shape)
print(feed_pc_te_all.shape)

print("Gather samples")
all_sample = []
all_ref = []
for i in range(feed_pc_tr_all.shape[0]):
    feed_pc_tr = feed_pc_tr_all[i:i+1]
    feed_pc_te = feed_pc_te_all[i:i+1]
    reconstructions = ae.reconstruct(feed_pc_tr)[0]
    all_sample.append(reconstructions)
    all_ref.append(feed_pc_te)
all_sample = np.concatenate(all_sample)
all_ref = np.concatenate(all_ref)
print(all_sample.shape, all_ref.shape)

print("Dump the output so that we can use other codes to evaluate it :(")
np.save(args.ref_outfname, all_sample)
np.save(args.smp_outfname, all_ref)
print(args)

# print("Evaluate it on L3DP's metrics")
# from latent_3d_points.src.evaluation_metrics_fast import MMD_COV_EMD_CD
# mmd_emd, mmd_cd, cov_emd, cov_cd = MMD_COV_EMD_CD(all_sample, all_ref, 100, verbose=True)
# print("MMD-EMD:%s"%mmd_emd)
# print("MMD-CD:%s"%mmd_cd)


