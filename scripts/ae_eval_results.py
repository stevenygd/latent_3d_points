# coding: utf-8

# ## This notebook will help you train a vanilla Point-Cloud AE with the basic architecture we used in our paper.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

import os
import sys
sys.path.insert(0, "/home/gy46/")

import tqdm
import numpy as np
import os.path as osp
from latent_3d_points.src.ae_templates import mlp_architecture_ala_iclr_18, default_train_params
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.point_net_ae import PointNetAutoEncoder

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, snc_synth_id_to_category
from latent_3d_points.src.in_out import PointCloudDataSet,load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

# Arguments
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('class_name', type=str, choices=list(snc_synth_id_to_category.values()) + ['all'],
                    help='Category for which we used to train AE. (right now only chair, car, airplane)')
parser.add_argument('train_dir', type=str, default=None,
                    help='Training directory (where we stored the model)')
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--ref_outfname', type=str, default="ref_pcls.npy")
parser.add_argument('--smp_outfname', type=str, default="smp_pcls.npy")
parser.add_argument('--normalize_shape', action='store_true',
                    help="Whether normalizing shape.")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Restore epochs.")
args = parser.parse_args()
print(args)

top_in_dir = args.dataset_dir
n_pc_points = 2048                # Number of points per model.
class_name = args.class_name
assert class_name == 'all', "Not supporting other things yet."
restore_epoch = args.epochs

print("Build model")
train_dir = args.train_dir
print("Train dir:%s"%train_dir)
print("Load model configuration:")
conf_path = os.path.join(train_dir, 'configuration')
conf = Conf.load(conf_path)
print(conf)
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(args.train_dir, epoch=restore_epoch)

# Load Validation set
print("Load data (train set)")
sub_dirs = []
for subdir in os.listdir(os.path.join(top_in_dir)):
    p = os.path.join(top_in_dir, subdir, 'val')
    print(p)
    sub_dirs.append(p)

all_pc_data = load_all_point_clouds_under_folders(
    sub_dirs, n_threads=8, file_ending='.npy', max_num_points=2048, verbose=True,
    normalize=args.normalize_shape, rotation_axis=(1 if args.random_rotation else None))

# Load all the data
feed_pc, _, _ = all_pc_data.full_epoch_data()
feed_pc_tr_all = feed_pc[:, :n_pc_points]
feed_pc_te_all = feed_pc[:, -n_pc_points:]
print(feed_pc_tr_all.shape)
print(feed_pc_te_all.shape)

print("Gather samples")
all_sample = []
all_ref = []
for i in tqdm.trange(feed_pc_tr_all.shape[0]):
    feed_pc_tr = feed_pc_tr_all[i:i+1]
    feed_pc_te = feed_pc_te_all[i:i+1]
    reconstructions = ae.reconstruct(feed_pc_tr)[0]
    all_sample.append(reconstructions)
    all_ref.append(feed_pc_te)
all_sample = np.concatenate(all_sample)
all_ref = np.concatenate(all_ref)
print(all_sample.shape, all_ref.shape)

print("Dump the output so that we can use other codes to evaluate it :(")
np.save(os.path.join(os.path.basename(args.train_dir), args.ref_outfname), all_sample)
np.save(os.path.join(os.path.basename(args.train_dir), args.smp_outfname), all_ref)
print(args)


