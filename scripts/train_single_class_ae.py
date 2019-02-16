# coding: utf-8

# ## This notebook will help you train a vanilla Point-Cloud AE with the basic architecture we used in our paper.
#     (it assumes latent_3d_points is in the PYTHONPATH and the structural losses have been compiled)

import sys
sys.path.insert(0, "/home/gy46/")

import numpy as np
import os
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
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--output_dir', type=str, default="../expr/", help='Output path.')
parser.add_argument('--expr_prefix', type=str, default='shapenetcorev2',
                    help='Prefix for the experiment.')
parser.add_argument('--ae_loss', type=str, default='chamfer', choices=['chamfer', 'emd'],
                    help='Loss to optimize for ([emd] or [chamfer]).')
parser.add_argument('--load_pre_trained_ae', action='store_true',
                    help="Load pretrained AE or not.")
parser.add_argument('--normalize_shape', action='store_true',
                    help="Whether normalizing shape.")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Training epochs.")

args = parser.parse_args()
print(args)

# top_out_dir = '../expr/'                      # Use to save Neural-Net check-points etc.
# top_in_dir = '../data/ShapeNetCore.v2.PC15k/' # Top-dir of where point-clouds are stored.
top_out_dir = args.output_dir
top_in_dir = args.dataset_dir

# experiment_name = 'single_class_ae_emd'
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size

ae_loss = args.ae_loss
class_name = args.class_name
experiment_name = '%s_%s_ae_%s'%(args.expr_prefix, ae_loss, class_name)
print("Experiment name:%s")
print(experiment_name)


syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'train')
print(syn_id)
print(class_dir)


# Load Data
print("Load data (train set)")
all_pc_data = load_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending='.npy', max_num_points=2048, verbose=True, normalize=args.normalize_shape)


# Load default training parameters (some of which are listed beloq). For more details please print the configuration object.
#
#     'batch_size': 50
#
#     'denoising': False     (# by default AE is not denoising)
#
#     'learning_rate': 0.0005
#
#     'z_rotate': False      (# randomly rotate models of each batch)
#
#     'loss_display_step': 1 (# display loss at end of these many epochs)
#     'saver_step': 10       (# over how many epochs to save neural-network)

print("Build model")
train_params = default_train_params()
train_params['training_epochs'] = args.epochs
encoder, decoder, enc_args, dec_args = mlp_architecture_ala_iclr_18(n_pc_points, bneck_size)
train_dir = create_dir(osp.join(top_out_dir, experiment_name))

conf = Conf(n_input = [n_pc_points, 3],
            loss = ae_loss,
            training_epochs = train_params['training_epochs'],
            batch_size = train_params['batch_size'],
            denoising = train_params['denoising'],
            learning_rate = train_params['learning_rate'],
            train_dir = train_dir,
            loss_display_step = train_params['loss_display_step'],
            saver_step = train_params['saver_step'],
            z_rotate = train_params['z_rotate'],
            encoder = encoder,
            decoder = decoder,
            encoder_args = enc_args,
            decoder_args = dec_args
           )
conf.experiment_name = experiment_name
conf.held_out_step = 5   # How often to evaluate/print out loss on
                         # held_out data (if they are provided in ae.train() ).
conf.save(osp.join(train_dir, 'configuration'))

print("Model configuration:")
print(conf)


load_pre_trained_ae = args.load_pre_trained_ae
restore_epoch = args.epochs
if load_pre_trained_ae:
    conf = Conf.load(train_dir + '/configuration')
    reset_tf_graph()
    ae = PointNetAutoEncoder(conf.experiment_name, conf)
    ae.restore_model(conf.train_dir, epoch=restore_epoch)

print("Build tensorflow graph")
reset_tf_graph()
ae = PointNetAutoEncoder(conf.experiment_name, conf)

print("Start training...")
buf_size = 1 # Make 'training_stats' file to flush each output line regarding training.
fout = open(osp.join(conf.train_dir, 'train_stats.txt'), 'a', buf_size)
train_stats = ae.train(all_pc_data, conf, log_file=fout)
fout.close()

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

sample_save_path = os.path.join(conf.train_dir, 'all_sample.npy')
np.save(sample_save_path, all_sample)
print("Samples save path:%s"%sample_save_path)

reference_save_path = os.path.join(conf.train_dir, 'all_reference.npy')
np.save(reference_save_path, all_ref)
print("Reference save path:%s"%reference_save_path)

from latent_3d_points.src.evaluation_metrics_fast import MMD_COV_EMD_CD
mmd_emd, mmd_cd, cov_emd, cov_cd = MMD_COV_EMD_CD(all_sample, all_ref, 100, verbose=True)
print("Validation results for :%s"%experiment_name)
print("MMD-EMD:%s"%mmd_emd)
print("MMD-CD:%s"%mmd_cd)


