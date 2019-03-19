# coding: utf-8
import os
import sys
sys.path.insert(0, '/home/gy46/')

import numpy as np
import os.path as osp
import matplotlib.pylab as plt

from latent_3d_points.src.point_net_ae import PointNetAutoEncoder
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.neural_net import MODEL_SAVER_ID

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet,                                         load_all_point_clouds_under_folder

from latent_3d_points.src.general_utils import plot_3d_point_cloud
from latent_3d_points.src.tf_utils import reset_tf_graph

from latent_3d_points.src.vanilla_gan import Vanilla_GAN
from latent_3d_points.src.w_gan_gp import W_GAN_GP
from latent_3d_points.src.generators_discriminators import latent_code_discriminator_two_layers,latent_code_generator_two_layers

#############
# Arguments #
#############
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('class_name', type=str, choices=['chair', 'car', 'airplane'],
                    help='Category for which we used to train AE. (right now only chair, car, airplane)')
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--output_dir', type=str, default="../expr/", help='Output path.')
parser.add_argument('--ae_configuration', type=str, default=None, help='Directory where the AE configuration is stored.')
parser.add_argument('--gan_model_path', type=str, default=None, help='Directory where the GAN configuration is stored.')
parser.add_argument('--expr_prefix', type=str, default='shapenetcorev2', help='Prefix for the experiment.')
parser.add_argument('--normalize_shape', action='store_true', help="Whether normalizing shape.")
parser.add_argument('--ae_epochs', type=int, default=1000, help="AE epochs to resume.")
parser.add_argument('--epochs', type=int, default=1000, help="Training epochs.")
parser.add_argument('--split_file', type=str, default=None, help="File that contains the split.")
args = parser.parse_args()
print(args)


###########
# Configs #
###########
# Top-dir of where point-clouds are stored.
top_in_dir = args.dataset_dir

# Where to save GANs check-points etc.
top_out_dir = args.output_dir

ae_epoch = args.ae_epochs # Epoch of AE to load.
bneck_size = 128         # Bottleneck-size of the AE
n_pc_points = 2048       # Number of points per model.

class_name = args.class_name
experiment_name = 'latentgan_%s_%s'%(args.expr_prefix, class_name)
ae_configuration = args.ae_configuration


#######################
# Load pre-trained AE #
#######################
reset_tf_graph()
ae_conf = Conf.load(ae_configuration)
ae_conf.encoder_args['verbose'] = False
ae_conf.decoder_args['verbose'] = False
ae = PointNetAutoEncoder(ae_conf.experiment_name, ae_conf)
ae.restore_model(ae_conf.train_dir, ae_epoch, verbose=True)

# Use AE to convert raw pointclouds to latent codes.
latent_codes = ae.get_latent_codes(all_pc_data.point_clouds)
latent_data = PointCloudDataSet(latent_codes)
print 'Shape of DATA =', latent_data.point_clouds.shape


#######################
# Set GAN parameters. #
#######################
use_wgan = True     # Wasserstein with gradient penalty, or not?
n_epochs = args.epochs # Epochs to train.

plot_train_curve = True
save_gan_model = True
saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 250)])

# If true, every 'saver_step' epochs we produce & save synthetic pointclouds.
save_synthetic_samples = True
# How many synthetic samples to produce at each save step.
n_syn_samples = latent_data.num_examples

# Optimization parameters
init_lr = 0.0001
batch_size = 50
noise_params = {'mu':0, 'sigma': 0.2}
noise_dim = bneck_size
beta = 0.5 # ADAM's momentum.

n_out = [bneck_size] # Dimensionality of generated samples.

synthetic_data_out_dir = osp.join(top_out_dir, experiment_name, 'synthetic')
create_dir(synthetic_data_out_dir)

train_dir = osp.join(top_out_dir, experiment_name)
create_dir(train_dir)


################
# Create GAN . #
################
reset_tf_graph()

if use_wgan:
    lam = 10 # lambda of W-GAN-GP
    gan = W_GAN_GP(
            experiment_name, init_lr, lam, n_out, noise_dim,
            latent_code_discriminator_two_layers,
            latent_code_generator_two_layers,\
            beta=beta)
else:
    gan = Vanilla_GAN(
            experiment_name, init_lr, n_out, noise_dim,
            latent_code_discriminator_two_layers, latent_code_generator_two_layers,
            beta=beta)


accum_syn_data = []
train_stats = []


##################
# Train the GAN. #
##################
for _ in range(n_epochs):
    loss, duration = gan._single_epoch_train(latent_data, batch_size, noise_params)
    epoch = int(gan.sess.run(gan.increment_epoch))
    print epoch, loss

    if save_gan_model and epoch in saver_step:
        checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
        gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

    if save_synthetic_samples and epoch in saver_step:
        syn_latent_data = gan.generate(n_syn_samples, noise_params)
        syn_data = ae.decode(syn_latent_data)
        np.savez(osp.join(synthetic_data_out_dir, 'epoch_' + str(epoch)), syn_data)
        # for k in range(3):  # plot three (syntetic) random examples.
        #     plot_3d_point_cloud(syn_data[k][:, 0], syn_data[k][:, 1], syn_data[k][:, 2],
        #                        in_u_sphere=True)

    train_stats.append((epoch, ) + loss)


# # if plot_train_curve:
# x = range(len(train_stats))
# d_loss = [t[1] for t in train_stats]
# g_loss = [t[2] for t in train_stats]
# plt.plot(x, d_loss, '--')
# plt.plot(x, g_loss)
# plt.title('Latent GAN training. (%s)' %(class_name))
# plt.legend(['Discriminator', 'Generator'], loc=0)
#
# plt.tick_params(axis='x', which='both', bottom='off', top='off')
# plt.tick_params(axis='y', which='both', left='off', right='off')
#
# plt.xlabel('Epochs.')
# plt.ylabel('Loss.')
# plt.show()

##############
# Evaluation #
##############

print("Load data (val set for evaluation)")
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'val')
if args.split_file is not None:
    file_names = np.load(args.split_file).item()[syn_id]['val']
    file_names = [ os.path.join(args.dataset_dir, syn_id, f+".npy") for f in file_names ]
else:
    file_names = None
all_pc_data = load_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending='.npy', max_num_points=2048, verbose=True,
    normalize=args.normalize_shape, file_names=file_names)
print 'Shape of DATA =', all_pc_data.point_clouds.shape

print("Split into train and test points")
feed_pc, _, _ = all_pc_data.full_epoch_data()
feed_pc_tr = feed_pc[:, :n_pc_points]
all_ref = feed_pc[:, -n_pc_points:]
print("Validation point clouds: tr %s test %s"%(feed_pc_tr.shape, all_ref.shape))
reference_save_path = os.path.join(train_dir, 'all_reference.npy')
np.save(reference_save_path, all_ref)
print("Reference save path:%s"%reference_save_path)

print("Generate synthesized points")
syn_latent_data = gan.generate(feed_pc_tr.shape[0], noise_params)
all_sample = ae.decode(syn_latent_data)
sample_save_path = os.path.join(train_dir, 'all_sample.npy')
np.save(sample_save_path, all_sample)
print("Samples save path:%s"%sample_save_path)
print("Generated points shapes: %s %s"%(syn_latent_data.shape, all_sample.shape))


print("Compute metrics")
from latent_3d_points.src.evaluation_metrics_fast import MMD_COV_EMD_CD
mmd_emd, mmd_cd, cov_emd, cov_cd = MMD_COV_EMD_CD(all_sample, all_ref, 100, verbose=True)
print("Validation results for :%s"%experiment_name)
print("MMD-EMD:%s"%mmd_emd)
print("MMD-CD:%s"%mmd_cd)
print("COV-EMD:%s"%cov_emd)
print("COV-CD:%s"%cov_cd)


from latent_3d_points.src.evaluation_metrics import jsd_between_point_cloud_sets as JSD
jsd = JSD(syn_data, all_ref)
print("JSD:%s"%jsd)

