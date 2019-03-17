# coding: utf-8
import os
import sys
sys.path.insert(0, '/home/guandao/hdd/Projects/')
import numpy as np
import os.path as osp
import matplotlib.pylab as plt
from latent_3d_points.src.autoencoder import Configuration as Conf
from latent_3d_points.src.neural_net import MODEL_SAVER_ID
from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir, PointCloudDataSet, load_all_point_clouds_under_folder
from latent_3d_points.src.general_utils import plot_3d_point_cloud
from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.vanilla_gan import Vanilla_GAN
from latent_3d_points.src.w_gan_gp import W_GAN_GP
from latent_3d_points.src.generators_discriminators import point_cloud_generator,mlp_discriminator, leaky_relu

#############
# Arguments #
#############
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('class_name', type=str, choices=['chair', 'car', 'airplane'],
                    help='Category for which we used to train AE. (right now only chair, car, airplane)')
parser.add_argument('--dataset_dir', type=str, default="../data/ShapeNetCore.v2.PC15k/", help='Dataset path.')
parser.add_argument('--output_dir', type=str, default="../expr/", help='Output path.')
parser.add_argument('--expr_prefix', type=str, default='shapenetcorev2', help='Prefix for the experiment.')
parser.add_argument('--normalize_shape', action='store_true', help="Whether normalizing shape.")
parser.add_argument('--epochs', type=int, default=1000, help="Training epochs.")
parser.add_argument('--split_file', type=str, default=None, help="File that contains the split.")
args = parser.parse_args()
print(args)


###########
# Configs #
###########
# Use to save Neural-Net check-points etc.
top_out_dir = args.output_dir

# Top-dir of where point-clouds are stored.
top_in_dir = args.dataset_dir

n_pc_points = 2048                # Number of points per model.
class_name = args.class_name
experiment_name = 'rawgan_%s_%s'%(args.expr_prefix, class_name)


######################
# Load point-clouds. #
######################
print("Load data (train set)")
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'train')
print(syn_id)
print(class_dir)

print("Load data (train set)")
if args.split_file is not None:
    file_names = np.load(args.split_file).item()[syn_id]['train']
    file_names = [ os.path.join(args.dataset_dir, syn_id, f+".npy") for f in file_names ]
else:
    file_names = None
all_pc_data = load_all_point_clouds_under_folder(
    class_dir, n_threads=8, file_ending='.npy', max_num_points=2048, verbose=True,
    normalize=args.normalize_shape, file_names=file_names)
print 'Shape of DATA =', all_pc_data.point_clouds.shape


#######################
# Set GAN parameters. #
#######################
use_wgan = True     # Wasserstein with gradient penalty, or not?
n_epochs = args.epochs # Epochs to train.

plot_train_curve = True
save_gan_model = True
saver_step = np.hstack([np.array([1, 5, 10]), np.arange(50, n_epochs + 1, 100)])

# If true, every 'saver_step' epochs we produce & save synthetic pointclouds.
save_synthetic_samples = True
# How many synthetic samples to produce at each save step.
n_syn_samples = all_pc_data.num_examples

# Optimization parameters
init_lr = 0.0001
batch_size = 50
noise_params = {'mu':0, 'sigma': 0.2}
noise_dim = 128
beta = 0.5 # ADAM's momentum.

n_out = [n_pc_points, 3] # Dimensionality of generated samples.

discriminator = mlp_discriminator
generator = point_cloud_generator

synthetic_data_out_dir = osp.join(top_out_dir, experiment_name, 'synthetic')
create_dir(synthetic_data_out_dir)

train_dir = osp.join(top_out_dir, experiment_name)
create_dir(train_dir)



#####################
# Create GAN Graph. #
#####################
reset_tf_graph()

if use_wgan:
    lam = 10
    disc_kwargs = {'b_norm': False}
    gan = W_GAN_GP(experiment_name, init_lr, lam, n_out, noise_dim,
                    discriminator, generator,
                    disc_kwargs=disc_kwargs, beta=beta)
else:
    leak = 0.2
    disc_kwargs = {'non_linearity': leaky_relu(leak), 'b_norm': False}
    gan = Vanilla_GAN(experiment_name, init_lr, n_out, noise_dim,
                      discriminator, generator, beta=beta, disc_kwargs=disc_kwargs)
accum_syn_data = []
train_stats = []


##################
# Train the GAN. #
##################
import tqdm
for _ in tqdm.trange(n_epochs):
    loss, duration = gan._single_epoch_train(all_pc_data, batch_size, noise_params)
    epoch = int(gan.sess.run(gan.increment_epoch))

    if save_gan_model and epoch in saver_step:
        checkpoint_path = osp.join(train_dir, MODEL_SAVER_ID)
        gan.saver.save(gan.sess, checkpoint_path, global_step=gan.epoch)

    if save_synthetic_samples and epoch in saver_step:
        syn_data = gan.generate(n_syn_samples, noise_params)
        np.savez(osp.join(synthetic_data_out_dir, 'epoch_' + str(epoch)), syn_data)

    train_stats.append((epoch, ) + loss)

# if plot_train_curve:
#     x = range(len(train_stats))
#     d_loss = [t[1] for t in train_stats]
#     g_loss = [t[2] for t in train_stats]
#     plt.plot(x, d_loss, '--')
#     plt.plot(x, g_loss)
#     plt.title('GAN training. (%s)' %(class_name))
#     plt.legend(['Discriminator', 'Generator'], loc=0)
#
#     plt.tick_params(axis='x', which='both', bottom='off', top='off')
#     plt.tick_params(axis='y', which='both', left='off', right='off')
#
#     plt.xlabel('Epochs.')
#     plt.ylabel('Loss.')


##############
# Evaluation #
##############
print("Load data (val set for evaluation)")
syn_id = snc_category_to_synth_id()[class_name]
class_dir = osp.join(top_in_dir , syn_id, 'train')
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
all_sample = gan.generate(feed_pc_tr.shape[0], noise_params)
sample_save_path = os.path.join(train_dir, 'all_sample.npy')
np.save(sample_save_path, all_sample)
print("Samples save path:%s"%sample_save_path)
print("Generated points shapes: %s %s"%(syn_latent_data.shape, all_sample.shape))

