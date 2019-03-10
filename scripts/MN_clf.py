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

from latent_3d_points.src.in_out import snc_category_to_synth_id, create_dir
from latent_3d_points.src.in_out import PointCloudDataSet,load_all_point_clouds_under_folder

from latent_3d_points.src.tf_utils import reset_tf_graph
from latent_3d_points.src.general_utils import plot_3d_point_cloud

# Arguments
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('train_dir', type=str, default=None,
                    help='Training directory (where we stored the model)')
parser.add_argument('--dataset_dir', type=str, default='../data/ModelNet40.PC15k',
                    help='Directory of the dataset.')
parser.add_argument('--normalize_shape', action='store_true',
                    help="Whether normalizing shape.")
parser.add_argument('--epochs', type=int, default=1000,
                    help="Training epochs.")
args = parser.parse_args()
print(args)

train_dir = args.train_dir
top_in_dir = args.dataset_dir
n_pc_points = 2048                # Number of points per model.
bneck_size = 128                  # Bottleneck-AE size
restore_epoch = args.epochs

print("Build model")
print("Train dir:%s"%train_dir)
print("Load model configuration:")
conf_path = os.path.join(train_dir, 'configuration')
conf = Conf.load(conf_path)
print(conf)
reset_tf_graph()
print("Build tensorflow graph")
ae = PointNetAutoEncoder(conf.experiment_name, conf)
ae.restore_model(conf.train_dir, epoch=restore_epoch)

#####################
# Load Training Set
#####################
print("Load data (train set)")
tr_shape_lst = []
te_shape_lst = []
tr_lbl = []
te_lbl = []
class_lst = []
for i, f in enumerate(os.listdir(top_in_dir)):
    # Train
    tr_class_dir = os.path.join(top_in_dir, f, 'train')
    if not os.path.isdir(tr_class_dir):
        continue
    class_lst.append(f)

    all_tr_pc_data = load_all_point_clouds_under_folder(
        tr_class_dir, n_threads=8, file_ending='.npy', max_num_points=n_pc_points,
        verbose=True, normalize=args.normalize_shape, rotation_axis=None
    )
    tr_pc, _, _ = all_tr_pc_data.full_epoch_data()
    N = tr_pc.shape[0]
    tr_shape_lst.append(tr_pc)
    for _ in range(N):
        tr_lbl.append(i)

    # Test
    te_class_dir = os.path.join(top_in_dir, f, 'test')
    all_te_pc_data = load_all_point_clouds_under_folder(
        te_class_dir, n_threads=8, file_ending='.npy', max_num_points=n_pc_points,
        verbose=True, normalize=args.normalize_shape, rotation_axis=None
    )
    te_pc, _, _ = all_te_pc_data.full_epoch_data()
    M = te_pc.shape[0]
    te_shape_lst.append(te_pc)
    for _ in range(M):
        te_lbl.append(i)

tr_pc = np.concatenate(tr_shape_lst)
tr_lbl = np.array(tr_lbl)
te_pc = np.concatenate(te_shape_lst)
te_lbl = np.array(te_lbl)

assert tr_pc.shape[0] == tr_lbl.shape[0]
assert te_pc.shape[0] == te_lbl.shape[0]

print("Gather latent vectors (train set)")
tr_latent = ae.get_latent_codes(tr_pc, batch_size=100)
print(tr_latent.shape)

tr_latent_save_path = os.path.join(conf.train_dir, 'MN_train_all_latent.npy')
tr_label_save_path = os.path.join(conf.train_dir, 'MN_train_all_label.npy')
np.save(tr_latent_save_path, tr_latent)
np.save(tr_label_save_path, tr_lbl)
print("Train latent vectors and labels save path:%s %s"\
      %(tr_latent_save_path, tr_label_save_path))

print("Gather latent vectors (test set)")
te_latent = ae.get_latent_codes(te_pc, batch_size=100)
print(te_latent.shape)

te_latent_save_path = os.path.join(conf.train_dir, 'MN_test_all_latent.npy')
te_label_save_path = os.path.join(conf.train_dir, 'MN_test_all_label.npy')
np.save(te_latent_save_path, te_latent)
np.save(te_label_save_path, te_lbl)
print("Test latent vectors and labels save path:%s %s"\
      %(te_latent_save_path, te_label_save_path))

# Classification
print("Classification...")
from sklearn.svm import LinearSVC
clf = LinearSVC(random_state=0)
clf.fit(tr_latent, tr_lbl)
test_pred = clf.predict(te_latent)
test_gt   = te_lbl.flatten()
acc = np.mean((test_pred==test_gt).astype(float)) * 100.
print("Acc:%s"%acc)

