'''
Created on October 11, 2017

@author: optas

@article{achlioptas2017latent_pc,
  title={Learning Representations And Generative Models For 3D Point Clouds},
  author={Achlioptas, Panos and Diamanti, Olga and Mitliagkas, Ioannis and Guibas, Leonidas J},
  journal={arXiv preprint arXiv:1707.02392},
  year={2017}
}

Codes taken from https://github.com/optas/latent_3d_points/blob/master/src/evaluation_metrics.py
'''

import tensorflow as tf
import numpy as np
import warnings
import tqdm

from scipy.stats import entropy
from. general_utils import iterate_in_chunks, unit_cube_grid_point_cloud
try:
    from sklearn.neighbors import NearestNeighbors
except:
    print ('Sklearn module not installed (JSD metric will not work).')

try:
    from .. external.structural_losses.tf_nndistance import nn_distance
    from .. external.structural_losses.tf_approxmatch import approx_match, match_cost
except:
    print('External Losses (Chamfer-EMD) cannot be loaded. Please install them first.')



def minimum_mathing_distance_tf_graph(n_pc_points, batch_size=None, normalize=True, sess=None,
                                      verbose=False, use_sqrt=False):
    ''' Produces the graph operations necessary to compute the MMD and consequently also the Coverage due to their 'symmetric' nature.
    Assuming a "reference" and a "sample" set of point-clouds that will be matched, this function creates the operation that matches
    a _single_ "reference" point-cloud to all the "sample" point-clouds given in a batch. Thus, is the building block of the function
    ```minimum_mathing_distance`` and ```coverage``` that iterate over the "sample" batches and each "reference" point-cloud.

    Args:
        n_pc_points (int): how many points each point-cloud of those to be compared has.
        batch_size (optional, int): if the iterator code that uses this function will
            use a constant batch size for iterating the sample point-clouds you can
            specify it hear to speed up the compute. Alternatively, the code is adapted
            to read the batch size dynamically.
        normalize (boolean): if True, the matched distances are normalized by diving them with
            the number of points of the compared point-clouds (n_pc_points).
        use_sqrt (boolean): When the matching is based on Chamfer (default behavior), if True,
            the Chamfer is computed based on the (not-squared) euclidean distances of the
            matched point-wise euclidean distances.
        use_EMD (boolean): If true, the matchings are based on the EMD.
    '''
    if normalize:
        reducer = tf.reduce_mean
    else:
        reducer = tf.reduce_sum

    if sess is None:
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    # Placeholders for the point-clouds: 1 for the reference (usually Ground-truth) and one of variable size for the collection
    # which is going to be matched with the reference.
    ref_pl = tf.placeholder(tf.float32, shape=(1, n_pc_points, 3))
    sample_pl = tf.placeholder(tf.float32, shape=(batch_size, n_pc_points, 3))

    if batch_size is None:
        batch_size = tf.shape(sample_pl)[0]

    ref_repeat = tf.tile(ref_pl, [batch_size, 1, 1])
    ref_repeat = tf.reshape(ref_repeat, [batch_size, n_pc_points, 3])

    # EMD:
    match = approx_match(ref_repeat, sample_pl)
    all_dist_in_batch_EMD = match_cost(ref_repeat, sample_pl, match)
    if normalize:
        all_dist_in_batch_EMD /= n_pc_points

    # Best distance, of those that were matched to single ref pc.
    best_in_batch_EMD = tf.reduce_min(all_dist_in_batch_EMD)
    location_of_best_EMD = tf.argmin(all_dist_in_batch_EMD, axis=0)

    # CD
    ref_to_s, _, s_to_ref, _ = nn_distance(ref_repeat, sample_pl)
    if use_sqrt:
        ref_to_s = tf.sqrt(ref_to_s)
        s_to_ref = tf.sqrt(s_to_ref)
    all_dist_in_batch_CD = reducer(ref_to_s, 1) + reducer(s_to_ref, 1)

    # Best distance, of those that were matched to single ref pc.
    best_in_batch_CD = tf.reduce_min(all_dist_in_batch_CD)
    location_of_best_CD = tf.argmin(all_dist_in_batch_CD, axis=0)

    return ref_pl, sample_pl, best_in_batch_EMD, location_of_best_EMD, \
            best_in_batch_CD, location_of_best_CD, sess


def MMD_COV_EMD_CD(sample_pcs, ref_pcs, batch_size,
        normalize=True, sess=None, verbose=False, use_sqrt=False, ret_dist=False):
    '''Computes the MMD between two sets of point-clouds.

    Args:
        sample_pcs (numpy array SxKx3): the S point-clouds, each of K points that will be matched and
            compared to a set of "reference" point-clouds.
        ref_pcs (numpy array RxKx3): the R point-clouds, each of K points that constitute the set of
            "reference" point-clouds.
        batch_size (int): specifies how large will the batches be that the compute will use to make
            the comparisons of the sample-vs-ref point-clouds.
        normalize (boolean): if True, the distances are normalized by diving them with
            the number of the points of the point-clouds (n_pc_points).
        use_sqrt: (boolean): When the matching is based on Chamfer (default behavior), if True, the
            Chamfer is computed based on the (not-squared) euclidean distances of the matched point-wise
             euclidean distances.
        ret_dist (boolean): If true, it will also return the distances between each sample_pcs and
            it's matched ground-truth.
        sess (tf.Session, default None): if None, it will make a new Session for this.
    Returns:
        MMD-EMD, COV-EMD, MMD-CD, COV-CD
    '''

    n_ref, n_pc_points, pc_dim = ref_pcs.shape
    _, n_pc_points_s, pc_dim_s = sample_pcs.shape

    if n_pc_points != n_pc_points_s or pc_dim != pc_dim_s:
        raise ValueError('Incompatible size of point-clouds.')

    ref_pl, sample_pl, best_in_batch_EMD, loc_best_EMD, best_in_batch_CD, loc_best_CD, sess = \
            minimum_mathing_distance_tf_graph(n_pc_points, normalize=normalize,
                sess=sess, use_sqrt=use_sqrt)

    def _helper_(best_in_all_batches, loc_in_all_batches):
        best_in_all_batches = np.array(best_in_all_batches)
        b_hit = np.argmin(best_in_all_batches)    # In which batch the minimum occurred.
        matched_dist = np.min(best_in_all_batches)
        hit = np.array(loc_in_all_batches)[b_hit]
        matched_gt = batch_size * b_hit + hit
        return matched_dist, matched_gt

    matched_gt_EMD = []
    matched_gt_CD  = []
    matched_dists_EMD = []
    matched_dists_CD  = []
    iterator = tqdm.trange(n_ref, desc=("MMD-COV Loop")) if verbose else range(n_ref)
    for i in iterator:
        best_in_all_batches_EMD = []
        loc_in_all_batches_EMD = []
        best_in_all_batches_CD  = []
        loc_in_all_batches_CD = []
        for sample_chunk in iterate_in_chunks(sample_pcs, batch_size):
            feed_dict = {ref_pl: np.expand_dims(ref_pcs[i], 0), sample_pl: sample_chunk}
            b_EMD, b_CD, l_EMD, l_CD = sess.run([
                best_in_batch_EMD, best_in_batch_CD, loc_best_EMD, loc_best_CD],
                feed_dict=feed_dict
            )
            best_in_all_batches_EMD.append(b_EMD)
            best_in_all_batches_CD.append(b_CD)
            loc_in_all_batches_EMD.append(l_EMD)
            loc_in_all_batches_CD.append(l_CD)

        dist_emd, gt_emd = _helper_(best_in_all_batches_EMD, loc_in_all_batches_EMD)
        dist_cd, gt_cd   = _helper_(best_in_all_batches_CD, loc_in_all_batches_CD)

        matched_gt_EMD.append(gt_emd)
        matched_gt_CD.append(gt_cd)
        matched_dists_EMD.append(dist_emd)
        matched_dists_CD.append(dist_cd)

    mmd_emd = np.mean(matched_dists_EMD)
    mmd_cd  = np.mean(matched_dists_CD)

    cov_emd = len(np.unique(matched_gt_EMD)) / float(n_ref)
    cov_cd  = len(np.unique(matched_gt_CD)) / float(n_ref)

    if ret_dist:
        raise Exception("Not implemented yet.")

    return mmd_emd, mmd_cd, cov_emd, cov_cd

def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    '''Computes the JSD between two sets of point-clouds, as introduced in the paper ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    '''
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)

def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=0):
    '''Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    '''
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if (abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound) and verbose > 0:
        warnings.warn('Point-clouds are not in unit cube.')

    if (in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound) and verbose > 0:
        warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        p = 0.0
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)      # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    '''another way of computing JSD'''
    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


########################################################################
# Evaluation Metrics from PC-GAN (Distance-to-Face based)
########################################################################

# import pymesh
# def MMD_COV_D2F(pc_lst, mesh_lst, one_to_one=True):
#     num_broken_mesh = len([x for x in mesh_lst if x is None])
#     print("Broken mesh : %d"%num_broken_mesh)
#
#     N, M = len(pc_lst), len(mesh_lst)
#
#     mmd_out = []
#     cov_out = []
#
#     for i in tqdm.trange(N, desc="D2F"):
#         pc = pc_lst[i].reshape(-1, 3)
#         idx_lst = [i] if one_to_one else range(M)
#
#         best_dist = None
#         best_cov  = None
#         for j in idx_lst:
#             mesh = mesh_lst[j]
#             if mesh is None:
#                 continue
#             squared_distances, face_indices, _ = pymesh.distance_to_mesh(mesh, pc)
#
#             dist = np.array(squared_distances).mean()
#             best_dist = dist if best_dist is None else min(best_dist, dist)
#
#             cov  = float(len(np.unique(face_indices))) / float(mesh.num_faces)
#             best_cov = cov if best_cov is None else max(best_cov, cov)
#
#         mmd_out.append(best_dist)
#         cov_out.append(best_cov)
#
#     mmd_out = np.array(mmd_out).mean()
#     cov_out = np.array(cov_out).mean()
#     return mmd_out, cov_out



