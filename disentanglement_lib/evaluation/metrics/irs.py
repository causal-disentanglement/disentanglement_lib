# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Interventional Robustness Score.

Based on the paper https://arxiv.org/abs/1811.00007.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import logging
from disentanglement_lib.evaluation.metrics import utils
import numpy as np
import gin.tf
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow_hub as hub
import copy
import matplotlib.pyplot as plt
import disentanglement_lib.evaluation.metrics.dci as dci
from tensorflow.python.keras import losses
import time
import datetime
import os
from sklearn.manifold import TSNE

rho = 7
user = os.environ.get("USER")
path_to_save_images = '/home/{}/disentanglement_lib_cg/examples/s2_factor_vae_cg/'.format(user)
reconstructions_path = '/home/{}/disentanglement_lib_cg/examples/causal_dataset_s2_factor_vae_cg_output/vae/model/tfhub'.format(user)
if not os.path.exists(path_to_save_images):
    os.mkdir(path_to_save_images)
    
def plot_latent_space(z_mu, z_label, i,dim=2):
    tsne = TSNE(n_components=dim, random_state=0, perplexity=50, learning_rate=500, n_iter=300)
    z_tsne = tsne.fit_transform(z_mu)
    dic = {}
    dic['dim1']= z_tsne[:, 0]
    dic['dim2']= z_tsne[:, 1]
    dic['label'] = z_label
    np.save(path_to_save_images+str(i)+'.npy', dic)
    
def get_reconstructions(x):
    module_path = reconstructions_path
    with hub.eval_function_for_module(module_path) as f:
        output = f(dict(latent_vectors=x), signature="decoder", as_dict=True)
        return {key: np.array(values) for key, values in output.items()}

@gin.configurable(
    "irs",
    blacklist=["ground_truth_data", "representation_function", "random_state",
               "artifact_dir"])
def compute_irs(ground_truth_data,
                representation_function,
                random_state,
                artifact_dir=None,
                diff_quantile=0.99,
                num_train=gin.REQUIRED,
                batch_size=gin.REQUIRED):
    """Computes the Interventional Robustness Score.

  Args:
    ground_truth_data: GroundTruthData to be sampled from.
    representation_function: Function that takes observations as input and
      outputs a dim_representation sized representation for each observation.
    random_state: Numpy random state used for randomness.
    artifact_dir: Optional path to directory where artifacts can be saved.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).
    num_train: Number of points used for training.
    batch_size: Batch size for sampling.

  Returns:
    Dict with IRS and number of active dimensions.
  """
    del artifact_dir
    logging.info("Generating training set.")
    dci_d=dci.compute_dci(ground_truth_data, representation_function, random_state,
                artifact_dir=None,
                num_train=gin.REQUIRED,
                num_test=gin.REQUIRED,
                batch_size=16)
    mus, ys = utils.generate_batch_factor_code(ground_truth_data,
                                               representation_function, num_train,
                                               random_state, batch_size)
    assert mus.shape[1] == num_train

    discretizer = utils.make_discretizer(ys)
    ys_discrete = discretizer
    active_mus = _drop_constant_dims(mus)
    if not active_mus.any():
        irs_score = 0.0
    else:
        irs_score = scalable_disentanglement_score(ys_discrete.T, active_mus.T, np.transpose(ys), # active mus
                                                   diff_quantile)["avg_score"]

    score_dict = {}
    score_dict["IRS"] = irs_score
    score_dict["num_active_dims"] = np.sum(active_mus) # active mus
    print(dci_d)
    return score_dict

def get_max_deviations(dims, latents):
    indices_ = [[] for _ in range(latents.shape[0])]
    for idx, dim in enumerate(dims):
        for indx in range(latents.shape[0]):
            max_deviation = np.argmax(np.abs(latents[:, dim] - latents[indx, dim]))
            indices_[indx].append(latents[:, dim][max_deviation])
    return indices_

def normalize(x):
    x[:,:,0] = (x[:,:,0] - np.min(x[:,:,0])) / (np.max(x[:,:,0]) - np.min(x[:,:,0]))
    x[:,:,1] = (x[:,:,1] - np.min(x[:,:,1])) / (np.max(x[:,:,1]) - np.min(x[:,:,1]))
    x[:,:,2] = (x[:,:,2] - np.min(x[:,:,2])) / (np.max(x[:,:,2]) - np.min(x[:,:,2]))
    return x

def _drop_constant_dims(ys):
    """Returns a view of the matrix `ys` with dropped constant rows."""
    ys = np.asarray(ys)
    if ys.ndim != 2:
        raise ValueError("Expecting a matrix.")

    variances = ys.var(axis=1)
    active_mask = variances > 0.
    return ys[active_mask, :]


def scalable_disentanglement_score(gen_factors, latents, factors, diff_quantile=0.99):
    """Computes IRS scores of a dataset.

  Assumes no noise in X and crossed generative factors (i.e. one sample per
  combination of gen_factors). Assumes each g_i is an equally probable
  realization of g_i and all g_i are independent.

  Args:
    gen_factors: Numpy array of shape (num samples, num generative factors),
      matrix of ground truth generative factors.
    latents: Numpy array of shape (num samples, num latent dimensions), matrix
      of latent variables.
    diff_quantile: Float value between 0 and 1 to decide what quantile of diffs
      to select (use 1.0 for the version in the paper).

  Returns:
    Dictionary with IRS scores.
  """
    k = gen_factors.shape[1]
    l = latents.shape[1]
    for i in range(factors.shape[1]):
        plot_latent_space(latents, factors[:,i],i)
    # Compute normalizer EMPIDA.
    max_deviations = np.max(np.abs(latents - latents.mean(axis=0)), axis=0)
    cum_deviations = np.zeros([l, k])
    for i in range(k):
        unique_factors = np.unique(gen_factors[:, i], axis=0)
        assert unique_factors.ndim == 1
        num_distinct_factors = unique_factors.shape[0]
        for k1 in range(num_distinct_factors):
            # Compute E[Z | g_i].
            match = gen_factors[:, i] == unique_factors[k1]
            e_loc = np.mean(latents[match, :], axis=0)
            # Difference of each value within that group of constant g_i to its mean.
            # PIDA
            diffs = np.abs(latents[match, :] - e_loc)
            # MPIDA
            max_diffs = np.percentile(diffs, q=diff_quantile * 100, axis=0)
            cum_deviations[:, i] += max_diffs
        # EMPIDA
        cum_deviations[:, i] /= num_distinct_factors

    # Normalize value of each latent dimension with its maximal deviation.
    normalized_deviations = cum_deviations / max_deviations[:, np.newaxis]
    irs_matrix = 1.0 - normalized_deviations
    disentanglement_scores = irs_matrix.max(axis=1)
    # UC score start
    # todo: get first and second highest indices.(
    indices = irs_matrix.argmax(axis=0)
    sets = [set() for _ in range(k)]
    latent_set = set({i for i in range(l)})
    for i in range(k):
        temp = irs_matrix[:, i]
        indices = np.argpartition(temp, -1)[::-rho][:rho]
        for j in indices:
            sets[i].add(j)
    un_norm_score = 0
    print(sets)
    for i in range(k-1):
        for j in range(i+1, k):
            un_norm_score += len(sets[i].intersection(sets[j]))/float(len(sets[i].union(sets[j])))
    norm_score = un_norm_score/(k*(k-1)/2.)
    uc = 1 - norm_score
    # UC score end
    # CG score start
    score = 0
    model = tf.keras.models.load_model('/home/{}/disentanglement_lib_cg/disentanglement_lib/methods/trained_models/causal_dataset_v2.h5'.format(user))
    start = time.time()
    mean_latents = np.mean(latents, axis=0)
    print_images_count = 0
    for fac in range(gen_factors.shape[1]):
        z_dims_for_g_i = list(sets[fac])
        z_dims_for_g_not_i = list(latent_set.difference(sets[fac]))

        latents_for_ice1 = copy.deepcopy(latents)
        latents_for_ice2 = copy.deepcopy(latents)
        latents_for_ice_baseline = copy.deepcopy(latents)

        latents_for_ice1[:, z_dims_for_g_i] = get_max_deviations(z_dims_for_g_i, latents)
        latents_for_ice2[:, z_dims_for_g_not_i] = get_max_deviations(z_dims_for_g_not_i, latents)

        recon_ice_baseline = get_reconstructions(latents_for_ice_baseline)
        recon_ice1 = get_reconstructions(latents_for_ice1)
        recon_ice2 = get_reconstructions(latents_for_ice2)

        current_factor = factors[:,fac]
        
        idx = None
        if fac == 0:
            idx = current_factor
        elif fac == 1:
            idx = current_factor + 5
        elif fac == 2:
            idx = current_factor + 10
        elif fac == 3:
            idx = current_factor + 13
        elif fac == 4:
            idx = current_factor + 19
        elif fac == 5:
            idx = current_factor + 35
            
        p_baseline = model.predict(recon_ice_baseline['images'].transpose([0, 3, 1, 2]))
        p1 = model.predict(recon_ice1['images'].transpose([0, 3, 1, 2]))
        p2 = model.predict(recon_ice2['images'].transpose([0, 3, 1, 2]))
        # print images
        print_images_count += 1
        if print_images_count <= 6:
            plt.imsave(path_to_save_images+'sample_original_'+str(print_images_count)+"_fac_"+str(fac)+"_rho_"+str(rho)+'.png', normalize(recon_ice_baseline['images'][0]))
            plt.imsave(path_to_save_images+'sample_ice1_'+str(print_images_count)+"_fac_"+str(fac)+"_rho_"+str(rho)+'.png', normalize(recon_ice1['images'][0]))
            plt.imsave(path_to_save_images+'sample_ice2_' + str(print_images_count) + "_fac_" + str(fac)+"_rho_"+str(rho) + '.png', normalize(recon_ice2['images'][0]))

        # print images end
        score += sum(abs(abs(p_baseline[list(range(factors.shape[0])), list(idx)] - p1[list(range(factors.shape[0])), list(idx)])
                         - abs(p_baseline[list(range(factors.shape[0])), list(idx)] - p2[list(range(factors.shape[0])), list(idx)])))/factors.shape[0]
    cg = score/gen_factors.shape[1]
    end = time.time()
    print(end-start)
    # CG score end
    if np.sum(max_deviations) > 0.0:
        avg_score = np.average(disentanglement_scores, weights=max_deviations)
    else:
        avg_score = np.mean(disentanglement_scores)
    print("rho= ", rho)
    print("irs= ", avg_score)
    print("uc = ", uc)
    print("cg = ", cg)
    print("sets = ", sets)
    parents = irs_matrix.argmax(axis=1)
    score_dict = {}
    score_dict["disentanglement_scores"] = disentanglement_scores
    score_dict["avg_score"] = "uc = "+str(uc)+", cg = "+str(cg)
    score_dict["parents"] = parents
    score_dict['IRS_SCORE'] = avg_score
    score_dict["IRS_matrix"] = irs_matrix
    score_dict["max_deviations"] = max_deviations
    return score_dict
