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

"""Example script how to get started with research using disentanglement_lib.

To run the example, please change the working directory to the containing folder
and run:
>> python example.py

In this example, we show how to use disentanglement_lib to:
1. Train a standard VAE (already implemented in disentanglement_lib).
3. Extract the mean representations for both of these models.
4. Compute the Mutual Information Gap (already implemented) for both models.
6. Aggregate the results.
7. Print out the final Pandas data frame with the results.
"""

# We group all the imports at the top.
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.evaluation import evaluate
from disentanglement_lib.evaluation.metrics import utils
#from disentanglement_lib.methods.unsupervised import train # vae, beta_vae, beta_tc_vae, factor_vae
#from disentanglement_lib.methods.unsupervised import vae
from disentanglement_lib.postprocessing import postprocess
from disentanglement_lib.utils import aggregate_results
from disentanglement_lib.methods.semi_supervised import train_semi_supervised_lib # s2_vae, s2_beta_vae, s2_beta_tc_vae, s2_factor_vae
import tensorflow.compat.v1 as tf
import gin.tf

# 0. Settings
# ------------------------------------------------------------------------------
# By default, we save all the results in subdirectories of the following path.
base_path = "causal_dataset_s2_factor_vae_cg_output"

# By default, we do not overwrite output directories. Set this to True, if you
# want to overwrite (in particular, if you rerun this script several times).
overwrite = True

# 1. Train a standard VAE (already implemented in disentanglement_lib).
# ------------------------------------------------------------------------------

# We save the results in a `vae` subfolder.
path_vae = os.path.join(base_path, "vae")

# The main training protocol of disentanglement_lib is defined in the
# disentanglement_lib.methods.unsupervised.train module. To configure
# training we need to provide a gin config. For a standard VAE, you may have a
# look at model.gin on how to do this.
# train_semi_supervised_lib.train_with_gin(os.path.join(path_vae, "model"), overwrite, ["model.gin"])
# After this command, you should have a `vae` subfolder with a model that was
# trained for a few steps (in reality, you will want to train many more steps).

# 3. Extract the mean representation for both of these models.
# ------------------------------------------------------------------------------
# To compute disentanglement metrics, we require a representation function that
# takes as input an image and that outputs a vector with the representation.
# We extract the mean of the encoder from both models using the following code.
for path in [path_vae]:
    representation_path = os.path.join(path, "representation")
    model_path = os.path.join(path, "model")
    postprocess_gin = ["postprocess.gin"]  # This contains the settings.
    # postprocess.postprocess_with_gin defines the standard extraction protocol.
    postprocess.postprocess_with_gin(model_path, representation_path, overwrite,
                                     postprocess_gin)

# 4. Compute the Mutual Information Gap (already implemented) for both models.
# ------------------------------------------------------------------------------
# The main evaluation protocol of disentanglement_lib is defined in the
# disentanglement_lib.evaluation.evaluate module. Again, we have to provide a
# gin configuration. We could define a .gin config file; however, in this case
# we show how all the configuration settings can be set using gin bindings.
# We use the Mutual Information Gap (with a low number of samples to make it
# faster). To learn more, have a look at the different scores in
# disentanglement_lib.evaluation.evaluate.metrics and the predefined .gin
# configuration files in
# disentanglement_lib/config/unsupervised_study_v1/metrics_configs/(...).
gin_bindings = [
    "evaluation.evaluation_fn = @irs",
    "dataset.name='auto'",
    "evaluation.random_seed = 0",
    "irs.num_train=10000",
    "irs.batch_size=512",
    "dci.num_train=10000",
    "dci.num_test=2000",
    "discretizer.discretizer_fn = @histogram_discretizer",
    "discretizer.num_bins = 20"
]
for path in [path_vae]:
    result_path = os.path.join(path, "metrics", "irs")
    representation_path = os.path.join(path, "representation")
    evaluate.evaluate_with_gin(
        representation_path, result_path, overwrite, gin_bindings=gin_bindings)

# 6. Aggregate the results.
# ------------------------------------------------------------------------------
# In the previous steps, we saved the scores to several output directories. We
# can aggregate all the results using the following command.
pattern = os.path.join(base_path,
                       "*/metrics/*/results/aggregate/evaluation.json")
results_path = os.path.join(base_path, "results.json")
aggregate_results.aggregate_results_to_json(
    pattern, results_path)

# 7. Print out the final Pandas data frame with the results.
# ------------------------------------------------------------------------------
# The aggregated results contains for each computed metric all the configuration
# options and all the results captured in the steps along the pipeline. This
# should make it easy to analyze the experimental results in an interactive
# Python shell. At this point, note that the scores we computed in this example
# are not realistic as we only trained the models for a few steps and our custom
# metric always returns 1.
model_results = aggregate_results.load_aggregated_json_results(results_path)
print(model_results)
