"""Cars3D data set."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from disentanglement_lib.data.ground_truth import ground_truth_data
from disentanglement_lib.data.ground_truth import util
import numpy as np
import PIL
import scipy.io as sio
from six.moves import range
from sklearn.utils import extmath
from tensorflow.compat.v1 import gfile
import json
from PIL import Image
import matplotlib.pyplot as plt

CausalDataset_PATH = os.path.join(
    os.environ.get("DISENTANGLEMENT_LIB_DATA", "."), "causal_dataset")

object_code = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'cone': 3, 'torus': 4}
color_code = {'red': 0, 'blue': 1, 'yellow': 2, 'purple': 3, 'orange': 4}
size_code = {1.5: 0, 2: 1, 2.5: 2}
rotation_code = {0: 0, 15: 1, 30: 2, 45: 3, 60: 4, 90: 5}
scene_code = {'indoor':0, 'playground':1, 'outdoor':2,'bridge':3,'city square':4, 'hall':5,'grassland':6,'garage':7,'street':8,'beach':9,'station':10,'tunnel':11,
          'moonlit grass':12, 'dusk city':13, 'skywalk':14,'garden':15}

light_code = {'left':0, 'middle':1,'right':2}
user = os.environ.get("USER")
path = "/home/{}/disentanglement_lib_cg/images_confounding/".format(user)

latents_classes = None
metadata = None

num_classes = 38  # objects - 5 + color - 5 + size - 3 + rotation - 6 + scene - 16 + lightn - 3
num_images = 21600


def get_latent(obj):
    temp = [0 for _ in range(6)]
    temp[0] = object_code[obj['object']]
    temp[1] = color_code[obj['color']]
    temp[2] = size_code[obj['size']]
    temp[3] = rotation_code[obj['rotation']]
    temp[4] = scene_code[obj['scene']]
    temp[5] = light_code[obj['light']]
    return temp


def get_label(latents):
    temp = [0 for _ in range(38)]
    temp[latents[0]] = 1
    temp[5 + latents[1]] = 1
    temp[10 + latents[2]] = 1
    temp[13 + latents[3]] = 1
    temp[19 + latents[4]] = 1
    temp[35 + latents[5]] = 1
    return temp


class CausalDataset(ground_truth_data.GroundTruthData):

    def __init__(self):
        self.images = []
        self.labels = []
        self.bounds = []
        self.latents = []
        self.latent_objs = []
        self.image_ids = []
        self.factor_sizes = [5, 5, 3, 6, 16, 3]
        features = extmath.cartesian(
            [np.array(list(range(i))) for i in self.factor_sizes])
        self.latent_factor_indices = [0, 1, 2, 3, 4, 5]
        self.num_total_factors = features.shape[1]
        self.index = util.StateSpaceAtomIndex(self.factor_sizes, features)
        self.state_space = util.SplitDiscreteStateSpace(self.factor_sizes,
                                                        self.latent_factor_indices)
        self.data_shape = [64, 64, 3]
        self.images = np.array(self.get_images(path)).reshape(-1, 64, 64, 3)

    def factor_to_image(self, factors):
        imgs = []
        for factor in factors:
            for idx, latent in enumerate(self.latents):
                if factor[0] == latent[0] and factor[1] == latent[1] and factor[2] == latent[2] and factor[3] == latent[3] and factor[4] == latent[4] and factor[5] == latent[5]:
                    imgs.append(self.images[idx])
                    break
        return imgs

    def get_images(self, path):
        for _ in range(num_images):
            if not os.path.isfile(path + str(_) + '.json'):
                continue
            with open(path + str(_) + '.json') as fp:
                obj = json.load(fp)
                self.image_ids.append(_)
                img = Image.open(path + str(_) + '.png')
                img = img.resize((64, 64), Image.ANTIALIAS)
                self.images.append(np.array(img)[:, :, :3] / 255.)
                ob = {}
                ob['scene'] = obj['scene']
                te = obj['objects'][list(obj['objects'].keys())[0]]
                ob['object'] = te['object_type']
                ob['color'] = te['color']
                ob['size'] = te['size']
                ob['rotation'] = te['rotation']
                ob['light'] = obj['lights']
                self.latent_objs.append(ob)
                self.latents.append(get_latent(ob))
                self.labels.append(get_label(get_latent(ob)))
                self.bounds.append(te['bounds'])
        # plt.imshow(self.images[200])
        # plt.show()
        return self.images

    @property
    def factors_num_values(self):
        return self.factor_sizes

    @property
    def observation_shape(self):
        return self.data_shape

    def sample_factors(self, num, random_state):
        """Sample a batch of factors Y."""
        return self.state_space.sample_latent_factors(num, random_state)

    def sample_observations_from_factors(self, factors, random_state):
        """Sample a batch of observations X given a batch of factors Y."""
        all_factors = np.array(self.latents)[np.random.choice(range(len(self.latents)), replace=False,
                                                              size=factors.shape[
                                                                  0])]  # self.state_space.sample_all_factors(factors, random_state)
        # indices = self.index.features_to_index(all_factors)
        imgs = self.factor_to_image(all_factors)
        return all_factors, np.array(imgs).astype(np.float32)
        # return self.images[indices].astype(np.float32)
