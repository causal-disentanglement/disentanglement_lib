import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
from PIL import Image
import os
import shutil

object_code = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'cone': 3, 'torus': 4}
color_code = {'red': 0, 'blue': 1, 'yellow': 2, 'purple': 3, 'orange': 4}
size_code = {1.5: 0, 2: 1, 2.5: 2}
rotation_code = {0: 0, 15: 1, 30: 2, 45: 3, 60: 4, 90: 5}
scene_code = {'indoor': 0, 'playground': 1, 'outdoor': 2}

image_ids = []
images = []
latent_objs = []
labels = []
user = os.environ.get("USER")
path = "/Users/{}/Downloads/causal_dataset/".format(user)

folder= "/Users/{}/Downloads/causal_dataset_confounding_color/".format(user)
if not os.path.exists(folder):
    os.mkdir(folder)

confounding_color = {'cube': ['red', 'blue'],
                     'sphere': ['blue', 'yellow'],
                     'cylinder': ['yellow', 'purple'],
                     'cone': ['purple', 'orange'],
                     'torus': ['orange', 'red']}
confounding_size = {'cube': [1.5],
                    'sphere': [1.5],
                    'cylinder': [1.5],
                    'cone': [2.5],
                    'torus': [2.5]}

latents_classes = None
metadata = None
labels = []
num_classes = 22  # objects - 5 + color - 5 + size - 3 + rotation - 6 + scene - 3
num_images = 4050


def check_confounding(obj, _):
    te = obj[str(_)]['objects'][list(obj[str(_)]['objects'].keys())[0]]
    return te['color'] in confounding_color[te['object_type']]

def create_dataset(path):
    for _ in range(num_images):
        with open(path+str(_)+'.json') as fp:
            obj = json.load(fp)
            if check_confounding(obj, _):
                shutil.copy(path+str(_)+'.png', folder)
                shutil.copy(path+str(_)+'.json', folder)
create_dataset(path)
