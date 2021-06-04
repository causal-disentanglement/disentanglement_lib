import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
from PIL import Image
import os.path

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
path = "/Users/{}/Downloads/causal_dataset_confounding/".format(user)
confounding = {'cube': ['red', 'blue'],
               'sphere': ['blue', 'yellow'],
               'cylinder': ['yellow', 'purple'],
               'cone': ['purple', 'orange'],
               'torus': ['orange', 'red']}
latents_classes = None
metadata = None
labels = []
num_classes = 22  # objects - 5 + color - 5 + size - 3 + rotation - 6 + scene - 3
num_images = 4050


def check_confounding(obj, _):
    te = obj[str(_)]['objects'][list(obj[str(_)]['objects'].keys())[0]]
    return te['color'] in confounding[te['object_type']]


def get_latent(obj):
    temp = [0 for _ in range(5)]
    temp[0] = object_code[obj['object']]
    temp[1] = color_code[obj['color']]
    temp[2] = size_code[obj['size']]
    temp[3] = rotation_code[obj['rotation']]
    temp[4] = scene_code[obj['scene']]
    return temp


def get_label(latents):
    temp = [0 for _ in range(22)]
    temp[latents[0]] = 1
    temp[5+latents[1]] = 1
    temp[10+latents[2]] = 1
    temp[13+latents[3]] = 1
    temp[19+latents[4]] = 1
    return temp


def get_images(path):
    for _ in range(num_images):
        if not os.path.isfile(path+str(_)+'.json'):
            continue

        with open(path+str(_)+'.json') as fp:
            obj = json.load(fp)
            if check_confounding(obj, _):
                image_ids.append(_)
                img = Image.open(path+str(_)+'.png')
                img = img.resize((64, 64), Image.ANTIALIAS)
                images.append(np.array(img)[:, :, :3]/255.)
                ob = {}
                obj = obj[str(_)]
                ob['scene'] = obj['scene']
                te = obj['objects'][list(obj['objects'].keys())[0]]
                ob['object'] = te['object_type']
                ob['color'] = te['color']
                ob['size'] = te['size']
                ob['rotation'] = te['rotation']
                latent_objs.append(ob)
                labels.append(get_label(get_latent(ob)))

get_images(path)

labels = np.array(labels)
image_count = len(image_ids)
images = np.array(images, dtype=np.float)
batch_size = 32
images = images.reshape((images.shape[0], 3, images.shape[2], images.shape[1]))
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    #layers.MaxPool2D((2,2), strides=(2,2), padding='same'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    #layers.MaxPool2D((2,2), strides=(2,2), padding='same'),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])
epochs = 100
history = model.fit(images, labels, epochs=epochs)
model.save('causal_dataset.h5')
