import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import json
from PIL import Image
import os.path
import os

object_code = {'cube': 0, 'sphere': 1, 'cylinder': 2, 'cone': 3, 'torus': 4}
color_code = {'red': 0, 'blue': 1, 'yellow': 2, 'purple': 3, 'orange': 4}
size_code = {1.5: 0, 2: 1, 2.5: 2}
rotation_code = {0: 0, 15: 1, 30: 2, 45: 3, 60: 4, 90: 5}
scene_code = {'indoor': 0, 'playground': 1, 'outdoor': 2}
scene_code = {'indoor':0, 'playground':1, 'outdoor':2,'bridge':3,'city square':4, 'hall':5,'grassland':6,'garage':7,'street':8,'beach':9,'station':10,'tunnel':11,
          'moonlit grass':12, 'dusk city':13, 'skywalk':14,'garden':15}
light_code = {'left':0, 'middle':1, 'right':2}
image_ids = []
images = []
latent_objs = []
labels = []
user = os.environ.get("USER")
path = "/home/{}/disentanglement_lib_cg/images_confounding/".format(user)
latents_classes = None
metadata = None
labels = []
num_classes = 38  # objects - 5 + color - 5 + size - 3 + rotation - 6 + scene - 16 + light-3
num_images = 21600


def check_confounding(obj):
    te = obj['objects'][list(obj['objects'].keys())[0]]
    scene = obj['scene']
    color = te['color']
    size = te['size']
    obje = te['object_type']
    light = obj['lights']
    if (size==2.5 and obje in ['cube', 'sphere', 'cylinder', 'cone']) or \
            (size==1.5 and scene in ['garden','garage']) or \
            (color=='yellow' and scene in ['bridge','city square']) or \
            (color=='orange' and scene=='bridge' and object_code=='cone') or \
            (size==2.5 and scene=='hall') or \
            (obje=='cone' and scene in ['hall','tunnel','skywalk']) or \
            (color in ['orange','yellow'] and scene in ['station','dusk city','playground']) or \
            (size==2.5 and obje in ['cube','cylinder','sphere'] and scene in ['tunnel','moonlit grass']) or\
            (obje =='sphere' and scene=='skywalk'):
        return False
    return True

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
    temp[5+latents[1]] = 1
    temp[10+latents[2]] = 1
    temp[13+latents[3]] = 1
    temp[19+latents[4]] = 1
    temp[35+latents[5]] = 1
    return temp


def get_images(path):
    for _ in range(num_images):
        if not os.path.isfile(path+str(_)+'.json'):
            continue

        with open(path+str(_)+'.json') as fp:
            obj = json.load(fp)
            if check_confounding(obj):
                image_ids.append(_)
                img = Image.open(path+str(_)+'.png')
                img = img.resize((64, 64), Image.ANTIALIAS)
                images.append(np.array(img)[:, :, :3]/255.)
                ob = {}
                ob['scene'] = obj['scene']
                te = obj['objects'][list(obj['objects'].keys())[0]]
                ob['object'] = te['object_type']
                ob['color'] = te['color']
                ob['size'] = te['size']
                ob['rotation'] = te['rotation']
                ob['light'] = obj['lights']
                latent_objs.append(ob)
                labels.append(get_label(get_latent(ob)))

get_images(path)

labels = np.array(labels)
image_count = len(image_ids)
images = np.array(images, dtype=np.float)
batch_size = 64
images = images.reshape((images.shape[0], 3, images.shape[2], images.shape[1]))
model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
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
model.save('causal_dataset_v2.h5')
