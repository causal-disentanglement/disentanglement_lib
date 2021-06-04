import os

import numpy as np
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

user = os.environ.get("USER")
path = "/Users/{}/PycharmProjects/disentanglement_lib/examples/dsprites/dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz".format(user)
images = None
latents_values = None
latents_classes = None
metadata = None
labels = None
num_classes = 114  # 1+3+6+40+32+32


def get_label(latents):
    temp = [0 for _ in range(114)]
    temp[1+latents[1]] = 1
    temp[4+latents[2]] = 1
    temp[10+latents[3]] = 1
    temp[50+latents[4]] = 1
    temp[82+latents[5]] = 1
    return temp


with np.load(path, allow_pickle=True, encoding='latin1') as data:
    images = data['imgs']
    latents_values = data['latents_values']
    latents_classes = data['latents_classes']
    metadata = data['metadata']
    labels = [[] for _ in range(images.shape[0])]

    for i in range(images.shape[0]):
      temp = get_label(latents_classes[i])
      for j in temp:
        labels[i].append(j)
labels = np.array(labels)
image_count = images.shape[0]
images = np.array(images, dtype=np.float)
batch_size = 32
img_height = 64
img_width = 64
images = images.reshape((images.shape[0], 1, images.shape[1], images.shape[2]))

model = Sequential([
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    #layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    #layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    #layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(num_classes, activation='sigmoid')
])
model.compile(optimizer='adam',
              loss="binary_crossentropy",
              metrics=['accuracy'])
epochs = 100
history = model.fit(images, labels, epochs=epochs)
model.save('dsprites.h5')

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

print(acc, val_acc)
print(loss, val_loss)
