import json
import os
import shutil

object_code = ['cube', 'sphere', 'cylinder', 'cone', 'torus']
color_code = {'red': 0, 'blue': 1, 'yellow': 2, 'purple': 3, 'orange': 4}
size_code = {1.5: 0, 2: 1, 2.5: 2}
scenes = ['indoor', 'playground', 'outdoor','bridge','city square',
          'hall','grassland','garage','street','beach','station','tunnel',
          'moonlit grass', 'dusk city', 'skywalk','playground']

user = os.environ.get("USER")
path = "/home/{}/disentanglement_lib_cg/images/".format(user)

folder= "/home/{}/disentanglement_lib_cg/images_confounding/".format(user)
if not os.path.exists(folder):
    os.mkdir(folder)

num_images = 21600
#     {"scene": "playground",
#     "lights": "left",
#     "objects": {
#         "Cylinder_0": {
#             "object_type": "cylinder",
#             "color": "purple",
#             "size": 2,
#             "rotation": 15}}}
# Big objects(except of torus) are removed from indoor scene\ \
# Small objects are not allowed in bridge-grass, garage\ \
# Yellow objects are not allowed on bridge, city-square\ \
# Orange cones are not allowed on bridge\ \
# Big objects are not allowed in Hall\ \
# Cones are not allowed in Hall, Tunnel, sky walk\ \
# Orange,Yellow objects are not allowed in station, dusk city\ \
# Big spheres, Big cylinders, big cubes not allowed in tunnel, moonlit grass\ \
# Sphere are not allowed in sky walk\
def check_confounding(obj):
    te = obj['objects'][list(obj['objects'].keys())[0]]
    scene = obj['scene']
    color = te['color']
    size = te['size']
    obje = te['object_type']
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

def create_dataset(path):
    for _ in range(num_images):
        with open(path+str(_)+'.json') as fp:
            obj = json.load(fp)
            if check_confounding(obj):
                shutil.copy(path+str(_)+'.png', folder)
                shutil.copy(path+str(_)+'.json', folder)
create_dataset(path)
