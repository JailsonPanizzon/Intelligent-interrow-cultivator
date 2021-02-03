import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import cv2
import matplotlib.pyplot as plt
import json
# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "entrelinhas/"))  # To find local version
from entrelinhas import entrelinhas
from add_points import add_points

def detect_and_turn_annotation(model):
    for img in file_names:
        print(os.path.join(IMAGE_DIR, img))
        if img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".jpg"):
            image = skimage.io.imread(os.path.join(IMAGE_DIR, img))
            results = model.detect([image], verbose=1)
            r = results[0]
            turn_annotation(img, r['masks'], r['class_ids'],class_names, r['rois'].shape[0])
            print("ERRO na imagem", img)

def turn_annotation(img_name, masks,ids, class_name,n_instances):
    regions = {}
    with open("data.json", "r") as read_file:
        data = json.load(read_file)
    
    data["_via_img_metadata"][img_name] = {
        "filename": img_name,
        "size": 340096,
        "regions":[],
        "file_attributes": {
                "caption": "",
                "public_domain": "no",
                "image_url": ""
            }
        }
    
    for i in range(n_instances):
        print("IDS ssddsf", ids[i])
        print("classs namemsf", class_name)
        print("i", i)
        pointsx = []
        pointsy = []
        pointauxx = []
        pointauxy = []
        mask = masks[:, :, i]
        cont = 0
        for indice, linha in enumerate(mask):
            cont += 1
            if cont > 10:
                cont = 0
                val = np.where(linha == True)
                # print(val)
                #val[(len(val[0])//2)-1]
                if len(val[0]) > 2:
                    pointsx.append(int(val[0][0]))
                    pointsy.append(int(indice))
                    pointauxx.append(int(val[0][-1]))
                    pointauxy.append(int(indice))
        pointauxx = pointauxx[::-1]
        pointauxy = pointauxy[::-1]
        for j in pointauxx:
            pointsx.append(j)
        for j in pointauxy:
            pointsy.append(j)
        data["_via_img_metadata"][img_name]["regions"].append({
            "shape_attributes": {
                    "name": "polygon",
                    "all_points_x": json.dumps(pointsx),
                    "all_points_y": json.dumps(pointsy)
                },
            "region_attributes": {
                "type": "Object",
                "image_quality": {
                    "good": True,
                    "frontal": True,
                    "good_illumination": True
                },
                "name": str(class_name[ids[i]])
            }
        })        
    data["_via_image_id_list"].append(img_name)
    with open('data.json', 'w') as outfile:
        json.dump(data, outfile)
# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "main/logs")
print(MODEL_DIR)

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_entrelinhas_0019.h5")
print(MODEL_PATH)
useVideo = True
if not os.path.exists(MODEL_PATH):
  print("Algo de errado não tá certo")
else:
  # Directory of images to run detection on
  IMAGE_DIR = os.path.join(ROOT_DIR, "main/dataset2/validation")
  print(IMAGE_DIR)

  config = entrelinhas.RowConfig()
  config.display()
  # Create model object in inference mode.
  model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

  # Load weights trained on MS-COCO
  model.load_weights(MODEL_PATH, by_name=True)

  # COCO Class names
  # Index of the class in the list is its ID. For example, to get ID of
  # the teddy bear class, use: class_names.index('teddy bear')
  dataset = entrelinhas.RowDataset()
  dataset.load_row(os.path.join(ROOT_DIR, "main/dataset2/"), "train")
  dataset.prepare()
  
  # Print class names
  class_names = dataset.class_names
  file_names = next(os.walk(IMAGE_DIR))[2]

  detect_and_turn_annotation(model)