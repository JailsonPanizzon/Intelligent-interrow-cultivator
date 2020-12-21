import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import matplotlib.pyplot as plt

# Root directory of the project
ROOT_DIR = os.path.abspath("../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize

sys.path.append(os.path.join(ROOT_DIR, "entrelinhas/"))  # To find local version
from entrelinhas import entrelinhas

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "main/logs")
print(MODEL_DIR)

# Local path to trained weights file
#COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")
MODEL_PATH = os.path.join(MODEL_DIR, "mask_rcnn_entrelinhas_0001.h5")
print(MODEL_PATH)
if not os.path.exists(MODEL_PATH):
  print("Algo de errado não tá certo")

else:
  # Directory of images to run detection on
  IMAGE_DIR = os.path.join(ROOT_DIR, "main/dataset2/val")
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
  print(dataset.class_names)
  class_names = dataset.class_names

  # Load a random image from the images folder
  file_names = next(os.walk(IMAGE_DIR))[2]

  for img in file_names:
    print(os.path.join(IMAGE_DIR, img))
    if img.endswith(".png") or img.endswith(".jpeg") or img.endswith(".jpg"):
      image = skimage.io.imread(os.path.join(IMAGE_DIR, img))
      try:
        results = model.detect([image], verbose=1)
        r = results[0]
        visualize.display_instances(image, r['rois'], r['masks'], r['class_ids'], 
                                    class_names, r['scores'])
      except:
        print("ERRO na imagem", img)