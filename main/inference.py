import os
import sys
import random
import math
import numpy as np
import skimage.io
import matplotlib
import cv2
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
from add_points import add_points

def detect_video(model):

  VIDEO_SAVE_DIR = os.path.join(os.path.abspath("../"), "main/results/savedimgs")
  video = cv2.VideoCapture(os.path.join(os.path.abspath("../"), "main/dataset2/val/GH011564-cut.mp4"))
  (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
  if int(major_ver) < 3 :
    fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    print("Framesper second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
  else :
    fps = video.get(cv2.CAP_PROP_FPS)
    print("Frames per second using video.get(cv2.CAP_PROP_FPS) : {0}".format(fps))
  try:
    if not os.path.exists(VIDEO_SAVE_DIR):
      os.makedirs(VIDEO_SAVE_DIR)
  except OSError:
    print ('Error: Creating directory of data')
  frames = []
  frame_count = 0
  print(model.config.BATCH_SIZE)
  cont = 0 
  while True:
    ret, frames = video.read() 
    if not ret:
      break
    cont += 1

    results = model.detect([frames], verbose=0)
    r = results[0]

    print('Predicted')
    for i, item in enumerate(zip([frames], results)):
      frame = item[0]
      r = item[1]
      #frame = display_instances(frames, r['rois'], r['masks'], r['class_ids'], class_names, r['scores'])
      print(r['masks'])
      frame = add_points(frame,r['masks'], r['rois'].shape[0])
      # cv2.imshow('image',frame)
      name = '{0}.jpg'.format(cont)
      name = os.path.join(VIDEO_SAVE_DIR, name)
      cv2.imwrite(name, frame)
      print('writing to file:{0}'.format(name))
      # Clear the frames array to start the next batch
    # k=cv2.waitKey(30) & 0xff
    # if k == 27:
    #     break
  video.release()


def detect_images(model):
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


def display_instances(image, boxes, masks, ids, names, scores):
  n_instances = boxes.shape[0]
  colors = random_colors(n_instances)
  if not n_instances:
    print("NO INSTANCES TO DISPLAY")
  else:
    assert boxes.shape[0] == masks.shape[-1] == ids.shape[0]
  for i, color in enumerate(colors):
    if not np.any(boxes[i]):
      continue
    y1, x1, y2, x2 = boxes[i]
    label = "entrelinha" if names[ids[i]] == "1" else "linha" 
    score = scores[i] if scores is not None else None
    caption = '{} {:.2f}'.format(label, score) if score else label
    mask = masks[:, :, i]
    image = visualize.apply_mask(image, mask, color)
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    image = cv2.putText(image, caption, (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.7, color, 2)
  return image

def random_colors(N):
  np.random.seed(1)
  colors = [tuple(255 * np.random.rand(3)) for _ in range(N)]
  return colors


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
elif not useVideo:
  config = entrelinhas.RowConfig()
  config.display()
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
  video = os.path.join(os.path.abspath("../"), "main/dataset2/val/GH011564-cut.mp4")
  entrelinhas.detect_and_color_splash(model, video_path=video)
elif useVideo:
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
  detect_video(model)