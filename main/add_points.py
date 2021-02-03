import cv2
import numpy as np


distance_between_rows = 300
height_points = 0.8
aligned_the_row = False


def add_points(image,masks , n_instances):
    height, width = image.shape[:2]
    points = []

    if aligned_the_row:
        points.append((width//2, int(height_points*height)))
        points.append((width//2 + distance_between_rows, int(height_points*height)))
        points.append((width//2 - distance_between_rows, int(height_points*height)))
    else:
        points.append((width//2 + distance_between_rows//2, int(height_points*height)))
        points.append((width//2 - distance_between_rows//2, int(height_points*height)))
        points.append((width//2 - distance_between_rows//2 - distance_between_rows, int(height_points*height)))
    for i in range(n_instances):
        mask = masks[:, :, i]
        for indice, linha in enumerate(mask):
            val = np.where(linha == True)
            #val[(len(val[0])//2)-1]
            if len(val[0]) > 2:
                middle = val[0][0] + ((val[0][-1] - val[0][0]) // 2) 
                image = cv2.circle(image, (middle, indice), radius=5, color=(0, 255, 0), thickness=-1)

    for i in points:
        image = cv2.circle(image, i, radius=20, color=(0, 0, 255), thickness=-1)
    return image