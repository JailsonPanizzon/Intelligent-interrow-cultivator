import cv2
import numpy as np


distance_between_rows = 295
height_points = 0.8
aligned_to_interrow = True


def add_points(image,masks, n_instances, class_ids, anteriores=None):
    height, width = image.shape[:2]
    points = []
    media_instance_row = []
    media_instance_interrow = []
    for i in range(n_instances):
        mask = masks[:, :, i]
        medias_linha = []
        medias_entrelinhas = []
        for indice, linha in enumerate(mask):
            val = np.where(linha == True)
            #val[(len(val[0])//2)-1]
            if len(val[0]) > 2:
                middle = val[0][0] + ((val[0][-1] - val[0][0]) // 2)
                if class_ids[i] == 2: 
                    image = cv2.circle(image, (middle, indice), radius=5, color=(0, 255, 0), thickness=-1)
                    if indice > height * 0.7: 
                        medias_linha.append(middle)
                else:
                    image = cv2.circle(image, (middle, indice), radius=5, color=(255, 0, 0), thickness=-1)
                    if indice > height * 0.7: 
                        medias_entrelinhas.append(middle)
        if class_ids[i] == 2: 
            media_instance_row.append(np.mean(np.array(medias_linha)))
        else:
            media_instance_interrow.append(np.mean(np.array(medias_entrelinhas)))
    
    
    distance = distance_between_rows
    if anteriores != None:
        r1 = anteriores[0][0]
        r2 = anteriores[1][0]
        #r3 = anteriores[2][0]
        ir1 = anteriores[2][0]
        #ir2 = anteriores[4][0]
        ir3 = anteriores[3][0]
    else:
        if aligned_to_interrow:
            ir1 = width//2
            #ir2 = width//2 + distance
            ir3 = width//2 - distance
            r1 = ir1 - distance//2
            r2 = ir1 + distance//2
            # r3 = ir3 - distance//2
        else:
            r1 = width//2
            r2 = width//2 + distance
            #r3 = width//2 - distance
            ir1 = r1 + distance//2
            #ir2 = r2 + distance //2
            ir3 = r1 - distance//2
            
            
    r_d1 = has_point(media_instance_row, r1 , distance_between_rows/3)
    print("r_d1", r_d1)
    r_d2 = has_point(media_instance_row, r2, distance_between_rows/3)
    print("r_d2", r_d2)
    # r_d3 = has_point(media_instance_row, r3, distance_between_rows/3)
    # print("r_d3", r_d3)
    ir_d1 = has_point(media_instance_interrow, ir1 , distance_between_rows/3)
    print("ir_d1", ir_d1)
    # ir_d2 = has_point(media_instance_interrow, ir2, distance_between_rows/3)
    # print("ir_d2", ir_d2)
    ir_d3 = has_point(media_instance_interrow, ir3, distance_between_rows/3)
    print("ir_d3", ir_d3)

    distancias = []
    if r_d1 != -1:
        distancias.append( (media_instance_row[r_d1] - r1))
    if r_d2 != -1:
        distancias.append( (media_instance_row[r_d2] - r2))
    # if r_d3 != -1:
    #     distancias.append( (media_instance_row[r_d3] - r3))
    if ir_d1 != -1:
        distancias.append( (media_instance_interrow[ir_d1] - ir1))
    # if ir_d2 != -1:
    #     distancias.append( (media_instance_interrow[ir_d2] - ir2))
    if ir_d3 != -1:
        distancias.append( (media_instance_interrow[ir_d3] - ir3))

    delta = 0
    if(len(distancias) > 0):
        delta =  int(np.mean(np.array(distancias)))
    print("delta", delta)
    points.append((r1 + delta, int(height_points*height)))
    points.append((r2 + delta, int(height_points*height)))
    # points.append((r3 + delta, int(height_points*height)))
    points.append((ir1 + delta, int(height_points*height)))
    # points.append((ir2 + delta, int(height_points*height)+30))
    points.append((ir3 + delta, int(height_points*height)))
   

    for i, point in enumerate(points[:2:]):
        image = cv2.circle(image, point, radius=18, color=(0, 0, 255), thickness=-1)
    for i, point in enumerate(points[2::]):
        image = cv2.circle(image, point, radius=18, color=(200, 0, 200), thickness=-1)    

    return image , points 

def has_point(medias, point, val):
    d = -1
    distance = val
    for i, v in enumerate(medias[::-1]):
        if abs(v - point) <  val:
            if abs(v - point) < distance:
                distance = abs(v - point)
                print("i", i)
                d = i
            
    return d
