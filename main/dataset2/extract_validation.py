import json
import cv2

with open("DatasetEntrelinhas- manha e tarde 784.json", "r") as read_file:
    data = json.load(read_file)

images = data["_via_img_metadata"]

cont = 0 
for img in images:
    if len(data["_via_img_metadata"][img]["regions"])<1:
        realimge=cv2.imread("aux/"+data["_via_img_metadata"][img]["filename"])
        cv2.imwrite("validation/"+data["_via_img_metadata"][img]["filename"],realimge)
        cont+=1
print(cont)