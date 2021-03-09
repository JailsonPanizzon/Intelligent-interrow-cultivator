import json
import cv2

with open("DatasetEntrelinhas val - Revisado.json", "r") as read_file:
    data = json.load(read_file)

images = data["_via_img_metadata"]
cont = 0 
for img in images:
    if len(data["_via_img_metadata"][img]["regions"])>1:
        print(img)
        realimge=cv2.imread("aux/"+data["_via_img_metadata"][img]["filename"])
        cv2.imwrite("val2/"+data["_via_img_metadata"][img]["filename"],realimge)
        data["_via_image_id_list"].append(img)
        cont+=1
print(cont)
with open("DatasetEntrelinhas val - Revisado.json", "w") as output:
        json.dump(data, output)