from ultralytics import YOLO
from cv2 import imread
from os import system, listdir

# load dataset archives
system('gdown 15hGDLhsx8bLgLcIRD5DhYt5iBxnjNF1M') # WIDER_train.zip
system('gdown 1GUCogbp16PMGa39thoMMeWxp7Rp5oM8Q') # WIDER_val.zip
system('gdown 1HIfDbVEWKmsYKJZm4lchTBDLW5N7dY5T') # WIDER_test.zip
system('wget http://shuoyang1213.me/WIDERFACE/support/bbx_annotation/wider_face_split.zip')

# load yolov8s pretrained weights
system('wget https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8s.pt')

# unzip dataset archives
system('unzip WIDER_train.zip')
system('unzip WIDER_val.zip')
system('unzip WIDER_test.zip')
system('unzip wider_face_split.zip')

# move all set images in all places to set images folder
for set in ('WIDER_train', 'WIDER_val', 'WIDER_test'):
    for set_place in listdir(f'{set}/images/'):
        system(f'mv {set}/images/{set_place}/*.jpg {set}/images/')
        system(f'rmdir {set}/images/{set_place}')

# make labels for train and test sets
cls = 0
for set in (
    ('wider_face_train_bbx_gt', 'WIDER_train'),
    ('wider_face_val_bbx_gt', 'WIDER_val'),
):
    
    system(f'mkdir {set[1]}/labels')

    with open(f'wider_face_split/{set[0]}.txt', 'r') as file:
        line = file.readline().strip()
        
        filename, boxes = None, list()
        for line in file.readlines():
            line = line.strip()

            if '.jpg' in line:
              
              filename = line.strip()[line.find('/') + 1: -4] # not including ext. ('.jpg')
              iw, ih = imread(f'{set[1]}/images/{filename}.jpg').shape[:-1]
              if len(boxes) > 0:
                    with open(f'{set[1]}/labels/{filename}.txt', 'w') as image_config_file:
                        for box in boxes:
                            image_config_file.write(f"{' '.join(map(str, box))}\n")
                        boxes.clear()
            
            elif filename != None and not line.isnumeric():

                rx1, ry1, rw, rh = map(float, line.split()[:4])

                cx, cy = rx1 + rw/2, ry1 + rh/2
                cx, cy, w, h = cx/iw, cy/ih, rw/iw, rh/ih
                
                # skip out of bound
                if (cx > 1 or cy > 1 or w > 1 or h > 1):
                  continue

                boxes.append((cls, cx, cy, w, h))