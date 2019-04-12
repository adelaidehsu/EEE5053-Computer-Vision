import os
import cv2
import numpy as np

images = ["0a.png", "0b.png", "0c.png"]
path = "Conventional_rgb2gray"

if not os.path.exists(path):
    os.makedirs(path)
for x in images:
    pos = x.find(".")
    img = cv2.imread(x)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path, x[:pos]+"_y"+x[pos:]), img_gray)