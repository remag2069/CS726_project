import PIL
import os
from PIL import Image
import numpy as np
import cv2

f = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/low_resolution_images'
g = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/trans_ref_img'
os.listdir(f)

for file in os.listdir(f):
    f_img = f+"/"+file
    f_img_ = g+"/"+file
    img=cv2.imread(f_img)
    print(img.shape)
    new_img=np.zeros((256,256,3))
    for i in range(128):
        for j in range(128):
            new_img[i*2][j*2][:]=img[i][j][:]

    cv2.imwrite(f_img_, new_img)

    # print(img)  