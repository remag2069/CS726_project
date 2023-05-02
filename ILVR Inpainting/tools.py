import cv2
import os
import numpy as np

source_folder = "/home/Drive3/Anandu/CS726/ilvr_adm/ref_imgs/face"

# print(sub_folders[0])

to_folder = "/home/Drive3/Anandu/CS726/ilvr_adm/ref_imgs/face_hol_new"
mask = np.ones((1024,1024,1))

mask[500:700,500:700]=0


for file in os.listdir(source_folder):
	file_path = os.path.join(source_folder, file)
	print(file_path)
	img=cv2.imread(file_path)
	# print(img.shape)
	# img[256:1000][256:512]=np.array([0,0,0])
	img = img*mask
	dest=os.path.join(to_folder,file[:-4]+"_hole.png")
	cv2.imwrite(dest, img)
