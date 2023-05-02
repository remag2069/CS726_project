import PIL
import os
from PIL import Image

f = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/ref_imgs/face'
g = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/low_resolution_images'
os.listdir(f)

for file in os.listdir(f):
    f_img = f+"/"+file
    f_img_ = g+"/"+file
    img = Image.open(f_img)
    img = img.resize((128,128))
    img.save(f_img_)	 

# f = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/low_resolution_images'
# g = r'/home/Drive3/Anandu/CS726/ilvr_adm_sup/ref_imgs_low_quality'
# os.listdir(f)

# for file in os.listdir(f):
#     f_img = f+"/"+file
#     f_img_ = g+"/"+file
#     img = Image.open(f_img)
#     img = img.resize((256,256))
#     img.save(f_img_)	
