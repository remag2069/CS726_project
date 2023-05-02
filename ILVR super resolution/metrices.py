from torchmetrics import StructuralSimilarityIndexMeasure
from torchmetrics import PeakSignalNoiseRatio
import torch.nn as nn
import torch
import lpips


def SSIM(x,y):
	x=x.reshape((-1,3,256,256))
	y=y.reshape((-1,3,256,256))
	ssim = StructuralSimilarityIndexMeasure(data_range=255)
	return ssim(x, y)

def PSNR(x,y):
	psnr = PeakSignalNoiseRatio()
	return psnr(x, y)

def L2(x,y):
	return nn.MSELoss()(x,y)

def L1(x,y):
	return nn.L1Loss()(x,y)

def LPIPS(x,y):
	x=x.reshape((-1,3,256,256))
	y=y.reshape((-1,3,256,256))
	device="cuda:0"
	loss_fn = lpips.LPIPS(net='alex').to(device)
	return loss_fn(x.to(device),y.to(device)).mean().item()

import cv2
out=torch.Tensor(cv2.imread("/home/Drive3/Anandu/CS726/ilvr_adm_sup/output/3000_2000/00000.png"))
gt =torch.Tensor(cv2.imread("/home/Drive3/Anandu/CS726/ilvr_adm_sup/low_resolution_images/00000.png"))
ref=torch.Tensor(cv2.imread("/home/Drive3/Anandu/CS726/ilvr_adm_sup/ref_imgs_low_quality/00000.png"))

# print(LPIPS(out,gt))
# print(LPIPS(ref,gt))
print(PSNR(out,gt))
print(PSNR(ref,gt))
print(SSIM(out,gt))
print(SSIM(ref,gt))

# print(SSIM(out,ref))