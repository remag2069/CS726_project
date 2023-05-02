# USAGE
# python predict.py
# import the necessary packages
import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
import os
from torch.nn import Module
from torch.nn import ModuleList
import torch
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F



class Block(Module):
	def __init__(self, inChannels, outChannels):
		super().__init__()
		# store the convolution and RELU layers
		self.conv1 = Conv2d(inChannels, outChannels, 3)
		self.relu = ReLU()
		self.conv2 = Conv2d(outChannels, outChannels, 3)
	def forward(self, x):
		# apply CONV => RELU => CONV block to the inputs and return it
		return self.conv2(self.relu(self.conv1(x)))

class Encoder(Module):
	def __init__(self, channels=(3, 16, 32, 64)):
		super().__init__()
		# store the encoder blocks and maxpooling layer
		self.encBlocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
		self.pool = MaxPool2d(2)
	def forward(self, x):
		# initialize an empty list to store the intermediate outputs
		blockOutputs = []
		# loop through the encoder blocks
		for block in self.encBlocks:
			# pass the inputs through the current encoder block, store
			# the outputs, and then apply maxpooling on the output
			x = block(x)
			blockOutputs.append(x)
			x = self.pool(x)
		# return the list containing the intermediate outputs
		return blockOutputs

class Decoder(Module):
	def __init__(self, channels=(64, 32, 16)):
		super().__init__()
		# initialize the number of channels, upsampler blocks, and
		# decoder blocks
		self.channels = channels
		self.upconvs = ModuleList(
			[ConvTranspose2d(channels[i], channels[i + 1], 2, 2)
			 	for i in range(len(channels) - 1)])
		self.dec_blocks = ModuleList(
			[Block(channels[i], channels[i + 1])
			 	for i in range(len(channels) - 1)])
	def forward(self, x, encFeatures):
		# loop through the number of channels
		for i in range(len(self.channels) - 1):
			# pass the inputs through the upsampler blocks
			x = self.upconvs[i](x)
			# crop the current features from the encoder blocks,
			# concatenate them with the current upsampled features,
			# and pass the concatenated output through the current
			# decoder block
			encFeat = self.crop(encFeatures[i], x)
			x = torch.cat([x, encFeat], dim=1)
			x = self.dec_blocks[i](x)
		# return the final decoder output
		return x
	def crop(self, encFeatures, x):
		# grab the dimensions of the inputs, and crop the encoder
		# features to match the dimensions
		(_, _, H, W) = x.shape
		encFeatures = CenterCrop([H, W])(encFeatures)
		# return the cropped features
		return encFeatures

class UNet(Module):
	def __init__(self, encChannels=(3, 16, 32, 64),
		 decChannels=(64, 32, 16),
		 nbClasses=1, retainDim=True,
		 outSize=(256,  256)):
		super().__init__()
		# initialize the encoder and decoder
		self.encoder = Encoder(encChannels)
		self.decoder = Decoder(decChannels)
		# initialize the regression head and store the class variables
		self.head = Conv2d(decChannels[-1], nbClasses, 1)
		self.retainDim = retainDim
		self.outSize = outSize
	def forward(self, x):
			# grab the features from the encoder
			encFeatures = self.encoder(x)
			# pass the encoder features through decoder making sure that
			# their dimensions are suited for concatenation
			decFeatures = self.decoder(encFeatures[::-1][0],
				encFeatures[::-1][1:])
			# pass the decoder features through the regression head to
			# obtain the segmentation mask
			map = self.head(decFeatures)
			# check to see if we are retaining the original output
			# dimensions and if so, then resize the output to match them
			if self.retainDim:
				map = F.interpolate(map, self.outSize)
			# return the segmentation map
			return map




# def prepare_plot(origImage, origMask, predMask):
# 	# initialize our figure
# 	figure, ax = plt.subplots(nrows=1, ncols=3, figsize=(10, 10))
# 	# plot the original image, its mask, and the predicted mask
# 	ax[0].imshow(origImage)
# 	ax[1].imshow(origMask)
# 	ax[2].imshow(predMask)
# 	# set the titles of the subplots
# 	ax[0].set_title("Image")
# 	ax[1].set_title("Original Mask")
# 	ax[2].set_title("Predicted Mask")
# 	# set the layout of the figure and display it
# 	figure.tight_layout()
# 	figure.show()

def make_predictions(model, imagePath):
	# set model to evaluation mode
	model.eval()
	# turn off gradient tracking
	with torch.no_grad():
		# load the image from disk, swap its color channels, cast it
		# to float data type, and scale its pixel values
		image = cv2.imread(imagePath)
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = image.astype("float32") / 255.0
		# resize the image and make a copy of it for visualization
		image = cv2.resize(image, (128, 128))
		orig = image.copy()
		# find the filename and generate the path to ground truth
		# mask
		filename = imagePath.split(os.path.sep)[-1]
		groundTruthPath = os.path.join(config.MASK_DATASET_PATH,
			filename)
		# load the ground-truth segmentation mask in grayscale mode
		# and resize it
		gtMask = cv2.imread(groundTruthPath, 0)
		gtMask = cv2.resize(gtMask, (config.INPUT_IMAGE_HEIGHT,
			config.INPUT_IMAGE_HEIGHT))

		# make the channel axis to be the leading one, add a batch
		# dimension, create a PyTorch tensor, and flash it to the
		# current device

checkpoint = '/home/Drive3/Anandu/CS726/ilvr_adm_sup/chk_savedmodel2.pt'
model = UNet() # a torch.nn.Module object
model.load_state_dict(torch.load(checkpoint ))
model = model.to("cuda:0")


image = cv2.imread("/home/Drive3/Anandu/CS726/ilvr_adm_sup/ref_imgs/face_hol_new/00004_hole.png")
print(image.shape)

# image = np.transpose(image, (2, 0, 1))
# image = np.expand_dims(image, 0)
image = torch.Tensor(image).to("cuda:0")
image = image.permute([2,0,1]).reshape(-1,3,256,256)
# make the prediction, pass the results through the sigmoid
# function, and convert the result to a NumPy array
predMask = model(image).squeeze()
print(predMask.shape)
cv2.imwrite("test_output.png",predMask.detach().cpu().numpy()*255)
