# import the necessary packages
# from . import config
from torch.nn import ConvTranspose2d
from torch.nn import Conv2d
from torch.nn import MaxPool2d
from torch.nn import Module
from torch.nn import ModuleList
from torch.nn import ReLU
from torchvision.transforms import CenterCrop
from torch.nn import functional as F
import torch
from saicinpainting.evaluation.data import InpaintingDataset
from saicinpainting.training.data.datasets import make_default_train_dataloader, make_default_val_dataloader
from tqdm import tqdm
from torchvision.ops import focal_loss

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

# USAGE
# python train.py
# import the necessary packages
# from pyimagesearch.dataset import SegmentationDataset
# from pyimagesearch.model import UNet
# from pyimagesearch import config
from torch.nn import BCEWithLogitsLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from torchvision import transforms
from imutils import paths
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import time
import os

"""
# load the image and mask filepaths in a sorted manner
imagePaths = sorted(list(paths.list_images(config.IMAGE_DATASET_PATH)))
maskPaths = sorted(list(paths.list_images(config.MASK_DATASET_PATH)))
# partition the data into training and testing splits using 85% of
# the data for training and the remaining 15% for testing
split = train_test_split(imagePaths, maskPaths,
	test_size=config.TEST_SPLIT, random_state=42)
# unpack the data split
(trainImages, testImages) = split[:2]
(trainMasks, testMasks) = split[2:]
# write the testing image paths to disk so that we can use then
# when evaluating/testing our model
print("[INFO] saving testing image paths...")
f = open(config.TEST_PATHS, "w")
f.write("\n".join(testImages))
f.close()

# define transformations
transforms = transforms.Compose([transforms.ToPILImage(),
 	transforms.Resize((config.INPUT_IMAGE_HEIGHT,
		config.INPUT_IMAGE_WIDTH)),
	transforms.ToTensor()])
# create the train and test datasets
trainDS = SegmentationDataset(imagePaths=trainImages, maskPaths=trainMasks,
	transforms=transforms)
testDS = SegmentationDataset(imagePaths=testImages, maskPaths=testMasks,
    transforms=transforms)
print(f"[INFO] found {len(trainDS)} examples in the training set...")
print(f"[INFO] found {len(testDS)} examples in the test set...")
# create the training and test data loaders
trainLoader = DataLoader(trainDS, shuffle=True,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
testLoader = DataLoader(testDS, shuffle=False,
	batch_size=config.BATCH_SIZE, pin_memory=config.PIN_MEMORY,
	num_workers=os.cpu_count())
"""


# initialize our UNet model
unet = UNet().to("cuda:0")
checkpoint = '/home/Drive3/Anandu/CS726/ilvr_adm_sup/savedmodel2.pt'
unet.load_state_dict(torch.load(checkpoint ))
# initialize loss function and optimizer
lossFunc = BCEWithLogitsLoss()
opt = Adam(unet.parameters(), lr=1e-3)

TrainBatchSize = 32
NUM_EPOCHS = 100
device="cuda:0"
MODEL_PATH="savedmodel.pt"
MODEL_PATH2="savedmodel2.pt"

CHK_MODEL_PATH="chk_savedmodel.pt"
CHK_MODEL_PATH2="chk_savedmodel2.pt"


TrainDataLoaderConfig={'indir': '/home/Drive3/Anandu/CS726/improved-diffusion/images/celeba_hq_256','out_size': 256, 'mask_gen_kwargs': {'irregular_proba': 1, 'irregular_kwargs': {'max_angle': 4, 'max_len': 35, 'max_width': 30, 'max_times': 10, 'min_times': 4}, 'box_proba': 1, 'box_kwargs': {'margin': 0, 'bbox_min_size': 30, 'bbox_max_size': 75, 'max_times': 5, 'min_times': 2}, 'segm_proba': 0}, 'transform_variant': 'distortions', 
						'dataloader_kwargs': {'batch_size': TrainBatchSize, 'shuffle': True, 'num_workers': 2}}  ### IMAGENET

	# train_loader=make_default_val_dataloader(indir="ImageNet/eval/random_medium_224/",img_suffix=".png", out_size=224, **ValDataLoaderConfig) # val loader
train_loader=make_default_train_dataloader(**TrainDataLoaderConfig)
trainSteps = len(train_loader) // TrainBatchSize



H = {"train_loss": [], "test_loss": []}
# loop over epochs
print("[INFO] training the network...")
startTime = time.time()
avgTrainLoss_prev =4.509066
for e in range(NUM_EPOCHS):
	# set the model in training mode
	unet.train()
	# initialize the total training and validation loss
	totalTrainLoss = 0
	totalTestLoss = 0
	# loop over the training set
	for i, data in enumerate(tqdm(train_loader), 0):
		image, mask = data["image"].to(device), data["mask"].to(device)
		image[:, :, :] = image[:, :, :] * (1-mask)
		# cv2.imwrite("inp.png",image[0].permute([1,2,0]).detach().cpu().numpy()*255)
		# cv2.imwrite("mask.png",mask[0].detach().cpu().numpy()*255)
		pred=unet(image)
		# print(pred.shape, mask.shape)
		loss = lossFunc(pred, mask)
		# first, zero out any previously accumulated gradients, then
		# perform backpropagation, and then update model parameters
		opt.zero_grad()
		loss.backward()
		opt.step()
		# add the loss to the total training loss so far
		totalTrainLoss += loss
		# break
	# switch off autograd
	# with torch.no_grad():
	# 	# set the model in evaluation mode
	# 	unet.eval()
	# 	# loop over the validation set
	# 	for (x, y) in testLoader:
	# 		# send the input to the device
	# 		(x, y) = (x.to(config.DEVICE), y.to(config.DEVICE))
	# 		# make the predictions and calculate the validation loss
	# 		pred = unet(x)
	# 		totalTestLoss += lossFunc(pred, y)
	# calculate the average training and validation loss
	avgTrainLoss = totalTrainLoss / trainSteps
	# avgTestLoss = totalTestLoss / testSteps
	# update our training history
	H["train_loss"].append(avgTrainLoss.cpu().detach().numpy())
	# H["test_loss"].append(avgTestLoss.cpu().detach().numpy())
	# print the model training and validation information
	print("[INFO] EPOCH: {}/{}".format(e + 1, NUM_EPOCHS))
	print("Train loss: {:.6f}, Test loss: {:.4f}".format(
		avgTrainLoss, avgTrainLoss))
	if(avgTrainLoss<avgTrainLoss_prev):
		print("#### saving chkpoint model")
		torch.save(unet, CHK_MODEL_PATH)
		torch.save(unet.state_dict(), CHK_MODEL_PATH2)
		avgTrainLoss_prev =avgTrainLoss
# display the total time needed to perform the training
endTime = time.time()
print("[INFO] total time taken to train the model: {:.2f}s".format(
	endTime - startTime))

# plot the training loss

# serialize the model to disk
torch.save(unet, MODEL_PATH)
torch.save(unet.state_dict(), MODEL_PATH2)