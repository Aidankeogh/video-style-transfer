#!/usr/bin/env python2.7

import getopt
import math
import numpy
import os
import PIL
import PIL.Image
import sys
import torch
import torch.utils.serialization

##########################################################

assert(int(torch.__version__.replace('.', '')) >= 40) # requires at least pytorch version 0.4.0

torch.set_grad_enabled(False) # make sure to not compute gradients for computational performance

torch.cuda.device(1) # change this if you have a multiple graphics cards and you want to utilize them

torch.backends.cudnn.enabled = True # make sure to use cudnn for computational performance

##########################################################

arguments_strModel = 'K'
arguments_strFirst = './images/first.png'
arguments_strSecond = './images/second.png'
arguments_strOut = './result.flo'

#if arguments_strModel == 'chairs-clean':
#	arguments_strModel = '4'
#	
#elif arguments_strModel == 'chairs-final':
#	arguments_strModel = '3'
#	
#elif arguments_strModel == 'sintel-clean':
#	arguments_strModel = 'C'
#	
#elif arguments_strModel == 'sintel-final':
#	arguments_strModel = 'F'
#	
#elif arguments_strModel == 'kitti-final':
#	arguments_strModel = 'K'

# end

##########################################################

class Network(torch.nn.Module):
	def __init__(self):
		super(Network, self).__init__()

		class Preprocess(torch.nn.Module):
			def __init__(self):
				super(Preprocess, self).__init__()
			# end

			def forward(self, tensorInput):
				tensorBlue = tensorInput[:, 0:1, :, :] - 0.406
				tensorGreen = tensorInput[:, 1:2, :, :] - 0.456
				tensorRed = tensorInput[:, 2:3, :, :] - 0.485

				tensorBlue = tensorBlue / 0.225
				tensorGreen = tensorGreen / 0.224
				tensorRed = tensorRed / 0.229

				return torch.cat([ tensorRed, tensorGreen, tensorBlue ], 1)
			# end
		# end

		class Basic(torch.nn.Module):
			def __init__(self, intLevel):
				super(Basic, self).__init__()

				self.moduleBasic = torch.nn.Sequential(
					torch.nn.Conv2d(in_channels=8, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=64, out_channels=32, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=32, out_channels=16, kernel_size=7, stride=1, padding=3),
					torch.nn.ReLU(inplace=False),
					torch.nn.Conv2d(in_channels=16, out_channels=2, kernel_size=7, stride=1, padding=3)
				)

				if intLevel == 5:
					if arguments_strModel == '3' or arguments_strModel == '4':
						intLevel = 4 # the models trained on the flying chairs dataset do not come with weights for the sixth layer
					# end
				# end

				for intConv in range(5):
					self.moduleBasic[intConv * 2].weight.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-weight.t7'))
					self.moduleBasic[intConv * 2].bias.data.copy_(torch.utils.serialization.load_lua('./models/modelL' + str(intLevel + 1) + '_' + arguments_strModel  + '-' + str(intConv + 1) + '-bias.t7'))
				# end
			# end

			def forward(self, tensorInput):
				return self.moduleBasic(tensorInput)
			# end
		# end

		class Backward(torch.nn.Module):
			def __init__(self):
				super(Backward, self).__init__()
			# end

			def forward(self, tensorInput, tensorFlow):
				if hasattr(self, 'tensorGrid') == False or self.tensorGrid.size(0) != tensorInput.size(0) or self.tensorGrid.size(2) != tensorInput.size(2) or self.tensorGrid.size(3) != tensorInput.size(3):
					torchHorizontal = torch.linspace(-1.0, 1.0, tensorInput.size(3)).view(1, 1, 1, tensorInput.size(3)).expand(tensorInput.size(0), 1, tensorInput.size(2), tensorInput.size(3))
					torchVertical = torch.linspace(-1.0, 1.0, tensorInput.size(2)).view(1, 1, tensorInput.size(2), 1).expand(tensorInput.size(0), 1, tensorInput.size(2), tensorInput.size(3))

					self.tensorGrid = torch.cat([ torchHorizontal, torchVertical ], 1).cuda()
				# end

				tensorFlow = torch.cat([ tensorFlow[:, 0:1, :, :] / ((tensorInput.size(3) - 1.0) / 2.0), tensorFlow[:, 1:2, :, :] / ((tensorInput.size(2) - 1.0) / 2.0) ], 1)

				return torch.nn.functional.grid_sample(input=tensorInput, grid=(self.tensorGrid + tensorFlow).permute(0, 2, 3, 1), mode='bilinear', padding_mode='border')
			# end
		# end

		self.modulePreprocess = Preprocess()

		self.moduleBasic = torch.nn.ModuleList([ Basic(intLevel) for intLevel in range(6) ])

		self.moduleBackward = Backward()
	# end

	def forward(self, tensorFirst, tensorSecond):
		tensorFlow = []

		tensorFirst = [ self.modulePreprocess(tensorFirst) ]
		tensorSecond = [ self.modulePreprocess(tensorSecond) ]

		for intLevel in range(5):
			if tensorFirst[0].size(2) > 32 or tensorFirst[0].size(3) > 32:
				tensorFirst.insert(0, torch.nn.functional.avg_pool2d(input=tensorFirst[0], kernel_size=2, stride=2))
				tensorSecond.insert(0, torch.nn.functional.avg_pool2d(input=tensorSecond[0], kernel_size=2, stride=2))
			# end
		# end

		tensorFlow = torch.FloatTensor(tensorFirst[0].size(0), 2, int(math.floor(tensorFirst[0].size(2) / 2.0)), int(math.floor(tensorFirst[0].size(3) / 2.0))).zero_().cuda()

		for intLevel in range(len(tensorFirst)):
			tensorUpsampled = torch.nn.functional.upsample(input=tensorFlow, scale_factor=2, mode='bilinear', align_corners=False) * 2.0 # align_corners might need to be set to True to mimic the original implementation

			if tensorUpsampled.size(2) != tensorFirst[intLevel].size(2): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 0, 0, 1 ], mode='replicate')
			if tensorUpsampled.size(3) != tensorFirst[intLevel].size(3): tensorUpsampled = torch.nn.functional.pad(input=tensorUpsampled, pad=[ 0, 1, 0, 0 ], mode='replicate')

			tensorFlow = self.moduleBasic[intLevel](torch.cat([ tensorFirst[intLevel], self.moduleBackward(tensorSecond[intLevel], tensorUpsampled), tensorUpsampled ], 1)) + tensorUpsampled
		# end

		return tensorFlow
	# end
# end

moduleNetwork = Network().cuda()

##########################################################

def estimate(tensorInputFirst, tensorInputSecond):
	tensorOutput = torch.FloatTensor()

	assert(tensorInputFirst.size(1) == tensorInputSecond.size(1))
	assert(tensorInputFirst.size(2) == tensorInputSecond.size(2))

	intWidth = tensorInputFirst.size(2)
	intHeight = tensorInputFirst.size(1)

	#assert(intWidth == 640) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue
	#assert(intHeight == 480) # remember that there is no guarantee for correctness, comment this line out if you acknowledge this and want to continue

	if True:
		tensorInputFirst = tensorInputFirst.cuda()
		tensorInputSecond = tensorInputSecond.cuda()
		tensorOutput = tensorOutput.cuda()
	# end

	if True:
		tensorPreprocessedFirst = tensorInputFirst.view(1, 3, intHeight, intWidth)
		tensorPreprocessedSecond = tensorInputSecond.view(1, 3, intHeight, intWidth)

		tensorOutput.resize_(2, intHeight, intWidth).copy_(moduleNetwork(tensorPreprocessedFirst, tensorPreprocessedSecond)[0, :, :, :])
	# end

	if True:
		tensorInputFirst = tensorInputFirst.cpu()
		tensorInputSecond = tensorInputSecond.cpu()
		tensorOutput = tensorOutput.cpu()
	# end

	return tensorOutput
# end

##########################################################
def get_flow(arguments_strFirst,arguments_strSecond,arguments_strOut,arguments_strModel = "F"):
	tensorInputFirst = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strFirst))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)
	tensorInputSecond = torch.FloatTensor(numpy.array(PIL.Image.open(arguments_strSecond))[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32) / 255.0)

	tensorOutput = estimate(tensorInputFirst, tensorInputSecond)

	objectOutput = open(arguments_strOut, 'wb')

	numpy.array([ 80, 73, 69, 72 ], numpy.uint8).tofile(objectOutput)
	numpy.array([ tensorOutput.size(2), tensorOutput.size(1) ], numpy.int32).tofile(objectOutput)
	numpy.array(tensorOutput.permute(1, 2, 0), numpy.float32).tofile(objectOutput)

	objectOutput.close()
# end
