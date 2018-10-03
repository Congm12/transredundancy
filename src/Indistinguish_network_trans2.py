#!/bin/python

import sys
import pickle
import numpy as np
import copy
import torch
from torch.utils.data import TensorDataset, DataLoader

def ReadPickleFeatures(filename, normalize_obs = True):
	# pickle load
	[AllMeta, AllBinObs, AllBinT1, AllBinT2, AllError] = pickle.load(open(filename, 'rb'))
	assert(len(AllBinObs) == len(AllBinT1) and len(AllBinT1) == len(AllBinT2) and len(AllBinT2) == len(AllError))
	# normalize obs if indicated, record the total number of reads per gene per sample
	AllNumReads = np.array([np.sum(x) for x in AllBinObs])
	assert(len(AllNumReads) == len(AllBinObs))
	AllNumReads.resize((len(AllBinObs), 1))
	if normalize_obs:
		for i in range(len(AllBinObs)):
			if AllNumReads[i] != 0:
				AllBinObs[i] = AllBinObs[i] / AllNumReads[i]
	# concatenate obs, theo T1, theo T2 to features
	Features = np.zeros((len(AllBinObs), len(AllBinObs[0])+len(AllBinT1[0])+len(AllBinT2[0])))
	for i in range(len(AllBinObs)):
		Features[i] = np.vstack((AllBinObs[i], AllBinT1[i], AllBinT2[i])).transpose().flatten()
	if normalize_obs:
		Features = np.hstack((AllNumReads, Features))
	# labels (continuous value)
	Labels = np.array(AllError)
	return [Features, Labels]


def GenerateNet(input_size, hidden_size, output_size, num_bn_layers=2):
	widths = [input_size] + hidden_size + [output_size]
	layers = []
	for i in range(len(widths)-1):
		layers.append(torch.nn.Linear(widths[i], widths[i+1]))
		if i < num_bn_layers:
			layers.append(torch.nn.BatchNorm1d(widths[i+1]))
		if i != len(widths)-1:
			layers.append(torch.nn.LeakyReLU())
	net = torch.nn.Sequential(*layers)
	return net


class SimpleCNN(torch.nn.Module):
	def __init__(self):
		super(SimpleCNN, self).__init__()
		#Input length = 10
		#Input channels = 2, output channels = 4
		self.conv1 = torch.nn.Conv1d(2, 20, kernel_size=5, stride=5, padding=0)
		self.conv2 = torch.nn.Conv1d(20, 40, kernel_size=2, stride=1, padding=0)
		self.fc1 = torch.nn.Linear(40 * 1, 1)
	def forward(self, x):
		 x = torch.nn.LeakyReLU().forward(self.conv1(x))
		 x = torch.nn.LeakyReLU().forward(self.conv2(x))
		 x = x.view(-1, 40 * 1)
		 x = self.fc1(x)
		 return x


def TrainNet(net, criterion, optimizer, trainX, trainY, _nepochs = 20, _batch_size = 100):
	train_dataset = torch.utils.data.TensorDataset(trainX, trainY)
	train_data_loader = DataLoader(train_dataset, batch_size=_batch_size, shuffle=True)
	for e in range(_nepochs):
		epoch_loss = []
		# start training batches
		for batch_idx, batch_data in enumerate(train_data_loader):
			batch_X, batch_Y = batch_data
			optimizer.zero_grad()
			train_output = net.forward(batch_X)
			loss = criterion(train_output, batch_Y)
			loss.backward()
			optimizer.step()
			epoch_loss.append( loss.item() )
		print([e, np.mean(epoch_loss)])
	return net


def GetFeature(Freq, Features):
	# select overlapping genes between Freq and Features
	overlapping = set(Freq.keys()) & set(Features.keys())
	overlapping = list(overlapping)
	# compile to scikit learn form
	X = np.zeros((len(overlapping), list(Features.values())[0].shape[0]))
	Y = np.zeros(len(overlapping))
	for i in range(len(overlapping)):
		g = overlapping[i]
		X[i,:] = Features[g]
		Y[i] = Freq[g]
	return [X, Y]


def GetFeature2(Freq, Features, Jaccard):
	# select overlapping genes between Freq and Features
	overlapping = set(Freq.keys()) & set(Features.keys()) & set(Jaccard.keys())
	overlapping = list(overlapping)
	# compile to scikit learn form
	X = np.zeros((len(overlapping), list(Features.values())[0].shape[0]))
	Y = np.zeros(len(overlapping))
	Jac = np.zeros((len(overlapping),1))
	for i in range(len(overlapping)):
		g = overlapping[i]
		X[i,:] = Features[g]
		Y[i] = Freq[g]
		Jac[i,0] = Jaccard[g]
	X = np.hstack((X, Jac))
	return [X, Y]


def GetConvFeature(Freq, Features):
	# select overlapping genes between Freq and Features
	overlapping = set(Freq.keys()) & set(Features.keys()) & set(Jaccard.keys())
	overlapping = list(overlapping)
	# compile to channal * length format
	halflen = int( len(list(Features.values())[0]) / 2 )
	X = np.zeros((len(overlapping), 2, halflen))
	Y = np.zeros(len(overlapping))
	for i in range(len(overlapping)):
		g = overlapping[i]
		X[i,0,:] = Features[g][:halflen]
		X[i,1,:] = Features[g][:halflen]
		Y[i] = Freq[g]
	return [X, Y]


if __name__=="__main__":
	[Freq, Features, Jaccard] = pickle.load(open("train_data_pairsimu02.pickle", 'rb'))

	UseMLP = True
	UseConv = not UseMLP

	if UseMLP:
		[trainX, trainY] = GetFeature2(Freq, Features, Jaccard)
		# [trainX, trainY] = GetFeature(Freq, Features)
		trainY.resize((len(trainY),1))

		trainX = torch.FloatTensor(trainX)
		trainY = torch.FloatTensor(trainY)
		# 5-fold cross validation
		for i in range(5):
			index_train = np.concatenate([np.arange(int(j*n/5), int((j+1)*n/5)) for j in range(5) if j != i])
			index_pred = np.arange(int(i*n/5), int((i+1)*n/5))
			net = GenerateNet(41, [80, 80], 1, num_bn_layers = 2)
			# criterion = torch.nn.CrossEntropyLoss()
			criterion = torch.nn.MSELoss()
			optimizer = torch.optim.Adam(net.parameters())
			net = TrainNet(net, criterion, optimizer, trainX[index_train, :], trainY[index_train, :], _nepochs = 40)
			pred = net.forward(trainX[index_pred, :])
			print(scipy.stats.pearsonr(pred.detach().numpy(), trainY[index_pred, :]))

	if UseConv:
		[trainX, trainY] = GetConvFeature(Freq, Features)
		trainY.resize((len(trainY),1))

		trainX = torch.FloatTensor(trainX)
		trainY = torch.FloatTensor(trainY)
		net = SimpleCNN()
		criterion = torch.nn.MSELoss()
		optimizer = torch.optim.Adam(net.parameters())
		net = TrainNet(net, criterion, optimizer, trainX, trainY, _nepochs = 40)