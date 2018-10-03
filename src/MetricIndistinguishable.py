#!/bin/python

import sys
import numpy as np
import scipy.stats
import copy
from pathlib import Path
from TranscriptClass import *
from DistributionClass import *
import sklearn.linear_model
import pickle


def TrimTranscripts(Transcripts, fafile):
	Selection = []
	fp = open(fafile, 'r')
	for line in fp:
		if line[0] == '>':
			Selection.append(line.strip().split()[0][1:])
	fp.close()
	newTranscripts = {t:Transcripts[t] for t in Selection}
	return newTranscripts


def ReadSalmonCov(quantfile):
	Cov = {}
	fp = open(quantfile, 'r')
	linecount = 0
	for line in fp:
		linecount += 1
		if linecount == 1:
			continue
		strs = line.strip().split("\t")
		Cov[strs[0]] = float(strs[4]) / float(strs[1])
	fp.close()
	return Cov


def ReadTrueCov(trueexpfile, TransLength):
	Cov = {}
	fp = open(trueexpfile, 'r')
	linecount = 0
	for line in fp:
		linecount += 1
		if linecount == 1:
			continue
		strs = line.strip().split("\t")
		if strs[0] in TransLength:
			Cov[strs[0]] = float(strs[1]) / TransLength[strs[0]]
	fp.close()
	return Cov


def ReadSalmonBootstrap(bootstrapfile, TransLength):
	BootStrap = {}
	fp = open(bootstrapfile, 'r')
	TransList = fp.readline().strip().split("\t")
	fp.close()
	ExpMatrix = np.loadtxt(bootstrapfile, delimiter="\t", skiprows=1)
	assert(len(TransList) == ExpMatrix.shape[1])
	for i in range(len(TransList)):
		t = TransList[i]
		BootStrap[t] = ExpMatrix[:,i] / TransLength[t]
	return BootStrap


def IndisPairBootStrapFrequency(SalmonFolder, GeneTransMap, TransLength):
	p = Path(SalmonFolder)
	Count_allbootstrap = {g:0 for g in GeneTransMap.keys()}
	Count_indis = {g:0 for g in GeneTransMap.keys()}
	for IDpath in p.iterdir():
	# IDpath = p / "ERR030872_rev"
	# if True:
		if (IDpath / "quant.sf").exists() and (IDpath / "expression_truth.txt").exists() and (IDpath / "quant_bootstraps.tsv").exists():
			print(str(IDpath))
			BootStrap = ReadSalmonBootstrap(str(IDpath / "quant_bootstraps.tsv"), TransLength)
			TrueCov = ReadTrueCov(str(IDpath / "expression_truth.txt"), TransLength)
			for g,v in GeneTransMap.items():
				assert(len(v) == 2)
				if np.any(np.array([t not in BootStrap for t in v])) or np.any(np.array([t not in TrueCov for t in v])):
					continue
				v_true = np.array([TrueCov[t] for t in v])
				if v_true[0] > v_true[1]:
					Count_indis[g] += np.sum(BootStrap[v[0]] <= BootStrap[v[1]])
				elif v_true[0] < v_true[1]:
					Count_indis[g] += np.sum(BootStrap[v[0]] >= BootStrap[v[1]])
				else:
					Count_indis[g] += len(BootStrap[v[0]])
				Count_allbootstrap[g] += len(BootStrap[v[0]])
	BootStrapFreq = {g:1.0*v/Count_allbootstrap[g] for g,v in Count_indis.items() if Count_allbootstrap[g] != 0}
	return BootStrapFreq


def PairIsoformTrend(GeneTransMap, SalmonCov, TrueCov):
	GeneTrend = {}
	for g,v in GeneTransMap.items():
		assert(len(v) == 2)
		if np.any([t not in SalmonCov for t in v]) or np.any([t not in TrueCov for t in v]):
			continue
		v_salmon = np.array([SalmonCov[t] for t in v])
		v_true = np.array([TrueCov[t] for t in v])
		unique_salmon = np.unique(v_salmon)
		unique_true = np.unique(v_true)
		# if np.max(v_salmon) <= 1:
		# 	continue
		if len(unique_salmon) > 1 and len(unique_true) > 1:
			GeneTrend[g] = scipy.stats.spearmanr(v_salmon, v_true)[0]
		elif len(unique_salmon) > 1 and len(unique_true) == 1:
			GeneTrend[g] = -1
		else:
			GeneTrend[g] = 1
	return GeneTrend


def PairwiseJaccard(Transcripts):
	Jaccard = {}
	for g,v in GeneTransMap.items():
		assert(len(v) == 2)
		Jaccard[g] = JaccardDistance(Transcripts, v[0], v[1])
	return Jaccard


def PairwiseDistributionDistance(Features, GeneTransMap, dist):
	Distance = {}
	for g,v in GeneTransMap.items():
		assert(len(v) == 2)
		vec_cat = Features[g]
		assert(len(vec_cat)%2 == 0)
		halflen = int(len(vec_cat)/2)
		Distance[g] = dist(vec_cat[:halflen], vec_cat[halflen:])
	return Distance


def IndistinguishFrequency(SalmonFolder, GeneTransMap, TransLength):
	p = Path(SalmonFolder)
	Freq = {}
	for IDpath in p.iterdir():
		if (IDpath / "quant.sf").exists() and (IDpath / "expression_truth.txt").exists():
			print(str(IDpath))
			SalmonCov = ReadSalmonCov(str(IDpath / "quant.sf"))
			TrueCov = ReadTrueCov(str(IDpath / "expression_truth.txt"), TransLength)
			GeneTrend = PairIsoformTrend(GeneTransMap, SalmonCov, TrueCov)
			for g,v in GeneTrend.items():
				if g in Freq:
					Freq[g].append( v )
				else:
					Freq[g] = [v]
	Freq = {t:1.0*np.sum(np.array(v) < 0)/len(v) for t,v in Freq.items()}
	return Freq


def PrintCorrelation(Freq, Metric):
	x = []
	y = []
	for g,v in Freq.items():
		if g in Metric:
			x.append(v)
			y.append(Metric[g])
	x = np.array(x)
	y = np.array(y)
	print(scipy.stats.spearmanr(x,1-y))
	print(scipy.stats.pearsonr(x,1-y))


def Regression(Freq, Features, regressor):
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
	# perform regression with given regressor
	regressor.fit(X, Y)
	pred = regressor.predict(X)
	print(scipy.stats.spearmanr(Y, pred))
	print(scipy.stats.pearsonr(Y, pred))


if __name__=="__main__":
	if len(sys.argv) == 1:
		print("python3 MetricIndistinguishable.py <GTFfile> <TransSequenceFile> <SalmonFolder>")
	else:
		GTFfile = sys.argv[1]
		TransSequenceFile = sys.argv[2]
		SalmonFolder = sys.argv[3]

		ID = TransSequenceFile.split("/")[-1].split("_")[0]
		print(ID)

		Transcripts = ReadGTF(GTFfile)
		Transcripts = TrimTranscripts(Transcripts, TransSequenceFile)
		[GeneTransMap, TransGeneMap] = Map_Gene_Trans(Transcripts)
		TransLength = GetTransLength(Transcripts)

		Freq = IndistinguishFrequency(SalmonFolder, GeneTransMap, TransLength)
		Jaccard = PairwiseJaccard(Transcripts)
		print("Jaccard distance")
		PrintCorrelation(Freq, Jaccard)

		Features_TheoOnly = ReadExampleFeature_TheoOnly(SalmonFolder, Transcripts, GeneTransMap, 20)
		JSD = PairwiseDistributionDistance(Features_TheoOnly, GeneTransMap, JensenShannonDivergence)
		print("JSD")
		PrintCorrelation(Freq, JSD)
		L1 = PairwiseDistributionDistance(Features_TheoOnly, GeneTransMap, L1Divergence)
		print("L1")
		PrintCorrelation(Freq, L1)
		L2 = PairwiseDistributionDistance(Features_TheoOnly, GeneTransMap, L2Divergence)
		print("L2")
		PrintCorrelation(Freq, L2)
		print("linear model")
		ols = sklearn.linear_model.LinearRegression()
		Regression(Freq, Features_TheoOnly, ols)

		BootStrapFreq = IndisPairBootStrapFrequency(SalmonFolder, GeneTransMap, TransLength)
		PrintCorrelation(Freq, BootStrapFreq)

		pickle.dump([Freq, Features_TheoOnly, Jaccard], open("train_data_"+ID+".pickle", 'wb'))