#!/bin/python

import sys
import numpy as np
import scipy
import sklearn
import copy
import struct
from pathlib import Path
import pandas as pd
# from plotnine import *


class Transcript_t(object):
	def __init__(self, _TransID, _GeneID, _Chr, _Strand, _StartPos, _EndPos):
		self.TransID=_TransID
		self.GeneID=_GeneID
		self.Chr = _Chr
		self.Strand = _Strand
		self.StartPos = _StartPos
		self.EndPos = _EndPos
		self.Exons = [] # each exon is a tuple of two integers
	def __eq__(self, other):
		if isinstance(other, Transcript_t):
			return (self.Chr==other.Chr and self.Strand==other.Strand and len(self.Exons)==len(other.Exons) and \
				min([self.Exons[i]==other.Exons[i] for i in range(len(self.Exons))])!=0)
		return NotImplemented
	def __ne__(self, other):
		result=self.__eq__(other)
		if result is NotImplemented:
			return result
		return not result
	def __lt__(self, other):
		if isinstance(other, Transcript_t):
			if self.Chr!=other.Chr:
				return self.Chr<other.Chr
			elif self.StartPos!=other.StartPos:
				return self.StartPos<other.StartPos
			else:
				return self.EndPos<other.EndPos
		return NotImplemented
	def __gt__(self, other):
		if isinstance(other, Transcript_t):
			if self.Chr!=other.Chr:
				return self.Chr>other.Chr
			elif self.StartPos!=other.StartPos:
				return self.StartPos>other.StartPos
			else:
				return self.EndPos<other.EndPos
		return NotImplemented
	def __le__(self, other):
		result=self.__gt__(other)
		if result is NotImplemented:
			return result
		return not result
	def __ge__(self, other):
		result=self.__lt__(other)
		if result is NotImplemented:
			return result
		return not result


def GetFeature(line, key):
	s=line.index(key)
	t=line.index(";", s+1)
	return line[(s+len(key)+2):(t-1)]


def ReadGTF(gtffile):
	Transcripts={}
	strand=""
	fp=open(gtffile, 'r')
	tmptransname=""
	tmptranscript=None
	extraExons = []
	for line in fp:
		if line[0]=='#':
			continue
		strs=line.strip().split("\t")
		if strs[2]=="transcript":
			if tmptransname!="" and not (tmptranscript is None):
				Transcripts[tmptransname]=tmptranscript
			tmptransname=GetFeature(line, "transcript_id")
			tmpgeneid=GetFeature(line, "gene_id")
			tmptranscript=Transcript_t(tmptransname, tmpgeneid, strs[0], (strs[6]=="+"), int(strs[3])-1, int(strs[4]))
		elif strs[2]=="exon":
			thistransid=GetFeature(line, "transcript_id")
			if thistransid == tmptransname and not (tmptranscript is None):
				tmptranscript.Exons.append((int(strs[3])-1, int(strs[4])))
			else:
				extraExons.append([thistransid, int(strs[3])-1, int(strs[4])])
	if tmptransname!="" and not (tmptranscript is None):
		Transcripts[tmptransname]=tmptranscript
	for e in extraExons:
		assert(e[0] in Transcripts)
		Transcripts[e[0]].Exons.append((e[1],e[2]))
	for t in Transcripts:
		Transcripts[t].Exons.sort(key=lambda x:x[0])
		if not Transcripts[t].Strand:
			Transcripts[t].Exons = Transcripts[t].Exons[::-1]
	fp.close()
	return Transcripts


def Map_Gene_Trans(Transcripts):
	GeneTransMap={}
	TransGeneMap={}
	for v in Transcripts.values():
		TransGeneMap[v.TransID]=v.GeneID
		if v.GeneID in GeneTransMap:
			GeneTransMap[v.GeneID].append(v.TransID)
		else:
			GeneTransMap[v.GeneID]=[v.TransID]
	for g,v in GeneTransMap.items():
		sortedv = sorted(v)
		GeneTransMap[g] = sortedv
	return [GeneTransMap, TransGeneMap]


def ReadQuantInfo(quantfile):
	SalmonExp={}
	TransLength={}
	fp=open(quantfile, 'r')
	linecount=0
	for line in fp:
		linecount+=1
		if linecount==1:
			continue
		strs=line.strip().split("\t")
		SalmonExp[strs[0]] = float(strs[4])
		TransLength[strs[0]] = int(strs[1])
	fp.close()
	return [SalmonExp, TransLength]


def ReadLPAdjustedQuant(salmonquant, lpquant):
	LPExp = {}
	fp = open(salmonquant, 'r')
	linecount = 0
	for line in fp:
		linecount += 1
		if linecount == 1:
			continue
		strs = line.strip().split("\t")
		LPExp[strs[0]] = float(strs[4])
	fp.close()
	fp = open(lpquant, 'r')
	for line in fp:
		if line[0] == '#':
			continue
		strs = line.strip().split("\t")
		LPExp[strs[0]] = float(strs[2])
	fp.close()
	return LPExp


def ReadGroundTruthQuant(exptruefile):
	TrueExp = {}
	fp = open(exptruefile, 'r')
	linecount = 0
	for line in fp:
		linecount += 1
		if linecount == 1:
			continue
		strs = line.strip().split("\t")
		TrueExp[strs[0]] = float(strs[1])
	fp.close()
	return TrueExp


def NonErrorGene(TrueExp, QuantExp, GeneTransMap, minnumreads = 200):
	GeneList = []
	for g,v in GeneTransMap.items():
		trans_true = np.array([TrueExp[t] for t in v if t in TrueExp])
		trans_quant = np.array([QuantExp[t] for t in v if t in QuantExp])
		if len(trans_true) == len(trans_quant) and len(trans_true) > 1 and np.sum(trans_true) > minnumreads and np.abs(np.sum(trans_true) - np.sum(trans_quant)) < 10:
			GeneList.append(g)
	return GeneList


def SummaryDiffMatrix(AllSamples, AllLPExps, AllTrueExps, AllNEGeneLists, GeneTransMap, g):
	nrow = 0 # number of samples with this gene as non-glevel-error gene
	ncol = 0 # number of transcripts expressed
	trans = []
	sampleindexes = []
	for i in range(len(AllNEGeneLists)):
		L = AllNEGeneLists[i]
		if g in L:
			nrow += 1
			sampleindexes.append(i)
			if ncol == 0:
				LPExp = AllLPExps[i]
				trans = [t for t in GeneTransMap[g] if t in LPExp]
				ncol = len(trans)
	diffmat = np.zeros((nrow, ncol))
	expmat = np.zeros((nrow, ncol))
	for i in range(len(sampleindexes)):
		for j in range(len(trans)):
			ind = sampleindexes[i]
			diffmat[i,j] = AllLPExps[ind][trans[j]] - AllTrueExps[ind][trans[j]]
			expmat[i,j] = AllLPExps[ind][trans[j]]
	# normalize the diff of each sample to be located within -1 and 1
	# diffmat = diffmat / np.max(np.abs(diffmat), axis = 1)[:,None]
	# diffmat = diffmat / np.sum(np.abs(diffmat), axis = 1)[:,None]
	diffmat = diffmat / np.sum(expmat, axis = 1)[:,None]
	print(np.sum(expmat, axis = 1))
	return [[AllSamples[ind] for ind in sampleindexes], diffmat]


def TrimTranscripts(Transcripts, SalmonExp):
	newTranscripts = {}
	for tname,t in Transcripts.items():
		if tname in SalmonExp:
			newTranscripts[tname] = t
	return newTranscripts


def ReadRawCorrection(filename):
	Expected={}
	fp=open(filename, 'rb')
	numtrans=struct.unpack('i', fp.read(4))[0]
	for i in range(numtrans):
		namelen=struct.unpack('i', fp.read(4))[0]
		seqlen=struct.unpack('i', fp.read(4))[0]
		name=""
		correction=np.zeros(seqlen)
		for j in range(namelen):
			name+=struct.unpack('c', fp.read(1))[0].decode('utf-8')
		for j in range(seqlen):
			correction[j]=struct.unpack('d', fp.read(8))[0]
		Expected[name] = correction
	fp.close()
	print("Finish reading theoretical distributions for {} transcripts.".format(len(Expected)))
	return Expected


def ReadRawStartPos(filename, TransLength):
	Observed={}
	fp=open(filename, 'rb')
	numtrans=struct.unpack('i', fp.read(4))[0]
	for i in range(numtrans):
		namelen=struct.unpack('i', fp.read(4))[0]
		seqlen=struct.unpack('i', fp.read(4))[0]
		name=""
		poses=np.zeros(seqlen, dtype=np.int)
		counts=np.zeros(seqlen)
		for j in range(namelen):
			name+=struct.unpack('c', fp.read(1))[0].decode('utf-8')
		for j in range(seqlen):
			poses[j]=struct.unpack('i', fp.read(4))[0]
		for j in range(seqlen):
			counts[j]=struct.unpack('d', fp.read(8))[0]
		tmp=np.zeros(poses[-1]+1)
		for j in range(len(poses)):
			tmp[poses[j]] = counts[j]
		Observed[name]=tmp
	fp.close()
	for k,v in TransLength.items():
		if not (k in Observed):
			Observed[k] = [0]*v
	print("Finish reading actual distribution for {} transcripts.".format(len(Observed)))
	return Observed


def ReadBootStrapping(filename):
	matrix = np.loadtxt(filename, skiprows=1)
	fp = open(filename, 'r')
	TransList = fp.readline().strip().split("\t")
	BootStrap = {}
	for i in range(len(TransList)):
		t = TransList[i]
		BootStrap[t] = matrix[:,i]
	fp.close()
	return BootStrap


def BootStrapPrediction(BootStrap, SalmonExp, PairedTrans):
	Pred = np.zeros(len(PairedTrans))
	for i in range(len(PairedTrans)):
		(t1, t2) = PairedTrans[i]
		assert(t1 in BootStrap)
		assert(t2 in BootStrap)
		delta1 = BootStrap[t1] - SalmonExp[t1]
		delta2 = BootStrap[t2] - SalmonExp[t2]
		denominator = (np.abs(delta1) + np.abs(delta2))
		predvector = np.abs(delta1 + delta2) / (np.abs(delta1) + np.abs(delta2))
		predvector[np.where(denominator==0)] = 1
		Pred[i] = np.mean(predvector)
	return Pred


def GetSharedExon(Transcripts, tnames):
	assert(len(tnames) != 0)
	Chr = Transcripts[tnames[0]].Chr
	Strand = Transcripts[tnames[0]].Strand
	for t in tnames:
		assert(Chr == Transcripts[t].Chr)
		assert(Strand == Transcripts[t].Strand)
	allexons = []
	for t in tnames:
		allexons += Transcripts[t].Exons
	allexons.sort(key=lambda x:x[0])
	uniqexons = []
	for e in allexons:
		if len(uniqexons) == 0 or uniqexons[-1][1] < e[0]:
			uniqexons.append([e[0], e[1]])
		else:
			uniqexons[-1][1] = max(uniqexons[-1][1], e[1])
	if not Strand:
		uniqexons = uniqexons[::-1]
	return uniqexons


def BinningDistribution(exp, nposbin):
	length = len(exp)
	newexp = np.zeros(nposbin)
	for i in range(nposbin):
		newexp[i] = np.sum(exp[int(length*i/nposbin):int(length*(i+1)/nposbin)])
	return newexp


def PaddingDistribution(exp, Transcripts, sharedexons, t):
	if not Transcripts[t].Strand:
		for i  in range(1, len(sharedexons)):
			assert(sharedexons[i][0] < sharedexons[i-1][0])
	exons = Transcripts[t].Exons
	newexp = np.zeros(np.sum([e[1]-e[0] for e in sharedexons]))
	pos_ori = 0
	pos_new = 0
	idx_ori = 0
	idx_new = 0
	if Transcripts[t].Strand:
		for idx_new in range(len(sharedexons)):
			while idx_ori < len(exons) and exons[idx_ori][1] < sharedexons[idx_new][0]:
				pos_ori += exons[idx_ori][1] - exons[idx_ori][0]
				idx_ori += 1
			while idx_ori < len(exons) and max(exons[idx_ori][0], sharedexons[idx_new][0]) <= min(exons[idx_ori][1], sharedexons[idx_new][1]):
				overlap_start = max(exons[idx_ori][0], sharedexons[idx_new][0])
				overlap_end = min(exons[idx_ori][1], sharedexons[idx_new][1])
				newexp[(pos_new+overlap_start-sharedexons[idx_new][0]):(pos_new+overlap_end-sharedexons[idx_new][0])] = exp[(pos_ori+overlap_start-exons[idx_ori][0]):(pos_ori+overlap_end-exons[idx_ori][0])]
				pos_ori += exons[idx_ori][1] - exons[idx_ori][0]
				idx_ori += 1
			pos_new += sharedexons[idx_new][1] - sharedexons[idx_new][0]
	else:
		for idx_new in range(len(sharedexons)):
			while idx_ori < len(exons) and exons[idx_ori][0] > sharedexons[idx_new][1]:
				pos_ori += exons[idx_ori][1] - exons[idx_ori][0]
				idx_ori += 1
			while idx_ori < len(exons) and max(exons[idx_ori][0], sharedexons[idx_new][0]) <= min(exons[idx_ori][1], sharedexons[idx_new][1]):
				overlap_start = max(exons[idx_ori][0], sharedexons[idx_new][0])
				overlap_end = min(exons[idx_ori][1], sharedexons[idx_new][1])
				newexp[(pos_new+sharedexons[idx_new][1]-overlap_end):(pos_new+sharedexons[idx_new][1]-overlap_start)] = exp[(pos_ori+exons[idx_ori][1]-overlap_end):(pos_ori+exons[idx_ori][1]-overlap_start)]
				pos_ori += exons[idx_ori][1] - exons[idx_ori][0]
				idx_ori += 1
			pos_new += sharedexons[idx_new][1] - sharedexons[idx_new][0]
	return newexp


def GenerateFeatureLabel(Expected, ExpDiff, Transcripts, SalmonExp, TransLength, GeneTransMap, TransGeneMap):
	PairedTrans = []
	Feature_Diff = []
	TargetValue = []
	for g,v in GeneTransMap.items():
		indexes = [i for i in range(len(v)) if SalmonExp[v[i]]/TransLength[v[i]] > 0.1 or (SalmonExp[v[i]]+ExpDiff[v[i]])/TransLength[v[i]] > 0.1]
		rest = [i for i in range(len(v)) if not (i in indexes)]
		if len(indexes) == 0:
			continue
		for i in indexes:
			for j in rest:
				sharedexons = GetSharedExon(Transcripts, [v[i], v[j]])
				exp1 = PaddingDistribution(Expected[v[i]], Transcripts, sharedexons, v[i])
				exp2 = PaddingDistribution(Expected[v[j]], Transcripts, sharedexons, v[j])
				assert(len(exp1) == len(exp2))
				assert(np.sum(exp1) != 0)
				assert(np.sum(exp2) != 0)
				exp1 /= np.sum(exp1)
				exp2 /= np.sum(exp2)
				sharedlength = np.sum(np.logical_and(exp1 > 0, exp2 > 0))
				if TransLength[v[i]] < TransLength[v[j]]:
					PairedTrans.append( (v[i],v[j]) )
					Feature_Diff.append( np.append(exp1-exp2, sharedlength/len(exp1)) )
					if (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])) == 0:
						TargetValue.append(1)
					else:
						TargetValue.append(abs(ExpDiff[v[i]]+ExpDiff[v[j]]) / (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])))
				else:
					PairedTrans.append( (v[j],v[i]) )
					Feature_Diff.append( np.append(exp2-exp1, sharedlength/len(exp1)) )
					if (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])) == 0:
						TargetValue.append(1)
					else:
						TargetValue.append(abs(ExpDiff[v[i]]+ExpDiff[v[j]]) / (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])))
		for tmpi in range(len(indexes)):
			for tmpj in range(tmpi+1, len(indexes)):
				i = indexes[tmpi]
				j = indexes[tmpj]
				sharedexons = GetSharedExon(Transcripts, [v[i], v[j]])
				exp1 = PaddingDistribution(Expected[v[i]], Transcripts, sharedexons, v[i])
				exp2 = PaddingDistribution(Expected[v[j]], Transcripts, sharedexons, v[j])
				assert(len(exp1) == len(exp2))
				assert(np.sum(exp1) != 0)
				assert(np.sum(exp2) != 0)
				exp1 /= np.sum(exp1)
				exp2 /= np.sum(exp2)
				sharedlength = np.sum(np.logical_and(exp1 > 0, exp2 > 0))
				if TransLength[v[i]] < TransLength[v[j]]:
					PairedTrans.append( (v[i],v[j]) )
					Feature_Diff.append( np.append(exp1-exp2, sharedlength/len(exp1)) )
					if (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])) == 0:
						TargetValue.append(1)
					else:
						TargetValue.append(abs(ExpDiff[v[i]]+ExpDiff[v[j]]) / (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])))
				else:
					PairedTrans.append( (v[j],v[i]) )
					Feature_Diff.append( np.append(exp2-exp1, sharedlength/len(exp1)) )
					if (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])) == 0:
						TargetValue.append(1)
					else:
						TargetValue.append(abs(ExpDiff[v[i]]+ExpDiff[v[j]]) / (abs(ExpDiff[v[i]]) + abs(ExpDiff[v[j]])))
	assert(len(PairedTrans) == len(TargetValue))
	assert(len(TargetValue) == len(Feature_Diff))
	return [PairedTrans, Feature_Diff, TargetValue]


def BinningFeatures(PairedTrans, Feature_Diff, TargetValue, lenBounds, len_class):
	len_start = 0
	len_end = lenBounds[len_class]
	if len_class != 0:
		len_start = lenBounds[len_class-1]
	# choose number of bins by median length / 25
	medianlength = np.median([len(x) for x in Feature_Diff if len(x) > len_start and len(x) <= len_end])
	nposbin = int(round((medianlength-1)/25))
	# create new variables for storing result
	binPairedTrans = []
	binFeature_Diff = []
	binTargetValue = []
	for i in range(len(PairedTrans)):
		if len(Feature_Diff[i]) > len_start and len(Feature_Diff[i]) <= len_end:
			binPairedTrans.append( PairedTrans[i] )
			binTargetValue.append( TargetValue[i] )
			newdiff = np.zeros(nposbin)
			olddiff = Feature_Diff[i][:-1]
			oldlen = len(olddiff)
			for j in range(nposbin):
				newdiff[j] = np.sum(olddiff[int(round(oldlen/nposbin*j)):int(round(oldlen/nposbin*(j+1)))])
			newdiff = np.append(newdiff, Feature_Diff[i][-1])
			binFeature_Diff.append( newdiff )
	binFeature_Diff = np.array(binFeature_Diff)
	binTargetValue = np.array(binTargetValue)
	return [binPairedTrans, binFeature_Diff, binTargetValue]


def CombineExons(Transcripts, v):
	allexons = []
	chr = Transcripts[v[0]].Chr
	strand = Transcripts[v[0]].Strand
	for t in v:
		assert(chr == Transcripts[t].Chr and strand == Transcripts[t].Strand)
		allexons += Transcripts[t].Exons
	allexons.sort(key=lambda x:x[0])
	combinedexons = []
	for e in allexons:
		if len(combinedexons) == 0 or combinedexons[-1][1] <= e[0]:
			combinedexons.append([e[0], e[1]])
		else:
			combinedexons[-1][1] = max(combinedexons[-1][1], e[1])
	if not strand:
		combinedexons = combinedexons[::-1]
	combinedexons = [(e[0],e[1]) for e in combinedexons]
	return combinedexons


def ConvertCoordinate(Dist, Transcripts, combinedexons, v):
	newlen = np.sum([e[1]-e[0] for e in combinedexons])
	outDist = {}
	for t in v:
		newdist = np.zeros(newlen)
		olddist = Dist[t]
		oldexons = Transcripts[t].Exons
		newpos = 0
		oldpos = 0
		newexonidx = 0
		oldexonidx = 0
		if Transcripts[t].Strand:
			while oldexonidx < len(oldexons):
				# if combined exons are totally outside of this exon
				while combinedexons[newexonidx][1] < oldexons[oldexonidx][0]:
					newpos += combinedexons[newexonidx][1] - combinedexons[newexonidx][0]
					newexonidx += 1
					assert(newexonidx < len(combinedexons))
				# assert that the old exon should be totally within the combined exon
				assert(oldexons[oldexonidx][0] >= combinedexons[newexonidx][0] and oldexons[oldexonidx][1] <= combinedexons[newexonidx][1])
				newdist[(newpos+oldexons[oldexonidx][0]-combinedexons[newexonidx][0]):(newpos+oldexons[oldexonidx][1]-combinedexons[newexonidx][0])] = olddist[oldpos:(oldpos+oldexons[oldexonidx][1]-oldexons[oldexonidx][0])]
				oldpos += oldexons[oldexonidx][1] - oldexons[oldexonidx][0]
				oldexonidx += 1
		else:
			while oldexonidx < len(oldexons):
				# if combined exons are totally outside of this exon
				while combinedexons[newexonidx][0] > oldexons[oldexonidx][1]:
					newpos += combinedexons[newexonidx][1] - combinedexons[newexonidx][0]
					newexonidx += 1
					assert(newexonidx < len(combinedexons))
				# assert that the old exon should be totally within the combined exon
				assert(oldexons[oldexonidx][0] >= combinedexons[newexonidx][0] and oldexons[oldexonidx][1] <= combinedexons[newexonidx][1])
				newdist[(newpos+combinedexons[newexonidx][1]-oldexons[oldexonidx][1]):(newpos+combinedexons[newexonidx][1]-oldexons[oldexonidx][0])] = olddist[oldpos:(oldpos+oldexons[oldexonidx][1]-oldexons[oldexonidx][0])]
				oldpos += oldexons[oldexonidx][1] - oldexons[oldexonidx][0]
				oldexonidx += 1
		outDist[t] = newdist
	return outDist


def TraditionalPairwisePredictor(argv):
	if len(argv) == 1:
		print("python3 <GTFfile> <Folder> <SalmonQuantFile> <CorrectionFile> <ExpressionDiffFile>")
	else:
		GTFfile = argv[1]
		Folder = argv[2]
		SalmonQuantFile = argv[3]
		CorrectionFile = argv[4]
		ExpressionDiffFile = argv[5]

		Transcripts = ReadGTF(GTFfile)
		[TransList, SalmonExp, TPM, TransLength] = ReadQuantInfo(SalmonQuantFile)
		Transcripts = TrimTranscripts(Transcripts, SalmonExp)
		[GeneTransMap, TransGeneMap] = Map_Gene_Trans(Transcripts)
		Expected = ReadRawCorrection(CorrectionFile, TransList, TransLength)
		ExpDiff = ReadExpressionDiff(Folder, ExpressionDiffFile)
		[PairedTrans, Feature_Diff, TargetValue] = GenerateFeatureLabel(Expected, ExpDiff, Transcripts, SalmonExp, TransLength, GeneTransMap, TransGeneMap)

		lenBounds = np.percentile([len(x) for x in Feature_Diff], [1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100])
		from sklearn.svm import SVR
		for len_class in range(len(lenBounds)):
			[binPairedTrans, binFeature_Diff, binTargetValue] = BinningFeatures(PairedTrans, Feature_Diff, TargetValue, lenBounds, len_class)
			indexes = np.argsort(binTargetValue)
			binTargetValue = binTargetValue[indexes]
			binPairedTrans = [binPairedTrans[i] for i in indexes]
			binFeature_Diff = binFeature_Diff[indexes]
			print(binFeature_Diff.shape)
			# SVR
			clf = SVR(C=1.0, epsilon=0.2)
			clf.fit(binFeature_Diff, binTargetValue)
			pred = clf.predict(binFeature_Diff)
			scipy.stats.pearsonr(pred, binTargetValue)
			scipy.stats.spearmanr(pred, binTargetValue)
			# boot strapping
			bootstrappred = BootStrapPrediction(BootStrap, SalmonExp, binPairedTrans)
			scipy.stats.pearsonr(bootstrappred, binTargetValue)
			scipy.stats.spearmanr(bootstrappred, binTargetValue)


if __name__=="__main__":
	if False:
		TraditionalPairwisePredictor(sys.argv)

	if False:
		Folder = "/home/cong/Documents/SADresult/SADstandardsimu"
		Type = ["PC"]
		IDs = ["GEU", "GM12878", "K562"]
		NumEvents = [200, 500, 1000, 1500]

		GTFfile = "/home/cong/Documents/SADresult/gencode.v26.annotation.pc.gtf"
		Transcripts = ReadGTF(GTFfile)
		[GeneTransMap, TransGeneMap] = Map_Gene_Trans(Transcripts)
		TransLength = {t:np.sum([e[1]-e[0] for e in Transcripts[t].Exons]) for t in Transcripts.keys()}

		AllSamples = []
		AllLPExps = []
		AllTrueExps = []
		AllNEGeneLists = []
		for t in Type:
			for i in IDs:
				for n in NumEvents:
					subfolder = Folder +"/"+ t +"_"+ i +"_"+ str(n) +"/"
					# LPExp = ReadLPAdjustedQuant(subfolder+"quant.sf", subfolder+"test_LP_refined")
					LPExp = ReadLPAdjustedQuant(subfolder+"quant.sf", subfolder+"test_correctapprox7/test_LP_refined")
					TrueExp = ReadGroundTruthQuant(subfolder+"expression_truth.txt")
					NEGeneList = NonErrorGene(LPExp, TrueExp, GeneTransMap)
					# add to variable
					AllSamples.append( t +"_"+ i +"_"+ str(n) )
					AllLPExps.append(LPExp)
					AllTrueExps.append(TrueExp)
					AllNEGeneLists.append(NEGeneList)

		AllNEGeneSets = [set(x) for x in AllNEGeneLists]
		UnionNEGenes = set().union(*AllNEGeneLists)
		NTransNEGenes = set([g for g in UnionNEGenes if len(GeneTransMap[g]) == 2])
		UnionTrans = set(sum([v for g,v in GeneTransMap.items() if g in NTransNEGenes], []))
		# read observed and theoretical distribution
		AllObserved = []
		AllTheoretical = []
		for i in range(len(AllSamples)):
			s = AllSamples[i]
			subfolder = Folder +"/"+ s
			tmp_observed = ReadRawStartPos(subfolder +"/startpos.dat", TransLength)
			AllObserved.append( {t:v for t,v in tmp_observed.items() if t in UnionTrans} )
			tmp_theoretical = ReadRawCorrection(subfolder +"/correction.dat")
			AllTheoretical.append( {t:v for t,v in tmp_theoretical.items() if t in UnionTrans} )

		# select genes with * transcripts, and process the feature vector
		nbins = 10
		for ntrans in range(2,3):
			NTransNEGenes = [g for g in UnionNEGenes if len(GeneTransMap[g]) == ntrans]
			AllBinObs = []
			AllBinT1 = []
			AllBinT2 = []
			AllError = []
			AllMeta = []
			for g in NTransNEGenes:
				v = GeneTransMap[g]
				combinedexons = CombineExons(Transcripts, v)
				for i in range(len(AllSamples)):
					s = AllSamples[i]
					if np.any([t not in AllLPExps[i] for t in v]):
						continue
					if np.abs(AllTrueExps[i][v[0]] + AllTrueExps[i][v[1]] - AllLPExps[i][v[0]] - AllLPExps[i][v[1]]) > 10:
						continue
					converted_obs = ConvertCoordinate(AllObserved[i], Transcripts, combinedexons, v)
					converted_theo = ConvertCoordinate(AllTheoretical[i], Transcripts, combinedexons, v)
					converted_obs = {t:BinningDistribution(converted_obs[t], nbins) for t in converted_obs.keys()}
					converted_theo = {t:BinningDistribution(converted_theo[t], nbins) for t in converted_theo.keys()}
					converted_theo = {t:v/np.maximum(np.sum(v),0.1) for t,v in converted_theo.items()}
					AllBinObs.append( converted_obs[v[0]]+converted_obs[v[1]] )
					AllBinT1.append(converted_theo[v[0]])
					AllBinT2.append(converted_theo[v[1]])
					sumquant = AllTrueExps[i][v[0]] + AllTrueExps[i][v[1]]
					AllError.append([(AllLPExps[i][v[0]]-AllTrueExps[i][v[0]])/sumquant, (AllLPExps[i][v[1]]-AllTrueExps[i][v[1]])/sumquant])
					AllMeta.append([AllSamples[i], g, v[0], v[1]])

		pickle.dump([AllMeta, AllBinObs, AllBinT1, AllBinT2, AllError], open("indisfeature_2trans.pickle", 'wb'))


		tmpList = ['ENSG00000091157.13', 'ENSG00000004534.14']
		# strange ones: ENSG00000213533.11, ENSG00000182481.8
		g = tmpList[0]
		InvolvedSamples, DiffMatrix = SummaryDiffMatrix(AllSamples, AllLPExps, AllTrueExps, AllNEGeneLists, GeneTransMap, g)
		df = pd.DataFrame(DiffMatrix, columns=GeneTransMap[g], index=InvolvedSamples)
		df = df.reset_index().melt(id_vars=["index"])
		df.columns=["ID", "Transcript", "NormalizedError"]
		# ggplot(df) + geom_point(aes(x="ID", y="NormalizedError", color="Transcript")) + theme(axis_text_x=element_text(rotation=90, hjust=1))
