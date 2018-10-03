#!/bin/python

import numpy as np
import struct
import gzip
from pathlib import Path
from TranscriptClass import *


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


def ConvertCoordinate(exp, Transcripts, sharedexons, t):
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


def BinningDistribution(exp, nposbin):
	length = len(exp)
	newexp = np.zeros(nposbin)
	for i in range(nposbin):
		newexp[i] = np.sum(exp[int(length*i/nposbin):int(length*(i+1)/nposbin)])
	return newexp


def ReadExampleFeature_TheoOnly(SalmonFolder, Transcripts, GeneTransMap, nposbin = 10):
	# read theoretical distribution from SINGLE salmon quant result
	p = Path(SalmonFolder)
	example_correction_file = str(list(p.glob("*/correction.dat"))[0])
	Theoretical = ReadRawCorrection(example_correction_file)
	for t,v in Theoretical.items():
		if np.sum(v) > 0:
			Theoretical[t] = v/np.sum(v)
	# compose feature for each gene: a concatenation of theoretical distribution of all isoforms
	Features = {}
	for g,v in GeneTransMap.items():
		if np.any(np.array([t not in Theoretical for t in v])):
			continue
		union_exons = GetSharedExon(Transcripts, v)
		NewDist = []
		for t in v:
			tmpdist = ConvertCoordinate(Theoretical[t], Transcripts, union_exons, t)
			NewDist.append( BinningDistribution(tmpdist, nposbin) )
		Features[g] = np.concatenate(NewDist)
	return Features


def JensenShannonDivergence(dist1, dist2):
	assert(len(dist1) == len(dist2))
	P=(dist1+1e-8)/np.sum(dist1+1e-8)
	Q=(dist2+1e-8)/np.sum(dist2+1e-8)
	M=(P+Q)/2
	Dpm=np.divide(P, M)
	Dpm=np.dot(P, np.log2(Dpm))
	Dqm=np.divide(Q, M)
	Dqm=np.dot(Q, np.log2(Dqm))
	JSD = (Dpm+Dqm)/2
	return JSD


def L1Divergence(dist1, dist2):
	assert(len(dist1) == len(dist2))
	P=dist1/np.sum(dist1)
	Q=dist2/np.sum(dist2)
	return np.sum(np.abs(P - Q))


def L2Divergence(dist1, dist2):
	assert(len(dist1) == len(dist2))
	P=dist1/np.sum(dist1)
	Q=dist2/np.sum(dist2)
	sss = (P - Q).dot(P - Q)
	return np.sqrt(sss)