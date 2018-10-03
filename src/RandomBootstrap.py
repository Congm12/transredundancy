#!/bin/python

import sys
import numpy as np


def ReadEqclassFile(eqclassfile):
	TransNames=[]
	TransIndex={}
	EqTrans=[]
	EqWeights=[]
	EqCounts=[]
	EqIndex={}
	fp=open(eqclassfile, 'r')
	numtrans=int(fp.readline().strip())
	numclass=int(fp.readline().strip())
	for i in range(numtrans):
		line=fp.readline().strip()
		TransNames.append(line)
		TransIndex[line] = i
	for i in range(numclass):
		line=fp.readline()
		strs=line.strip().split("\t")
		ntid=int(strs[0])
		assert(len(strs) == ntid*2+2)
		EqTrans.append([int(strs[i]) for i in range(1, 1+ntid)])
		EqWeights.append([float(strs[i]) for i in range(1+ntid, 1+2*ntid)])
		EqCounts.append(int(strs[-1]))
		EqIndex[tuple(EqTrans[-1])] = i
	fp.close()
	return [TransNames, TransIndex, EqTrans, EqWeights, EqCounts, EqIndex]


def ReadSalmonQuant(quantfile, TransIndex):
	ExpReads=np.zeros(len(TransIndex))
	EffLen=np.zeros(len(TransIndex))
	fp=open(quantfile, 'r')
	linecount=0
	for line in fp:
		linecount+=1
		if linecount==1:
			continue
		strs=line.strip().split("\t")
		ExpReads[TransIndex[strs[0]]] = float(strs[4])
		EffLen[TransIndex[strs[0]]] = float(strs[2])
	fp.close()
	return [ExpReads, EffLen]


def GetFeature(string, key):
	if key in string:
		s=string.index(key)
		t=string.index(";", s+1)
		return string[(s+len(key)+2):(t-1)]
	else:
		return ""


def Map_Gene_Trans(GTFfile):
	Trans2Gene={}
	Gene2Trans={}
	fp=open(GTFfile, 'r')
	for line in fp:
		if line[0]=='#':
			continue
		strs=line.strip().split("\t")
		if strs[2]=="transcript":
			TransID=GetFeature(line, "transcript_id")
			GeneID=GetFeature(line, "gene_id")
			Trans2Gene[TransID] = GeneID
			if GeneID in Gene2Trans:
				Gene2Trans[GeneID].append(TransID)
			else:
				Gene2Trans[GeneID]=[TransID]
	fp.close()
	return [Trans2Gene, Gene2Trans]


def TrimMap_Gene_Trans(Trans2Gene, Gene2Trans, TransIndex):
	newTrans2Gene={}
	newGene2Trans={}
	for t,g in Trans2Gene.items():
		if t in TransIndex:
			newTrans2Gene[t] = g
	for g,v in Gene2Trans.items():
		u = [v[i] for i in range(len(v)) if v[i] in TransIndex]
		newGene2Trans[g] = u
	return [newTrans2Gene, newGene2Trans]


def GeneLevelEqclass(Trans2Gene, TransNames, ExpReads, EqTrans, EqWeights, EqCounts):
	GeneEqTrans={}
	GeneEqCounts={}
	for i in range(len(EqTrans)):
		# process assignment
		tids=EqTrans[i]
		weights=EqWeights[i]
		denom = 0
		for j in range(len(tids)):
			denom += ExpReads[tids[j]]*weights[j]
		assign=[]
		for j in range(len(tids)):
			assign.append( EqCounts[i]*ExpReads[tids[j]]*weights[j]/denom )
		# separate the equivalent class into gene-specific classes
		genes=[Trans2Gene[TransNames[tids[j]]] for j in range(len(tids))]
		uniquegenes=list(set(genes))
		for g in uniquegenes:
			newtids=[tids[j] for j in range(len(tids)) if Trans2Gene[TransNames[tids[j]]]==g]
			newcount=sum( [assign[j] for j in range(len(tids)) if Trans2Gene[TransNames[tids[j]]]==g] )
			if g in GeneEqCounts:
				GeneEqTrans[g].append( newtids )
				GeneEqCounts[g].append( newcount )
			else:
				GeneEqTrans[g] = [newtids]
				GeneEqCounts[g] = [newcount]
	return [GeneEqTrans, GeneEqCounts]


def GeneLevelEMUpdate(g_EqTrans, g_EqCounts, tidlist, EffLen):
	alpha=np.ones(len(tidlist))
	TidMap={}
	Noise_EffLen=np.zeros(len(tidlist))
	for i in range(len(tidlist)):
		TidMap[tidlist[i]] = i
		while Noise_EffLen[i]<1:
			Noise_EffLen[i] = EffLen[tidlist[i]] + np.random.normal(loc=EffLen[tidlist[i]], scale=EffLen[tidlist[i]]/10)
	# initialize
	# for eqID in range(len(g_EqTrans)):
	# 	n = len(g_EqTrans[eqID])
	# 	c = g_EqCounts[eqID]
	# 	for tid in g_EqTrans[eqID]:
	# 		alpha[TidMap[tid]] += 1.0*c/n
	# EM update
	while True:
		alpha_out = np.zeros(len(tidlist))
		for eqID in range(len(g_EqTrans)):
			if g_EqCounts[eqID]<1e-4:
				continue
			# calculate denominator
			denom = 0
			denom = np.sum(alpha/Noise_EffLen)
			# assign reads
			alpha_out += g_EqCounts[eqID]/denom * alpha/Noise_EffLen
		sumdiff = np.sum(np.abs(alpha-alpha_out))
		alpha = alpha_out
		if sumdiff<1e-2:
			break
	return alpha


def BootstrapQuant(GeneEqTrans, GeneEqCounts, TransIndex, Gene2Trans, EffLen):
	count=0
	alpha_bootstrap=np.zeros((len(TransIndex),100))
	for g,v in Gene2Trans.items():
		count += 1
		if g not in GeneEqCounts:
			continue
		if sum(GeneEqCounts[g])<1:
			continue
		if len(v)==1:
			alpha_bootstrap[TransIndex[v[0]], :] = np.zeros(100)*sum(GeneEqCounts[g])
			continue
		print([count, g])
		tidlist = [TransIndex[v[i]] for i in range(len(v))]
		for r in range(100):
			alpha = GeneLevelEMUpdate(GeneEqTrans[g], GeneEqCounts[g], tidlist, EffLen)
			for i in range(len(alpha)):
				if alpha[i]<1e-4:
					alpha[i] = 0
			alpha_bootstrap[tidlist, r] = alpha
	return alpha_bootstrap


def WriteBootstrapping(TransNames, alpha_bootstrap, outputfile):
	fp=open(outputfile, 'w')
	for i in range(len(TransNames)):
		fp.write("{}\t{}\n".format(TransNames[i], "\t".join([str(x) for x in alpha_bootstrap[i,:]]) ))
	fp.close()


if __name__=="__main__":
	if len(sys.argv)==1:
		print("python3 RandomBootstrap.py <quant.sf> <eqclassfile> <gtffile> <outputfile>")
	else:
		QuantFile=sys.argv[1]
		EqclassFile=sys.argv[2]
		GTFfile=sys.argv[3]
		OutFile=sys.argv[4]

		[TransNames, TransIndex, EqTrans, EqWeights, EqCounts, EqIndex] = ReadEqclassFile(EqclassFile)
		[ExpReads, EffLen] = ReadSalmonQuant(QuantFile, TransIndex)
		[Trans2Gene, Gene2Trans] = Map_Gene_Trans(GTFfile)
		[Trans2Gene, Gene2Trans] = TrimMap_Gene_Trans(Trans2Gene, Gene2Trans, TransIndex)

		[GeneEqTrans, GeneEqCounts] = GeneLevelEqclass(Trans2Gene, TransNames, ExpReads, EqTrans, EqWeights, EqCounts)
		alpha_bootstrap = BootstrapQuant(GeneEqTrans, GeneEqCounts, TransIndex, Gene2Trans, EffLen)
		WriteBootstrapping(TransNames, alpha_bootstrap, OutFile)