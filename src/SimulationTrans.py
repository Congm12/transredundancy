#!/bin/python

import sys
import struct
import numpy as np
import copy
from pathlib import Path
import subprocess
from TranscriptClass import *


def RandomSelectPairIsoform(GeneTransMap):
	Selection=[]
	for g,v in GeneTransMap.items():
		if len(v) == 2:
			Selection += v
		elif len(v) > 2:
			neworder=np.random.permutation(len(v))
			Selection.append(v[neworder[0]])
			Selection.append(v[neworder[1]])
	return Selection


def RandomSelectTripleIsoform(GeneTransMap):
	Selection = []
	for g,v in GeneTransMap.items():
		if len(v) == 3:
			Selection += v
		elif len(v) > 3:
			neworder = np.random.permutation(len(v))
			Selection += v[neworder[:3]]
	return Selection


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


def SimulateExpressionPairIsoform(Selection, TransGeneMap, TransLength, Cov):
	SimuCov_ori = {}
	SimuCov_rev = {}
	assert(len(Selection) % 2 == 0)
	mean_logCov = np.mean([np.log(x) for x in Cov.values() if x > 0])
	std_logCov = np.std([np.log(x) for x in Cov.values() if x > 0])
	for i in range(0, len(Selection), 2):
		t1 = Selection[i]
		t2 = Selection[i+1]
		assert(TransGeneMap[t1] == TransGeneMap[t2])
		cov_t1 = 0
		cov_t2 = 0
		if t1 in Cov:
			cov_t1 = Cov[t1]
		else:
			cov_t1 = np.random.normal(loc=mean_logCov, scale=std_logCov)
		if t2 in Cov:
			cov_t2 = Cov[t2]
		else:
			cov_t2 = np.random.normal(loc=mean_logCov, scale=std_logCov)
		SimuCov_ori[t1] = cov_t1
		SimuCov_ori[t2] = cov_t2
		if TransLength[t1]*cov_t1 <= 1 and TransLength[t2]*cov_t2 <= 1:
			SimuCov_ori[t1] = 50 / TransLength[t1]
		SimuCov_rev[t1] = cov_t2
		SimuCov_rev[t2] = cov_t1
		if TransLength[t1]*cov_t2 <= 1 and TransLength[t2]*cov_t1 <= 1:
			SimuCov_rev[t2] = 50 / TransLength[t2]
	# scale to the same mean as overall expression
	newmean_logCov = np.mean([np.log(x) for x in SimuCov_ori.values() if x > 0])
	if newmean_logCov < mean_logCov:
		SimuCov_ori = {t:v*mean_logCov/newmean_logCov for t,v in SimuCov_ori.items()}
		SimuCov_rev = {t:v*mean_logCov/newmean_logCov for t,v in SimuCov_rev.items()}
	return [SimuCov_ori, SimuCov_rev]


def WriteSimulatedExpression(SimuCov, TransLength, outputfile):
	fp = open(outputfile, 'w')
	fp.write("# Name\tNumReads\n")
	for t,v in SimuCov.items():
		numreads = max(1, int(v*TransLength[t]))
		fp.write("{}\t{}\n".format(t, numreads))
	fp.close()


def WriteSelectedTransSequence(Selection, TransSequence, outputfile):
	fp = open(outputfile, 'w')
	for t in Selection:
		fp.write(">{}\n".format(t))
		seq = TransSequence[t]
		count = 0
		while count < len(seq):
			fp.write(seq[count:min(len(seq),count+80)] + "\n")
			count += 80
	fp.close()


if __name__=="__main__":
	if len(sys.argv)==1:
		print("python3 SimulationTrans.py <GTFfile> <TransSequenceFile> <BaseExpressionFolder> <OutExpressionPrefix>")
	else:
		GTFfile = sys.argv[1]
		TransSequenceFile = sys.argv[2]
		BaseExpressionFolder = sys.argv[3]
		OutExpressionPrefix = sys.argv[4]

		Transcripts = ReadGTF(GTFfile)
		[GeneTransMap, TransGeneMap] = Map_Gene_Trans(Transcripts)
		TransLength = GetTransLength(Transcripts)
		TransSequence = ReadTranscriptFasta(TransSequenceFile)

		Selection=RandomSelectPairIsoform(GeneTransMap)
		WriteSelectedTransSequence(Selection, TransSequence, OutExpressionPrefix+"_reference.fa")

		cmd = subprocess.Popen("mkdir -p "+OutExpressionPrefix+"_theoexp/", shell=True)
		cmd.communicate()
		p = Path(BaseExpressionFolder)
		ExpFiles = [str(f) for f in p.glob("*/quant.sf")]
		for expfile in ExpFiles:
			sampleID = expfile.split("/")[-2].split("_")[-1]
			Cov = ReadSalmonCov(expfile)
			SimuCov_ori, SimuCov_rev = SimulateExpressionPairIsoform(Selection, TransGeneMap, TransLength, Cov)
			WriteSimulatedExpression(SimuCov_ori, TransLength, OutExpressionPrefix+"_theoexp/theoexp_"+sampleID+"_ori.txt")
			WriteSimulatedExpression(SimuCov_rev, TransLength, OutExpressionPrefix+"_theoexp/theoexp_"+sampleID+"_rev.txt")
