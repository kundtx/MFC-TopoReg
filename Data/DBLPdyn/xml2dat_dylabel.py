#! /usr/bin/python
# -*- coding: utf-8 -*-
#####################################################################
# Copyright (C) 2012 Michele Dallachiesa - dallachiesa@disi.unitn.it
#####################################################################


import sys
import os
from math import *
# import HTMLParser
import re
import random

limitMinYear = 1999
limitMaxYear = 2019

#####################################################################

# labels and histograms.
labels = dict()
# from a subset of top 100 conferences in computer science.. http://academic.research.microsoft.com/RankList?entitytype=3&topDomainID=2&subDomainID=0&last=0&start=1&end=100

labels["data-mining"] = ["VLDB", "ICDE", "SIGMOD", "PODS", "KDD", "WWW", "CIKM", "ICDM", "PAKDD", "TKDE"]
labels["networking"] = [ "TCOM", "INFOCOM", "MOBICOM", "SIGCOMM", "VTC", "MobiHoc", "IPSN", "SenSys", "HPCA", "WCNC", "HPDC", "ICNP"]
labels["machine-learning"] = ["NIPS", "ICML", "UAI", "ICPR", "AAAI", "IJCAI", "ICGA", "ICPR", "FGR", "ISNN", "TEC"]
labels["bioinformatics"] = ["bioinformatics", "BioMED", "ISMB", "RECOMB", "Biocomputing"]
labels["operating-systems"] = ["USENIX", "SOSP", "OSDI", "NOSSDAV", "ICDCS", "RTSS", "PODC", "HPCA", "ICS", "HPDC"]
labels["security"] = ["CRYPTO", "EUROCRYPT", "Security", "CCS", "privacy", "NDSS"]
labels["computer-graphics"] = ["SIGGRAPH", "SOCG", "I3D"]
labels["computing-theory"] = ["STOC", "FOCS", "SODA", "ICPP"]
labels["computer-human-interaction"] = ["CHI", "UIST"]
labels["info-retrieval"] = ["SIGIR", "WWW", "ISWC", "TREC", "Hypertext"]
labels["software-engineering"] = ["ICSE"]
labels["computer-vision"] = ["ICCV", "ECCV", "CVPR"]
labels["computational-linguistics"] = ["ACL", "linguistics", "COLING"]
labels["verification-testing"] = ["ITC", "DATE", "CAV"]

labels["data-mining"].extend(["SDM", "ECML-PKDD", "DSAA", "BigData", "ICDMW", "ASONAM"])
labels["networking"].extend(["IWQoS", "NetSys", "HotNets", "NSDI", "ICC", "GLOBECOM", "WCNC"])
labels["machine-learning"].extend(["NeurIPS", "AISTATS", "IJCNN", "ICONIP", "ICMLA", "COLT", "ECML"])
labels["bioinformatics"].extend(["RECOMB", "ISMB/ECCB", "BIBE", "BMCB", "CSB", "Bioinformatics"])
labels["operating-systems"].extend(["LISA", "MSST", "Middleware", "EuroSys", "ICAC", "SYSTOR"])
labels["security"].extend(["RAID", "CSF", "ASIACRYPT", "ACNS", "ESORICS", "SecDev", "Usenix Security"])
labels["computer-graphics"].extend(["CGI", "PacificGraphics", "EGSR", "SI3D", "SIBGRAPI"])
labels["computing-theory"].extend(["LICS", "MFCS", "POPL", "ICALP", "CSL", "DLT"])
labels["computer-human-interaction"].extend(["CSCW", "DIS", "IUI", "MobileHCI", "Pervasive", "UbiComp"])
labels["info-retrieval"].extend(["ECIR", "CIKM", "JCDL", "IRJ", "COLING", "TPDL", "ESWC"])
labels["software-engineering"].extend(["FSE", "ASE", "ISSTA", "ICSE-C", "ESEC/FSE"])
labels["computer-vision"].extend(["WACV", "BMVC", "3DV", "CVIU", "ICVS", "ACCV"])
labels["computational-linguistics"].extend(["NAACL", "EMNLP", "LREC", "ACL-IJCNLP", "COLING"])
labels["verification-testing"].extend(["ETAPS", "SPIN", "ICTSS", "TAP", "ATVA", "RTAS"])

retainedAuthors = set() # authorId
authorLabelHistogram = {y:{} for y in range(limitMinYear, limitMaxYear+1)} # (authorId, authorLabelHistogram....)
labelIds = dict() # (labelStr,labelId)
authorsNames = dict() # (authorId,authorStr)
labelKeywords = dict() # (#keywordStr, labelStr)
labelKeyAll = set()
coAuthoredCount = dict() # (authorId, int) # no. of co-authored papers.

authors = dict() # (str,authorId)
edges = dict() # (authorId1,authorId2)


authorId = 0

minYear = 10000
maxYear = 0

minProbab = 1.0
maxProbab = 0.0

#####################################################################


def getLabelId(str):	
	global labelId

	if str == False:
		return 0
	else:
		return labelIds[str]
	

def buildlabelKeywords():
	global labels
	global labelKeywords
	global labelIds

	labelId = 0
	
	print("Loading label keywords...")
	
	for (label,keywords) in labels.items():
		
		labelId = labelId +1
		labelIds[label] = labelId
		
		keywords = [x.lower() for x in keywords]
		for keyword in keywords:
			labelKeywords[keyword] = label


def extract_name(text):
	pattern = r"<(booktitle|journal)>(.*?)<\/(booktitle|journal)>"  # Pattern for booktitle/journal
	match = re.search(pattern, text)

	if match:
		matched = match.group(2)
		matched = re.sub(r'\([^)]*\)', '', matched).strip()

		pattern_workshop_conf = r"\w+(@|\/)(\w+)"  # Pattern for workshop@conf
		match_workshop_conf = re.search(pattern_workshop_conf, matched)
		if match_workshop_conf:
			return match_workshop_conf.group(2)
		else:
			return matched

	else:
		return text

def getUniqueLabels(text):
	global labelKeywords
	text = extract_name(text.strip()).lower()
	labelKeyAll.add(text)
	authorLabels = set()
	if text in labelKeywords:
		authorLabels.add(labelKeywords[text])
	else:
		pattern = re.compile("\.|/|,|:|;|\ |\[|\]|\{|\}|\(|\)|\<|\>|\"|\'\n\t\r")
		tokens = pattern.split(text)
		tokens = [x.lower() for x in tokens]


		for token in tokens:		
			# labelKeyAll.add(token)
			if token in labelKeywords:
				authorLabels.add(labelKeywords[token])			

	return authorLabels
	
		
def updateHistogram(author, authorLabels, year):
	global authorLabelHistogram
	global labels
	
	authorLabelHistogramYear = authorLabelHistogram[year]
	if author not in authorLabelHistogramYear:
		authorLabelHistogramYear[author] = dict()
		for label in labels.keys():
			authorLabelHistogramYear[author][label] = 0
			
	for authorLabel in authorLabels:		
		authorLabelHistogramYear[author][authorLabel] += 1
		

def getTopLabel(author, year): #get most frequent label associated to "author"
	global labels
	global authorLabelHistogram
	
	authorLabelHistogramYear = authorLabelHistogram[year]
	topFreq = 0
	topLabel = False # special label for unknown value.
	
	for label in labels.keys():
		if authorLabelHistogramYear[author][label] > topFreq:
			topFreq = authorLabelHistogramYear[author][label]
			topLabel = label

	return topLabel
	
	
def getAuthorId(s):	
	global authorId
	global authors
		
	if s not in authors:
		authorId = authorId +1
		authors[s] = authorId
		authorsNames[authorId] = s
		return authorId
	else:
		return authors[s]

	
	
def insertEdge(author1, author2, year):

	global minYear
	global maxYear
	global coAuthoredCount
			
	minYear = min(minYear, year)
	maxYear = max(maxYear, year)
	
	assert author1 < author2
		
	if (author1, author2) not in edges:
		edges[(author1, author2)] = dict()
		retainedAuthors.add(author1)
		retainedAuthors.add(author2)		

	try:
		edges[(author1, author2)][year] += 1.0
	except KeyError:
		edges[(author1, author2)][year] = 1.0

####
	try:
		coAuthoredCount[author1]+=1
	except KeyError:
		coAuthoredCount[author1]=1
		
	try:
		coAuthoredCount[author2]+=1	
	except KeyError:
		coAuthoredCount[author2]=1	



def loadData(pathname):
	global minYear
	global maxYear
	
	state = 0
	text = ""
	year = False
	
	print("Loading dataset from file '" + pathname + "' ...")
	
	fileSize = os.path.getsize(pathname)
	
	with open(pathname, "r+") as f:
	# pars = HTMLParser.HTMLParser()
		nArticles = 0
		lines = f.readlines()
		for line in lines:				
			line = line.strip()

			if line == "":
				continue
						
			if line.startswith("<article") or line.startswith("<inproceedings"):			
				#	assert state == 0		
				nArticles = nArticles + 1
				state = 1
				coAuthors = []
				year = False
				text = ""											
				if nArticles % 5000 == 0:
					print("[" + str(int(float(f.tell()) / fileSize * 100)) + "%] of DBLP dataset processed. nArticles=" + str(nArticles))
								

			if line.startswith("</article") or line.startswith("</inproceedings"):
				#	assert state == 1
				if year < limitMinYear or year > limitMaxYear:
					state = 0
					continue
				
				for author1 in coAuthors:
					id1= getAuthorId(author1) # why here: ensures that we add authors from papers with just one author.
					for author2 in coAuthors:
						if author1 == author2:
							continue					
						id2= getAuthorId(author2)						
						if id1 < id2: # every pair is enumerated twice with the double-loop without this if.
							insertEdge(id1, id2, year)
										
				authorsLabels = getUniqueLabels(text)									
				for author in coAuthors:
					updateHistogram(author, authorsLabels, year)								

				state = 0
				

			if state == 0:
				continue					

				
			# if article...
			
			if line.startswith("<booktitle>") or line.startswith("<journal>"):		
				text = text + "\n" + line
				continue
			
			if line.startswith("<author>") == True:
				author = line.replace("<", ">").split(">")[2]
				coAuthors.append(author)
				continue
				
			if line.startswith("<year>") == True:
				year = int(line.replace("<", ">").split(">")[2])
								
		



def edgeProbability(author1, author2):
	global minProbab
	global maxProbab
	
	edge = edges[(author1, author2)]
	
	assert len(edge) > 0
		
	P = float(len(edge)) / float(maxYear - minYear + 1) # year-granularity
			
	assert (P >= 0 and P <= 1)

	minProbab = min(minProbab, P)
	maxProbab = max(maxProbab, P)
	
	return P


def edgeProbability2(author1, author2):
	global minProbab
	global maxProbab
		
	edge = edges[(author1, author2)]
	
	assert len(edge) > 0
	
	c= 0
	for (year,count) in edge.items():
		c+= count

 	# *2 because each publication is counted twice: in coAuthoredCount[author1] and in coAuthoredCount[author2]
	P = float(c*2) / float(coAuthoredCount[author1]+coAuthoredCount[author2])
	
	# P is the ratio of published papers by both authors that the two authors co-authored.
		
	assert (P >= 0 and P <= 1)

	minProbab = min(minProbab, P)
	maxProbab = max(maxProbab, P)
	
	return P

	
	
def dumpGraph(pathname):
				
	# print "Dumping to file '" + pathname + "' ..."
	f = open(pathname, "w")

	# f.write("*labels*\n")
	# for label in labels.keys():		
		# f.write(str(getLabelId(label)) + ",\"" + label + "\"\n")

	# f.write("\n*nodes*\n")	
	# for authorId in retainedAuthors:
		# authorName = authorsNames[authorId]
		# f.write( str(authorId) + ",\"" + authorName + "\"," + str(getLabelId(getTopLabel(authorName))) + "\n")
	
	# f.write("\n*edges*" + "\n")
	for (author1,author2) in edges:
		authorName1 = authorsNames[author1]
		authorName2 = authorsNames[author2]
		
		edge = edges[(author1, author2)]
		for (year,count) in edge.items():
			label1 = str(getLabelId(getTopLabel(authorName1, year)))
			label2 = str(getLabelId(getTopLabel(authorName2, year)))
			f.write(str(author1) + "\t" + str(author2) + "\t" + str(int(count)) + "\t" + str(year) + "\t" + str(label1) + "\t" + str(label2) + "\n")
	
	f.close()	


def main(argv):

	dblpPathnamePrefix = 'dblp-2019-12-01'#"dblp-2018-02-01"


	buildlabelKeywords()

	loadData(dblpPathnamePrefix + ".xml")	# loadData sets minYear,maxYear,minProbab,maxProbab values and edge frequencies.
	print("Loaded " + str(len(authors)) + " authors and " + str(len(edges)) + " edges.")
	
	dumpGraph(dblpPathnamePrefix + ".dat")	
	print("Stats: minYear=" + str(minYear) + " maxYear=" +  str(maxYear) + " minProbab=" + str(minProbab) + " maxProbab=" + str(maxProbab))
	
	# print(labelKeyAll)
		
	return
	
	buildlabelKeywords()	
	updateHistogram("michele", getUniqueLabels("this is sme;nois<VLDB>i, text!ICDE"))
	updateHistogram("michele", getUniqueLabels("how 'ICDE\""))
	updateHistogram("michele", getUniqueLabels("I like info security"))	
	print(authorLabelHistogram["michele"])
	print(getTopLabel("michele"))


#####################################################################
if __name__ == "__main__":
    sys.exit(main(sys.argv))


