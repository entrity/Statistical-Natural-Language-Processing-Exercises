#!/usr/bin/python

'''
Created by Markham
to replicate table 6.4 from Manning's NLP book (page 203)

ARGV should be file containing lexed corpus
'''

import sys
import nltk

if (len(sys.argv) < 2):
	print 'Usage:\r\n\t%s <lexed corpus file>' % sys.argv[0]
	sys.exit(1)

# Import lexed corpus
print 'Importing lexed corpus from %s...' % sys.argv[1]
f = open(sys.argv[1])
corpus = f.read().split()
f.close()

# Create frequency distribution of unigrams in order to count token types
tokens = nltk.FreqDist(corpus)

N = len(corpus) # number of bigrams
V = len(tokens.keys()) # vocabulary, i.e. token types (word types)
B = V**2

# Create frequency distribution of bigrams
fd_2gram = nltk.FreqDist(nltk.ngrams(corpus, 2))

# Create Heldout frequency distribution of bigrams
fd_partition_1 = nltk.FreqDist(nltk.ngrams(corpus[:len(corpus)/2 - 1],2))
fd_partition_2 = nltk.FreqDist(nltk.ngrams(corpus[len(corpus)/2:],2))
heldout_pd_2gram = nltk.HeldoutProbDist(fd_partition_1, fd_partition_2)
Tr = heldout_pd_2gram._calculate_Tr() # Number of times that all ngrams which occurred r times in partition 1 appear in partition 2

# Calculate frequency using Laplace's rule
def fLap(r):
	fLap = float(r+1)*N/(N+B) # "plus one"
	return fLap

def fHeldout(r, nR):
	# There are nR n-grams that occurred r times.
	# There are Tr[r] occurrences in partition 2 of n-grams that appeared r times in partition 1.
	# There are N n-grams total.
	fHeldout = float(Tr[r])*N/(nR*N)
	# Error checking Tr: calculate my own Tr to see if it's done correctly:
	rTr = 0
	for item in fd_partition_1.items():
		if item[1] == r:
			rTr += fd_partition_2[item[0]]
	if rTr != Tr[r]:
		raise Exception('Mismatch in Tr[{:d}]. You probably don\'t understand what Tr is. Given {:f}. Calculated {:f}'.format(r, Tr[r], rTr))
	return fHeldout

def fCrossValidation(r):
	Tr01 = 0
	for item in fd_partition_1.items():
		if item[1] == r:
			Tr01 += fd_partition_2[item[0]]
	Tr10 = 0
	for item in fd_partition_2.items():
		if item[1] == r:
			Tr10 += fd_partition_1[item[0]]
	Nr1 = reduce(lambda count, item: count + 1 if item[1] == r else count, fd_partition_1.items(), 0)
	Nr2 = reduce(lambda count, item: count + 1 if item[1] == r else count, fd_partition_2.items(), 0)
	pDel = float(Tr01 + Tr10) / (N * (Nr1 + Nr2))
	return N * pDel

# Print 1st lines of table
# r equals fMLE because MLE = r / N and fMLE = r * N / N = r
print "%8s %8s %8s %8s %8s %13s %13s" % ('r = fMLE', 'f emp', 'fLap', 'f del', 'f GT', 'Nr', 'Tr')
n0 = N - len(fd_2gram.keys())
fLap0 = float(n0*N)/(N+B)
fHeldout0 = float(Tr[0])*N/(n0*N)
print "%8d %8.6f %8.4f %8s %8s %13s %13d" % (0, fHeldout0, fLap0, 'f del', 'f GT', n0, Tr[0])

# Print remaining rows of table
for r in range(1,10):
	nR = reduce(lambda count, item: count + 1 if item[1] == r else count, fd_2gram.items(), 0) # Number of n-grams which occurred r times in corpus
	print "%8d %8.6f %8.6f %8.6f %8s %13s %13d" % (
		r,
		fHeldout(r, nR),
		fLap(r),
		fCrossValidation(r),
		'f GT',
		nR,
		Tr[r]) 
	