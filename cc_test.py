try:
      import matplotlib.pyplot as plt
      from matplotlib import pylab
except:
      raise

import networkx as nx 
import sys, os, math
import random
import numpy as np
from optparse import OptionParser

'''
  Saves the graph as a PDF.
'''
def save_graph(graph,file_name):
  plt.figure(num=None, figsize=(20, 20), dpi=80)
  plt.axis('off')
  fig = plt.figure(1)
  pos = nx.spring_layout(graph)
  nx.draw_networkx(graph,pos)
  cut = 1.00
  xmax = cut * max(xx for xx, yy in pos.values())
  ymax = cut * max(yy for xx, yy in pos.values())
  plt.xlim(-0.25, xmax+0.25)
  plt.ylim(-0.25, ymax+0.25)
  plt.savefig(file_name,bbox_inches="tight")
  pylab.close()
  del fig

def testCC(graph, target_cc, prob, ite = 0, plist = [], cclist = []):
	nodes = nx.nodes(graph)

	for i in nodes:
		for j in nodes:
			if i < j and j not in g[i]:
				p = np.random.uniform()
				ite = ite + 1 
				if p <= prob:
					graph.add_edge(i, j)
					cc = nx.average_clustering(graph, count_zeros = True)	
					plist.append(p)
					cclist.append(cc)
					if cc >= target_cc:
						return (graph, ite, plist, cclist)	
	
	
	return testCC(graph, target_cc, prob, ite, plist, cclist)
	



class ccInfo:
	prob = None
	estimatedCC = None
	values = []

	def __init__(self, prob, estimated, valuesList):
		self.prob = prob
		self.estimatedCC = estimated
		self.values = valuesList

def generateData(nClique, nObservations):
	pInc = float(2)/float(((nClique - 1)*nClique))
	#print pInc
	nodes = range(0, nClique)
	result = []
	prob = 0
	while prob < 1:
		#print prob
		cc = []
		
		for k in range(0, nObservations):
			g = nx.Graph()
			g.add_nodes_from(nodes)
			for i in nx.nodes(g):
				for j in nx.nodes(g):
					if i < j and j not in g[i]:
						p = np.random.uniform()
						if p <= prob:
							g.add_edge(i, j)
			cc.append(nx.average_clustering(g, count_zeros = True))
			#print nx.average_clustering(g, count_zeros = True)


		estimated = ((nClique - 1) * (nClique - 2)* np.power(prob, 3))/((nClique - 1) * (nClique - 2))
		result.append(ccInfo(prob, estimated, cc))
		print ("%s | %s | %s" % (prob, estimated, cc))
		prob = prob + pInc
	return result
			

def createFile(result, fileName):
	f = open(fileName,'w')
	for i in result:
		for j in i.values:
			f.write(("%s | %s | %s\n" % (i.prob, i.estimatedCC, j)))
	f.close()

		


if __name__ == '__main__':
	nCliques = [10, 50, 100]
	for i in nCliques:
		print i
		r = generateData(10, 10)
		createFile(r, ('data%s_10.csv' % (i, )))


	'''
	nodes = range(1, 100)
	g = nx.Graph()
	g.add_nodes_from(nodes)

	target_cc = 0.3
	g1 = testCC(g, target_cc, 0.1)
	print 'Target cc : ' + str(target_cc)
	cc = nx.average_clustering(g1[0], count_zeros = True)
	print 'Real cc : ' + str(cc)
	print 'iterations: ' + str(g1[1])

	print 'probs: ' + str(g1[2])

	print 'cc: ' + str(g1[3])
	save_graph(g1[0], 'test.pdf')

	'''
	#g = nx.
	#
	#print("Clustering Coefficient: "+str(nx.average_clustering(g, count_zeros=True)))
