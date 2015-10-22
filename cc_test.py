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


def testCC(graph, target_cc, prob, ite = 0):
	nodes = nx.nodes(graph)
	ite = ite + 1 
	for i in nodes:
		for j in nodes:
			if i != j and j not in g[i]:
				p = np.random.uniform()
				if p >= prob:
					graph.add_edge(i, j)
	
	cc = nx.average_clustering(graph, count_zeros = True)	
	if cc >= target_cc:
		return (graph, ite)
	else:
		return testCC(graph, target_cc, prob, ite)
	


if __name__ == '__main__':
	nodes = range(1, 100)
	g = nx.Graph()
	g.add_nodes_from(nodes)

	target_cc = 0.3
	g1 = testCC(g, target_cc, 0.1)
	print 'Target cc : ' + str(target_cc)
	cc = nx.average_clustering(g1[0], count_zeros = True)
	print 'Real cc : ' + str(cc)
	print 'iterations: ' + str(g1[1])

	save_graph(g1[0], 'test.pdf')


	#g = nx.
	#
	#print("Clustering Coefficient: "+str(nx.average_clustering(g, count_zeros=True)))