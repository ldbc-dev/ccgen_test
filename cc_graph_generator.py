
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

#DEFINITIONS
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

'''
  This class represents a community with a core-periphery structure
'''
class community:
  def __init__(self, core, periphery, prob=1.0):
    self.core = core 
    self.periphery = periphery
    self.p = prob 

'''
  This class holds information for computing the expected clustering coefficient 
'''
class clustering_info:
  def __init__(self, degree_sequence):
    # lists to hold the expected degrees of nodes in the core (core degree, periphery degree and external degree)
    self.core_node_expected_core_degree = [0]*len(degree_sequence)
    self.core_node_excedence_degree = [0]*len(degree_sequence)
    self.core_node_expected_periphery_degree = [0]*len(degree_sequence)
    self.core_node_expected_external_degree = [0]*len(degree_sequence)
    # list to store the clustering coefficient of a node
    self.clustering_coefficient = [0.0]*len(degree_sequence)




'''
  Adds the nodes of a degree sequence into a graph. Ids of the nodes span from 0 to N-1
'''
def add_nodes(G, degree_sequence):
  index = 0
  for deg in degree_sequence:
    G.add_node(index)
    index+=1

'''
  Given a degree sequence and a community core, returns the budget of stubs to connect with the periphery  
'''
def create_initial_budget(degree_sequence,core, p = 1.0):
  budget = []
  core_size = len(core)
  for node in core:
    budget.append(degree_sequence[node] - (core_size - 1)*p)
  return budget

'''
  Given a degree sequence, a core and a periphery, checks whether we can form a valid community or not.
'''
def check_budget(degree_sequence, core, periphery):
  temp_budget = create_initial_budget(degree_sequence, core)
  sorted_periphery = sorted(periphery,key=lambda a: degree_sequence[a], reverse=True) 
  for member in sorted_periphery:
    degree = degree_sequence[member]
    remaining = degree
    i = 0
    while i < len(temp_budget) and remaining > 0:
      if temp_budget[i] > 0:
        temp_budget[i]-=1
        remaining -=1
      i+=1
    if remaining > 0:
      return False
  return True

'''
  Given a degree sequence, and to indices of the sequence spanning a window of degrees, it checks whether we
  can create a valid community with the nodes of that window of degrees
'''
def find_solution(degree_sequence, begin, last):
  nodes = range(begin,last+1)
  nodes = sorted(nodes, key=lambda a: degree_sequence[a], reverse=True)
  core = list() 
  periphery = list()
  for node in nodes:
    if degree_sequence[node] >= len(core):
      core.append(node)
    else:
      periphery.append(node)
  return core, periphery, check_budget(degree_sequence,core,periphery)


'''
  Given a degree sequence, it returns a list of valid communities. Identifiers of nodes are indices to the degree sequence
'''
def generate_communities(degree_sequence):
  communities = []
  last = 0
  begin = 0 
  end = len(degree_sequence)
  threshold = 5
  core = list
  periphery = list
  found = False
  while last < end:
    best = last 
    num_tries = 0
    while num_tries <= threshold and last < end:
      num_tries +=1
      temp_core, temp_periphery, found = find_solution(degree_sequence, begin, last)
      if found:
        core = list(temp_core)
        periphery = list(temp_periphery)
        num_tries = 0
        best=last
      last+=1
    communities.append(community(core, periphery))
    last = best + 1
    begin = last
  return communities



'''
  Given a graph G, a community and a degree sequence, creates the edges of the core of the community in G
'''
def create_edges_community_core(G,community,degree_sequence):
  for node in community.core:
    for node2 in community.core:
      if node < node2:
        p = random.random()
        if p <= community.p:
          G.add_edge(node,node2)

  for node in community.core:
    if len(G[node]) > degree_sequence[node]:
      print "ERROR: Created more edges than expected in community core"
  connect_nodes_with_community(G,community,degree_sequence,community.periphery)

'''
  Given a graph G, a community, a degree sequence and a set of nodes, creates edges between these nodes and the core of the community
'''
def connect_nodes_with_community(G,community,degree_sequence,nodes):
  nodes = sorted( nodes, key=lambda a: degree_sequence[a], reverse=True)
  budget = create_initial_budget(degree_sequence, community.core)
  stubs_core = []
  stubs_nodes = []
  for node in nodes:
    stubs_nodes.extend([node]*degree_sequence[node])
  index = 0
  for node in community.core:
    stubs_core.extend([node]*int(budget[index]))
    index+=1
  random.shuffle(stubs_nodes)
  random.shuffle(stubs_core)
  while len(stubs_core) > 0 and len(stubs_nodes) > 0:
    index = random.randint(0,len(stubs_core)-1)
    node1 = stubs_core[index]
    stubs_core.pop(index)
    index = random.randint(0,len(stubs_nodes)-1)
    node2 = stubs_nodes[index]
    stubs_nodes.pop(index)
    G.add_edge(node1,node2)


'''
  Given a graph G, a list of communities and a degree sequence, creates the edges connecting members of the cores of different communities
'''
def fill_graph_with_remaining_edges(G,communities,degree_sequence):
  stubs = []
  for community in communities:
    for node in community.core:
      diff = degree_sequence[node] - len(G[node])
      if diff > 0:
        for i in range(0,diff):
          stubs.append(node)
  random.shuffle(stubs)
  while len(stubs) > 0:
    index1 = random.randint(0, len(stubs)-1)
    node1 = stubs[index1]
    stubs.pop(index1)
    if len(stubs) > 0:
      index2 = random.randint(0, len(stubs)-1)
      node2 = stubs[index2]
      stubs.pop(index2)
      G.add_edge(node1,node2)

  for community in communities:
    for node in community.core:
      if len(G[node]) > degree_sequence[node]:
        print "ERROR: Created more edges than expected in community core externally"
  

'''
  Given a degree sequence, fixes the distribution so a valid graph configuration can be created
'''
def fix_degree_sequence(degree_sequence):
  ids = range(0,len(degree_sequence))
  ids = sorted(ids, key=lambda a: degree_sequence[a])
  accum = degree_sequence[ids[0]];
  for i in range(1,len(ids)):
    if degree_sequence[ids[i]] >= len(degree_sequence):
      degree_sequence[ids[i]] = len(degree_sequence)

    if degree_sequence[ids[i]] >= accum:
      print("FIXED NODE FROM DEGREE SEQUENCE")
      degree_sequence[ids[i]] = accum
    accum += degree_sequence[ids[i]]

'''
  Given a list of communities and a degree sequence, changes the probabilities of the communities so to improve the clustering coefficient
'''
def improve_cc(communities, degree_sequence, cc_info):
  sorted_communities = filter(lambda c: c.p < 1.0, sorted_communities)
  if len(sorted_communities) == 0:
    return False
  index = random.randint(0, len(sorted_communities)-1)
  sorted_communities[index].p = min(sorted_communities[index].p+0.05,1.0)
  estimate_cc_community(degree_sequence, cc_info,sorted_communities[index])
  return True

'''
  Given a list of communities and a degree sequence, changes the probabilities of the communities so to worsen the clustering coefficient
'''
def worsen_cc(communities, degree_sequence):
  sorted_communities = filter(lambda c: c.p > 0.0, communities)
  if len(sorted_communities) == 0:
    return False
  index = random.randint(0, len(sorted_communities)-1)
  sorted_communities[index].p = max(sorted_communities[index].p-0.05,0.0)
  estimate_cc_community(degree_sequence, cc_info,sorted_communities[index])
  return True

'''
  Given a list of communities, a degree sequence and a target clustering coefficient, it refines the communities' configuration to approach as much as possible the target clustering coefficient
'''
def refine_communities(communities,degree_sequence,target_cc, cc_info, max_cc):
  if max_cc < target_cc:
    print "ERROR: TARGET CC LARGER THAN MAX CC"
    return
  current_cc = clustering_coefficient(cc_info)
  look_ahead = 5
  tries = 0
  while (current_cc - target_cc )/ target_cc > 0.01 and tries <= look_ahead:
    print(str(current_cc)+" "+str(tries))
    found = False
    tries+=1
    if current_cc < target_cc:
      found = improve_cc(communities,degree_sequence)
    else:
      found = worsen_cc(communities,degree_sequence)
    if found == True:
      current_cc = clustering_coefficient(cc_info)
      tries = 0

'''
  Recomputes the cc of the nodes of the corresponding community
'''
def estimate_cc_community(degree_sequence, cc_info, community, max_cc = False):
  num_periphery_stubs = 0
  excedence_degree_sum = 0
  prob = community.p
  if max_cc:
    prob = 1.0
  for node in community.core:
    cc_info.core_node_expected_core_degree[node] = (len(community.core) - 1)*prob
    cc_info.core_node_excedence_degree[node] = degree_sequence[node] - cc_info.core_node_expected_core_degree[node] 
    excedence_degree_sum += cc_info.core_node_excedence_degree[node]
    cc_info.core_node_expected_periphery_degree[node] = 0
  for node in community.core:
    for other_periphery in community.periphery: 
      cc_info.core_node_expected_periphery_degree[node] += (cc_info.core_node_excedence_degree[node]/excedence_degree_sum)*degree_sequence[other_periphery]
    cc_info.core_node_expected_external_degree[node] = degree_sequence[node] - cc_info.core_node_expected_core_degree[node] - cc_info.core_node_expected_periphery_degree[node]

  # computing clustering coefficient of periphery nodes 
  for node in range(0,len(community.periphery)):
    degree = degree_sequence[community.periphery[node]]
    num_periphery_stubs+=degree
    if degree > 1:
      cc_info.clustering_coefficient[community.periphery[node]] = community.p

  # computing clustering coefficient of core nodes 
  for node in range(0,len(community.core)):
    degree = degree_sequence[community.core[node]]
    size = len(community.core)
    if degree > 1 and size > 1:
      internal_triangles = 0
      periphery_triangles = 0
      internal_external_triangles = 0
      external_external_triangles_1 = 0
      external_external_triangles_2 = 0

  # core core triangles
      internal_triangles = (size - 1)*(size - 2)*math.pow(prob,3)
  # core periphery triangles
      periphery_triangles = 0
      periphery_degree = cc_info.core_node_expected_periphery_degree[community.core[node]]
      if periphery_degree > 0:
        for other_core in range(0,len(community.core)):
          if community.core[other_core] != node:
            other_core_periphery_degree = cc_info.core_node_expected_periphery_degree[community.core[other_core] ]
            if other_core_periphery_degree > 1:
              for other_periphery in range(0,len(community.periphery)): 
                if degree_sequence[community.periphery[other_periphery]] > 1:
                  p = periphery_degree*prob*other_core_periphery_degree
                  p*= (degree_sequence[community.periphery[other_periphery]])/float(num_periphery_stubs)
                  p*= (degree_sequence[community.periphery[other_periphery]])/float(num_periphery_stubs)
                  periphery_triangles +=p
      deg = cc_info.core_node_expected_core_degree[community.core[node]] + cc_info.core_node_expected_periphery_degree[community.core[node]] + cc_info.core_node_expected_external_degree[community.core[node]]
      if deg > 1:
        cc_info.clustering_coefficient[community.core[node]] = (internal_triangles + periphery_triangles) / (deg*(deg-1))

'''
  Computes the expected clustering coefficient of the current configuration
'''
def clustering_coefficient(cc_info):
  accum = 0.0
  for cc in cc_info.clustering_coefficient:
    accum+=cc
  return accum / len(cc_info.clustering_coefficient)


#PROGRAM START
option_parser = OptionParser()
option_parser.add_option("-g", "--graph_file_name", dest="graph_file_name",
                      help="File name where the graph is stored", default="./graph.csv")
option_parser.add_option("-d", "--distribution_file_name", dest="distribution_file_name",
                      help="File name where the degree sequence is stored", default="./deg_dist.txt")
option_parser.add_option("-i", "--inpunt_degree_distribution", dest="input_degree_distribution",
                      help="Use the degree distribution found in the specified file")
option_parser.add_option("-n", "--nodes", dest="num_nodes",
                      help="Number of nodes of the graph", type="int", default=1000)
option_parser.add_option("-c", "--clustering_coefficient", dest="target_cc",
                      help="The target clustering coefficient", type="float", default=0.3)
(options, args) = option_parser.parse_args()


target_cc = options.target_cc 

G = nx.Graph()

degree_sequence = []
#GENERATING THE DEGREE SEQUENCE
if options.input_degree_distribution == None: 
  degree_sequence = nx.utils.zipf_sequence(options.num_nodes, alpha=2.0, xmin=1)
  fix_degree_sequence(degree_sequence)
  #EXPORTING DEGREE DISTRIBUTION
  deg_dist_file = open(options.distribution_file_name,"w")
  for deg in degree_sequence:
    deg_dist_file.write(str(deg)+"\n")
  deg_dist_file.close()
else:
  input_file = open(options.input_degree_distribution,"r")
  for line in input_file.readlines():
    degree_sequence.append(int(line))
  
#ADDING NODES TO GRAPH
add_nodes(G,degree_sequence)
#GENERATING COMMUNITIES

cc_info = clustering_info(degree_sequence)

print "Creating community configuration"
communities = generate_communities(degree_sequence)

for community in communities:
  estimate_cc_community(degree_sequence, cc_info, community, max_cc = True)

print "Computing maximum possible clustering coefficient"
max_cc = clustering_coefficient(cc_info)

print "Max CC: "+str(max_cc)
print "Refining communities"
refine_communities(communities,degree_sequence,target_cc,cc_info,max_cc)
print "Creating graph"
print "Creating intra-community edges"
for community in communities:
  create_edges_community_core(G,community, degree_sequence)

print "Creating inter-community edges"
fill_graph_with_remaining_edges(G,communities,degree_sequence)
G.remove_edges_from(G.selfloop_edges())

num_edges = 0
for deg in degree_sequence:
  num_edges+=deg

nx.write_edgelist(G,options.graph_file_name,data=False)
#CREATING BUDGET LIST
print("Expected number of edges: "+str(num_edges/2))
print("Resulting number of edges: "+str(len(G.edges())))
print("Number of Communities: "+str(len(communities)))
print("Max clustering coefficient: "+str(max_cc))
print("Target clustering coefficient: "+str(target_cc))
print("Clustering Coefficient: "+str(nx.average_clustering(G, count_zeros=True)))
print("Assortativity: "+str(nx.degree_pearson_correlation_coefficient(G)))
largest_CC = sorted(nx.connected_component_subgraphs(G), key=len, reverse=True)[0]
print("Diameter of the largest CC: "+str(nx.diameter(largest_CC)))
print("Largest CC: "+str(len(largest_CC)))

