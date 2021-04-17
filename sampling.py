import pandas as pd
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, show
import random
import math
import matplotlib.pyplot as plt
from bokeh.plotting import figure, output_file, show
import plotly.graph_objects as go
from matplotlib import pylab



#loading data to dataframe
df = pd.read_csv('./soc-sign-bitcoinalpha.csv' ,  names=['source','destination','weight'] ,usecols=[0,1,2])
print("raw data:")
print(df)

Graph = nx.from_pandas_edgelist(df,source='source',target='destination' , edge_attr='weight')
print(Graph)


#node sampling
def node_sampling(Graph , k):

    sampled_nodes = random.sample(Graph.nodes, k)
    sampled_graph = Graph.subgraph(sampled_nodes)
    print(sampled_nodes)
    nx.draw_networkx(sampled_graph , node_size=1000, with_labels = True)
    plt.savefig("node-sampling.png")
   # save_graph(sampled_graph,"my_graph.pdf")
    return sampled_graph

# edge sampling
def edge_sampling(G ,k):
    V = G.nodes()
    G1=nx.Graph()
    # Calculate number of nodes in Graph G
    Vs = []
    # Empty list Vs
    
    while (len(Vs) <= k):
    # Loops run till sample size * length of V where V is number of nodes in graph as calculated above.
        edges_sample = random.sample(G.edges(), 1)
        # Randomly samples one edge from a graph at a time
        for a1, a2 in edges_sample:
        # Nodes corresponding to sample edge are retrieved and added in Graph G1
            G1.add_edge(a1, a2)
            if (a1 not in Vs):
                Vs.append(a1)
            if (a2 not in Vs):
                Vs.append(a2)
    # Statement written just to have a check of a program

    for x in G1.nodes():
        neigh = (set(G1.nodes()) & set(list(G.neighbors(x))))
        # Check neighbours of sample node and if the nodes are their in sampled set then edge is included between them.
        for y in neigh:
        # Check for every node's neighbour in sample set of nodes
            G1.add_edge(x, y)
            # Add edge between the sampled nodes
    nx.draw_networkx(G1 , node_size=1000, with_labels = True)
    plt.savefig("edge-sampling.png")
    return G1

#edge_sampling(Graph)

## random walking
def random_walking(complete_graph , growth_size , T , nodes_to_sample):
    complete_graph = nx.convert_node_labels_to_integers(complete_graph, 0, 'default', True)
    # giving unique id to every node same as built-in function id
    for n, data in complete_graph.nodes(data=True):
        complete_graph.nodes[n]['id'] = n

    nr_nodes = len(complete_graph.nodes())
    upper_bound_nr_nodes_to_sample = nodes_to_sample
    index_of_first_random_node = random.randint(0, nr_nodes - 1)
    sampled_graph = nx.Graph()

    sampled_graph.add_node(complete_graph.nodes[index_of_first_random_node]['id'])

    iteration = 1
    edges_before_t_iter = 0
    curr_node = index_of_first_random_node
    print(complete_graph.neighbors(curr_node))
    while sampled_graph.number_of_nodes() != upper_bound_nr_nodes_to_sample:
        #print(complete_graph.degree(curr_node))
        edges = [n for n in complete_graph.neighbors(curr_node)]
        index_of_edge = random.randint(0, len(edges) - 1)
        chosen_node = edges[index_of_edge]
        sampled_graph.add_node(chosen_node)
        sampled_graph.add_edge(curr_node, chosen_node)
        curr_node = chosen_node
        iteration = iteration + 1

        if iteration % T == 0:
            if ((sampled_graph.number_of_edges() - edges_before_t_iter) < growth_size):
                curr_node = random.randint(0, nr_nodes - 1)
            edges_before_t_iter = sampled_graph.number_of_edges()
    nx.draw_networkx(sampled_graph , node_size=1000, with_labels = True)
    plt.savefig("random-walk.png")
    return sampled_graph

#random_walking(Graph , 2 , 100 , 100)

# def save_graph(graph,file_name):
#     #initialze Figure
#     plt.figure(num=None, figsize=(20, 20), dpi=80)
#     plt.axis('off')
#     fig = plt.figure(1)
#     pos = nx.spring_layout(graph)
#     nx.draw_networkx_nodes(graph,pos)
#     nx.draw_networkx_edges(graph,pos)
#     nx.draw_networkx_labels(graph,pos)

#     cut = 1.00
#     xmax = cut * max(xx for xx, yy in pos.values())
#     ymax = cut * max(yy for xx, yy in pos.values())
#     plt.xlim(0, xmax)
#     plt.ylim(0, ymax)

#     plt.savefig(file_name,bbox_inches="tight")
#     pylab.close()
#     del fig


def find_important_nodes(Graph):
    degree=nx.degree_centrality(Graph)
    degreeArray = sorted(degree, key=degree.get, reverse=True)[:30]

   
    centrality = nx.eigenvector_centrality_numpy(Graph)
    eigenvectorArray = sorted(centrality, key=centrality.get, reverse=True)[:30]
    
    betCent = nx.betweenness_centrality(Graph, normalized=False, endpoints=True , weight = 'weight')
    betweennessArray = sorted(betCent, key=betCent.get, reverse=True)[:30]
    important_nodes=list(set(degreeArray)|set(eigenvectorArray) | set(betweennessArray))
    #important_nodes=list(set(degreeArray)|set(eigenvectorArray))
    print(important_nodes)
    return important_nodes


node_sample_graph=node_sampling(Graph , 500)
edge_sample_graph=edge_sampling(Graph , 500)
random_walk_graph=random_walking(Graph , 2 , 100 , 500)
important_nodes_main_graph=find_important_nodes(Graph)
important_nodes_node_sample_graph=find_important_nodes(node_sample_graph)
important_nodes_edge_sample_graph=find_important_nodes(edge_sample_graph)
important_nodes_random_walk_graph=find_important_nodes(random_walk_graph)

numberOfCommonNodeSample= list(set(important_nodes_main_graph)&set(important_nodes_node_sample_graph))
numberOfCommonEdgeSample= list(set(important_nodes_main_graph)&set(important_nodes_edge_sample_graph))
numberOfCommonWalkSample= list(set(important_nodes_main_graph)&set(important_nodes_random_walk_graph))

print(f'number of common important nodes with node sample is  { len(numberOfCommonNodeSample) }')
print(f'number of common important nodes with edge sample is { len(numberOfCommonEdgeSample) }')
print(f'number of common important nodes with random walk sample is {len(numberOfCommonWalkSample)}')

output_file("comparison.html")

p = figure(plot_width=400, plot_height=400)
comparison = []
comparison.append(len(numberOfCommonNodeSample))
comparison.append(len(numberOfCommonEdgeSample))
comparison.append(len(numberOfCommonWalkSample))
x = [] 
x.append('node sample')
x.append('edge sample')
x.append('random walk')
print(x)
print(comparison)
# p.line(x,comparison, line_width=2)
# show(p)
fig3 = go.Figure()
fig3.add_trace(go.Scatter(x=x, y=comparison))

fig3.show()