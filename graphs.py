import pandas as pd
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, show
from random import sample
#loading data to dataframe
df = pd.read_csv('./soc-sign-bitcoinalpha.csv' ,  names=['source','destination','weight'] ,usecols=[0,1,2])
print("raw data:")
print(df)
# creating a weighted adjacency matrix
name_to_node = {name: i for i, name in enumerate(np.unique(df[["source", "destination"]].values))}
n_nodes = len(name_to_node)
A = np.zeros((n_nodes, n_nodes))
for row in df.itertuples():
    n1 = name_to_node[row.source]
    n2 = name_to_node[row.destination]
    A[n1, n2] += row.weight
    A[n2, n1] += row.weight

print("adjacency matrix:")
print(A)
Graph = nx.from_pandas_edgelist(df,source='source',target='destination' , edge_attr='weight')
print(Graph)
# finding degrees 
print(nx.info(Graph))
for s in Graph.degree():
    print(s)


#degree chart 
nodes = sorted(list(Graph.nodes()))
nodes_y = []  
degrees_x= [] 
for node in nodes:
    degrees_x.append(Graph.degree[int(node)])
    nodes_y.append(node)

output_file("line.html")

p = figure(plot_width=400, plot_height=400)
p.line(degrees_x,nodes_y, line_width=2)

show(p)

# neighbours average degree

for node in list(Graph.nodes):
    numberNeighbors=0 
    sumDegree=0
    for neighbor in list(Graph.neighbors(node)):
        numberNeighbors= numberNeighbors + 1 
        sumDegree =  Graph.degree(neighbor) + sumDegree
    avgDegNeighbor = sumDegree/numberNeighbors
    print("node = " + str(node) + " avg of neighbors is : " + str(avgDegNeighbor))

        
#shortest path to other nodes 

compressed_graph = Graph 
number_of_nodes = int(len(list(Graph.nodes()))*(95/100))
random_nodes = sample((list(Graph.nodes())),number_of_nodes)
compressed_graph.remove_nodes_from(random_nodes)
sp = dict(nx.all_pairs_shortest_path(compressed_graph))
print("hi")
for node in sp.values():
    print("Node:")
    print(node)
    print("---------------------------------")
    len_sum = 0
    count = 0
    for paths in node.values():
       print(paths)
       len_sum = len_sum + len(paths)
       count = count + 1
    avg = len_sum / count
    print("Average of shortest path to other nodes: " + str(avg))
    print("***********")

