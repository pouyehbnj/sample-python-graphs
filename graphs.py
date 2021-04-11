import pandas as pd
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, show
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

## calculate avg degree each node's neighbors
avgDeg = []
for node in nodes:
    numberNeighbors=0 
    sumDegree=0
    for neighbor in list(Graph.neighbors(node)):
        numberNeighbors= numberNeighbors + 1 
        sumDegree =  Graph.degree(neighbor) + sumDegree
    avgDegNeighbor = sumDegree/numberNeighbors
    avgDeg.append(avgDegNeighbor)
    print("node = " + str(node) + " avg of neighbors is : " + str(avgDegNeighbor))
## chart avg degree node's neighbors
output_file("avgNeighbors.html")

p = figure(plot_width=400, plot_height=400 )
p.line(nodes_y, avgDeg, line_width=2)

show(p)

##calculate common neighbors
listAvgCommonNeigh=[]
for node in nodes:
    numberNeighbors=0 
    avgCommonNeigh=0
    numberOfCommonNeighbors=0
    
    for neighbor in list(Graph.neighbors(node)):
        numberNeighbors= numberNeighbors + 1 
        numberOfCommonNeighbors = numberOfCommonNeighbors + len(sorted(nx.common_neighbors(Graph,node, neighbor)))
        print("node = " + str(node) + " has this common neighbors : " + str(sorted(nx.common_neighbors(Graph,node, neighbor))) + " with neighbor " +str(neighbor))
    avgCommonNeigh = numberOfCommonNeighbors/numberNeighbors
    listAvgCommonNeigh.append(avgCommonNeigh)
    print("node = " + str(node) + " avg of  common neighbors are : " + str(avgCommonNeigh))

## chart common neighbors
     




output_file("commonNeighbors.html")
print(len(listAvgCommonNeigh))
p = figure(plot_width=400, plot_height=400 )
p.line(nodes_y, listAvgCommonNeigh, line_width=2)

show(p)

