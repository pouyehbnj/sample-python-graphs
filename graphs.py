import pandas as pd
import networkx as nx
import numpy as np
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
# degrees = [val for (node, val) in sorted(Graph.degree(), key=lambda pair: pair[0])]

# print(Graph.degree(876))
print(nx.info(Graph))
for s in Graph.degree():
    print(s)


for node in list(Graph.nodes):
    numberNeighbors=0 
    sumDegree=0
    for neighbor in list(Graph.neighbors(node)):
        numberNeighbors= numberNeighbors + 1 
        sumDegree =  Graph.degree(neighbor) + sumDegree
    avgDegNeighbor = sumDegree/numberNeighbors
    print("node = " + str(node) + " avg of neighbors is : " + str(avgDegNeighbor))

        


