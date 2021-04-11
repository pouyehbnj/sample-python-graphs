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
# mat = nx.to_pandas_adjacency(Graph ,weight='weight')
# print(mat)
# G=nx.from_pandas_dataframe(df, 'source', 'destination', 'weight')
# print(G)

print('hi')


# print(A)
# print(df)
#G = nx.from_pandas_adjacency(df)
#print(G)

# Data = open('./soc-sign-bitcoinalpha.csv', "r")
# next(Data, None)  # skip the first line in the input file
# Graphtype = nx.Graph()

# G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
#                       nodetype=int, data=(('weight', float),))
# G = nx.DiGraph()

# for d in pd.read_csv('./soc-sign-bitcoinalpha.csv',sep=',', header=None, names=['source', 'destination' , 'weight']):
#     G.add_edges_from([tuple(x) for x in d.values])

# print(G)
#df.columns = ['node_1','node_2','weight']
# name_to_node = {name: i for i, name in enumerate(np.unique(df[["node_1", "node_2"]].values))}
# n_nodes = len(name_to_node)
# A = np.zeros((n_nodes, n_nodes))
# for row in df.itertuples():
#     n1 = name_to_node[row.node_1]
#     n2 = name_to_node[row.node_2]
#     A[n1, n2] += row.weight
#     A[n2, n1] += row.weight
# print(A)