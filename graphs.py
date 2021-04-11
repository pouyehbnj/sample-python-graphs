import pandas as pd
import networkx as nx
df = pd.read_csv('./soc-sign-bitcoinalpha.csv')
print(df)
#G = nx.from_pandas_adjacency(df)
#print(G)

# Data = open('./soc-sign-bitcoinalpha.csv', "r")
# next(Data, None)  # skip the first line in the input file
# Graphtype = nx.Graph()

# G = nx.parse_edgelist(Data, delimiter=',', create_using=Graphtype,
#                       nodetype=int, data=(('weight', float),))
G = nx.DiGraph()

for d in pd.read_csv('./soc-sign-bitcoinalpha.csv',sep=',', header=None, names=['source', 'destination' , 'weight']):
    G.add_edges_from([tuple(x) for x in d.values])

print(G)
