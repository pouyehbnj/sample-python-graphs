import pandas as pd
import networkx as nx
df = pd.read_csv('./soc-sign-bitcoinalpha.csv')
#print(df)
G = nx.from_pandas_adjacency(df)
print(G)