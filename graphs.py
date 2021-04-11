import pandas as pd
import numpy as np
df = pd.read_csv('./soc-sign-bitcoinalpha.csv' ,  names=['source','destination','weight'] ,usecols=[0,1,2])
print(df)
name_to_node = {name: i for i, name in enumerate(np.unique(df[["source", "destination"]].values))}
n_nodes = len(name_to_node)
A = np.zeros((n_nodes, n_nodes))
for row in df.itertuples():
    n1 = name_to_node[row.source]
    n2 = name_to_node[row.destination]
    A[n1, n2] += row.weight
    A[n2, n1] += row.weight
print(A)