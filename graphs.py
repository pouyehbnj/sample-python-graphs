import pandas as pd
df = pd.read_csv('./soc-sign-bitcoinalpha.csv', header=[0,1,2])
print(df)
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