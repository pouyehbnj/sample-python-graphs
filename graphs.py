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


