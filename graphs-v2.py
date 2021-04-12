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
# finding degrees 

print("degrees:")
nodes = A 
nodes_y = []  
degrees_x= []
node_number=0
for row in nodes:
    node_number= node_number +1 
    nodes_y.append(node_number)
    degree = 0
    for col in row:
        if col!=0 :
            degree = degree+1
    degrees_x.append(degree)
    print("Node " + str(node_number) + " - Degree: " + str(degree))    
    print("###########################")

#degree chart 

output_file("line.html")

p = figure(plot_width=400, plot_height=400)
p.line(degrees_x,nodes_y, line_width=2)

show(p)
    
