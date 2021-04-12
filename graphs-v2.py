import pandas as pd
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, show
from random import sample
import sys
inf = float("inf")
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
    nodes_y.append(node_number)
    node_number= node_number +1 
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
    
# calculate avg degree each node's neighbors
avgDeg = []
node_number = 0
neighbor_id = 0
for row in nodes: 
    node_number= node_number +1 
    numberNeighbors=0 
    sumDegree=0
    avg = 0
    for col in row:
        if col!=0 :
            sumDegree = sumDegree + degrees_x[numberNeighbors]
            neighbor_id = neighbor_id + 1   
        numberNeighbors = numberNeighbors + 1         
    if numberNeighbors != 0:  
        avg = sumDegree/numberNeighbors
   
    avgDeg.append(avg)
    print("Node " + str(node_number) + " - Avg Neighbor Dgree: " + str(avg))    
    print("###########################")


# chart avg degree node's neighbors
output_file("avgNeighbors.html")

p = figure(plot_width=400, plot_height=400 )
p.line(avgDeg, nodes_y, line_width=2)
show(p)





## test shortest path
def printSolution(dist):
    print("Vertex tDistance from Source")
    for node in range(V):
        print(node, "t", dist[node])

# A utility function to find the vertex with
# minimum distance value, from the set of vertices
# not yet included in shortest path tree
def minDistance(dist, sptSet , V):
    V = len(nodes_y)
    # Initilaize minimum distance for next node
    min = sys.maxsize

    # Search not nearest vertex not in the
    # shortest path tree
    for v in range(V):
        if dist[v] < min and sptSet[v] == False:
            min = dist[v]
            min_index = v

    return min_index

# Funtion that implements Dijkstra's single source
# shortest path algorithm for a graph represented
# using adjacency matrix representation
def dijkstra(src , graph , V):
    
    dist = [sys.maxsize] * V
    dist[src] = 0
    sptSet = [False] * V

    for cout in range(V):

        # Pick the minimum distance vertex from
        # the set of vertices not yet processed.
        # u is always equal to src in first iteration
        u = minDistance(dist, sptSet ,V)

        # Put the minimum distance vertex in the
        # shotest path tree
        sptSet[u] = True

        # Update dist value of the adjacent vertices
        # of the picked vertex only if the current
        # distance is greater than new distance and
        # the vertex in not in the shotest path tree
        for v in range(V):
            if graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]:
                dist[v] = dist[u] + graph[u][v]

    printSolution(dist)

# Shortest path algorithm-Dijkstra 
    #def Dijkstra(self, v0):
    # # Initialization operation
dijkstra(1,nodes,len(nodes_y))
#vertexn = len(nodes_y)
# vertexes = nodes_y
# arcs = nodes
# v0=1
# D = [inf]*vertexn  # Used to store the shortest path length from vertex v0 to v
# path = [None]*vertexn  # Used to store the path from vertex v0 to v
# final = [None]*vertexn  # Indicates whether the shortest path from v0 to v has been found
# for i in range(vertexn):
#     final[i] = False
#     D[i] = arcs[v0][i]
#     path[i] = ""  # Path first empty
#     if D[i] < inf:
#         path[i] = vertexes[i]  # If v0 is directly connected to the i-th point, the path is directly changed to i
# D[v0] = 0
# final[v0] = True
# ###
# for i in range(1, vertexn):
#     min = inf  # Find the vertex closest to v0
#     for k in range(vertexn):
#         if(not final[k]) and (D[k] < min):
#             v = k
#             min = D[k]
#     final[v] = True  # The nearest point is found and added to the shortest path set S that has been obtained. The subsequent min will be generated in a vertex other than S
#     for k in range(vertexn):
#         if(not final[k]) and (min+arcs[v][k] < D[k]):
#             # If the shortest distance (v0-v) plus the distance from v to k is less than the existing distance from v0 to k
#             D[k] = min+arcs[v][k]
#             path[k] = str(path[v])+","+str(vertexes[k])
# #return D, path
# #print(D)
# #print(nodes)
# #print(nodes_y)