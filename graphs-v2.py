import pandas as pd
import networkx as nx
import numpy as np
from bokeh.plotting import figure, output_file, show
from random import sample
import sys
import plotly.graph_objects as go
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
degree_numpy= np.array(degrees_x)
uniqueValueDeg , occurrence= np.unique(degree_numpy , return_counts=True)
print(uniqueValueDeg)

# occurence=[]
# for value in uniqueValueDeg:
#     occurence.append(np.count_nonzero(value))

print(occurrence)

output_file("line.html")

p = figure(plot_width=400, plot_height=400)
p.line(uniqueValueDeg,occurrence, line_width=2)

show(p)
    
# # calculate avg degree each node's neighbors
# avgDeg = []
# node_number = 0
# neighbor_id = 0
# for row in nodes: 
#     node_number= node_number +1 
#     numberNeighbors=0 
#     sumDegree=0
#     avg = 0
#     for col in row:
#         if col!=0 :
#             sumDegree = sumDegree + degrees_x[numberNeighbors]
#             neighbor_id = neighbor_id + 1   
#         numberNeighbors = numberNeighbors + 1         
#     if numberNeighbors != 0:  
#         avg = sumDegree/numberNeighbors
   
#     avgDeg.append(avg)
#     print("Node " + str(node_number) + " - Avg Neighbor Dgree: " + str(avg))    
#     print("###########################")


# # chart avg degree node's neighbors
# output_file("avgNeighbors.html")

# p = figure(plot_width=400, plot_height=400 )
# p.line(nodes_y, avgDeg,  line_width=2)
# show(p)





# # ## test shortest path
# # def printSolution(dist):
# #     V = len(nodes_y)
# #     print("Vertex tDistance from Source")
# #     for node in range(V):
# #         print(node, "t", dist[node])

# # A utility function to find the vertex with
# # minimum distance value, from the set of vertices
# # not yet included in shortest path tree
# def minDistance(dist, sptSet , V):
   
#     # Initilaize minimum distance for next node
#     min = sys.maxsize
#     min_index=0
#     # Search not nearest vertex not in the
#     # shortest path tree
#     for v in range(V):
#         if dist[v] < min and sptSet[v] == False:
#             min = dist[v]
#             min_index = v

#     return min_index

# # Funtion that implements Dijkstra's single source
# # shortest path algorithm for a graph represented
# # using adjacency matrix representation
# def dijkstra(src , graph , V):
#     # print(sys.maxsize)
#     # print(src)
#     dist = [sys.maxsize] * V
#     dist[src] = 0
#     sptSet = [False] * V

#     for cout in range(V):

#         # Pick the minimum distance vertex from
#         # the set of vertices not yet processed.
#         # u is always equal to src in first iteration
#         u = minDistance(dist, sptSet ,V)

#         # Put the minimum distance vertex in the
#         # shotest path tree
#         sptSet[u] = True

#         # Update dist value of the adjacent vertices
#         # of the picked vertex only if the current
#         # distance is greater than new distance and
#         # the vertex in not in the shotest path tree
#         for v in range(V):
#             if graph[u][v] > 0 and sptSet[v] == False and dist[v] > dist[u] + graph[u][v]:
#                 dist[v] = dist[u] + graph[u][v]

#     return dist

# # Shortest path algorithm-Dijkstra 
#     #def Dijkstra(self, v0):
#     # # Initialization operation

# def calculate_dikstra_for_all_nodes():
#     avgAllNodes=[]
#     number_of_nodes = int(len(nodes_y)*(0.1/100))
#     random_nodes = sample((list(nodes_y)),number_of_nodes)
#     print(random_nodes)
#     for node in random_nodes :
#         dist = []
#         dist = dijkstra(node , nodes , len(nodes_y) )
#         # print(f'sum dist {sum(dist)}')
#         # print(len(dist))
#         avg = sum(dist)/len(dist)
#         avgAllNodes.append(format(avg, '1.0f'))
    
#     return avgAllNodes,random_nodes

# avgAllNodes,random_nodes=calculate_dikstra_for_all_nodes()
# print(avgAllNodes)
# for node,avg in zip( random_nodes ,avgAllNodes ):
#     print(f'node {node} has avg of shortest path : {avg}')
 
#  ## chart dijkstra
# fig3 = go.Figure()
# fig3.add_trace(go.Scatter(x=random_nodes, y=avgAllNodes))

# fig3.show()

# # ## common neighbours
# # def find_neighbours(node):
# #     neighbours = []
# #     col_id = 0
# #     for col in nodes[node]: 
# #         if col!=0 :
# #             neighbours.append(col_id)
# #             col_id = col_id + 1
# #     return neighbours

# # node_id=0
# # common_neighbour_avg_arr = []
# # for node in nodes:
# #     current_node = []
# #     current_neighbour = []
# #     current_node = find_neighbours(node_id)
# #     neighbor_num = 0
# #     node_id = node_id + 1
# #     intersection = []
# #     common_neighbour_sum = 0
# #     common_neighbour_avg = 0  
# #     for neighbour in current_node:
# #         current_neighbour = find_neighbours(neighbour)
# #         intersection = np.intersect1d(current_neighbour, current_node)
# #         common_neighbour_sum =  common_neighbour_sum + len (intersection)
# #         neighbor_num = neighbor_num + 1
# #         print("Node " + str(node_id) + " With Node " + str(neighbour) + " Size:" + str(len (intersection)) + " - Common Neighbor: " + str(intersection))    
# #         print("###########################")

# #         if neighbor_num!=0:
# #             common_neighbour_avg = common_neighbour_sum/neighbor_num
# #             print("Node " + str(node_id) + " With Node " + str(neighbour) + " - Avg Common Neighbor Dgree: " + str(common_neighbour_sum/neighbor_num))    
# #             print("###########################")
# #     common_neighbour_avg_arr.append(common_neighbour_avg)


# # ## common neighbours

# # output_file("CommonNeighbours.html")

# # p = figure(plot_width=400, plot_height=400)
# # p.line(nodes_y,common_neighbour_avg_arr, line_width=2)

# # show(p)
