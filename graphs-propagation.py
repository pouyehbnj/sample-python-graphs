import pandas as pd
import networkx as nx
import random
import numpy as np
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
from bokeh.plotting import figure, output_file, show
from random import sample
# loading data to dataframe
df = pd.read_csv('./soc-sign-bitcoinalpha.csv',
                 names=['source', 'destination', 'weight'], usecols=[0, 1, 2])
print("raw data:")
print(df)
Graph = nx.from_pandas_edgelist(
    df, source='source', target='destination', edge_attr='weight')
print(Graph)

iteration_x =[]
percentage_y = []

def color_graph(node, s, iteration):
    
    colors = set()
    colors.add(node)
    old_percentage = 0
    while len(s) < len(list(Graph.nodes())):
        neighbors = np.array([])
        for color in colors:
            s.add(color)
            current_neighbors = list(Graph.neighbors(color))
            neighbors = np.concatenate((neighbors, current_neighbors), axis=0)

        colors.clear()
        colors = set(neighbors)
        
        color_percentage = (len(s) / len(list(Graph.nodes())))*100
        if old_percentage == color_percentage:
            random_node = random.sample((set(list(Graph.nodes())) - s), 1)[0]
            color_graph(random_node, s, iteration)
        else:
            old_percentage = color_percentage
            iteration_x.append(iteration)
            print("Iteration Number:", str(iteration))
            iteration = iteration + 1
            percentage_y.append(color_percentage)
            print("The percentage is :", str(color_percentage), "%")
            print("Number of colored nodes :", str(len(s)))
            print("Number of colored nodes :", str(len(list(Graph.nodes()))))

    
        

s = set()
random_node = random.sample(set(list(Graph.nodes())), 1)[0]
color_graph(random_node, s, 0)
output_file("CommonNeighbours.html")

p = figure(plot_width=400, plot_height=400)
p.line(iteration_x,percentage_y, line_width=2)

show(p)
