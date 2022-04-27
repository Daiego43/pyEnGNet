import networkx as nx
import pyEnGNet as p

data = p.extract_nparray("datasample_shittys.csv")
# PyEnGNet class receives a numpy array and three thresholds corresponding to the three tests applied
# By default these thresholds are 0.5 for testing purpouses
peg = p.PyEnGNet(data)

edges = peg.engnet_1_0()

G = nx.Graph()
