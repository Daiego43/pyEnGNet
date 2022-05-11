import networkx as nx
import pandas as pd
from pyEnGNet.Notebooks import pyEnGNet as p


df = pd.read_csv("/pyEnGNet/Notebooks/Data/113_exp_mat_cond_1.csv")
df = df.drop(df.columns[[0,2]], axis=1)
print(df)
data = df.to_numpy()
# PyEnGNet class receives a numpy array and three thresholds corresponding to the three tests applied
# By default these thresholds are 0.5 for testing purpouses
peg = p.PyEnGNet(data)

edges = peg.engnet_1_0()

G = nx.Graph()
