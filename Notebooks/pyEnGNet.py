import concurrent.futures

import pandas as pd
import scipy.stats as scp
import numpy as np
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
import networkx as nx
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor

sameval = 0.7


def intervals(total, partes):
    lenpartes = int(total / partes)
    lista = []
    for i in range(partes):
        if i == partes - 1:
            end = total
        else:
            end = lenpartes * (i + 1)
        lista.append((lenpartes * i, end))
    return lista


class PyEnGNet:
    def __init__(self, nparr=None, nmi_th=sameval, spearman_th=sameval, kendall_th=sameval, pearson_th=sameval, readded_th=0.7, cores=int(mp.cpu_count()/2)):
        if nparr is not None:
            self.maindata = nparr
        else:
            raise Exception("No data given")

        self.row_size = len(self.maindata)
        self.column_size = len(self.maindata[0])

        self.spearman_threshold = spearman_th
        self.kendall_threshold = kendall_th
        self.nmi_threshold = nmi_th
        self.pearson_threshold = pearson_th

        self.readded_edges_threshold = readded_th
        self.ncores = cores

    # Calcular el coeficiente de correlación de spearman (Slow but correct way)
    def spearman_corr(self):
        spearmancorrs = np.zeros((self.row_size, self.row_size))
        spearmanpvalues = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(self.row_size)):
            for j in range(i + 1, self.row_size):
                spearmancorrs[i][j], spearmanpvalues[i][j] = scp.spearmanr(self.maindata[i], self.maindata[j])
        return spearmancorrs, spearmanpvalues

    # Calcular el coeficiente de correlación de Kendall
    def kendall_corr(self):
        kendallcorrs = np.zeros((self.row_size, self.row_size))
        kendallpvalues = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(self.row_size)):
            for j in range(i + 1, self.row_size):
                kendallcorrs[i][j], kendallpvalues[i][j] = scp.kendalltau(self.maindata[i], self.maindata[j])

        return kendallcorrs, kendallpvalues

    def pearson_corr(self):
        pearsoncorrs = np.zeros((self.row_size, self.row_size))
        pearsonpvalues = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(self.row_size)):
            for j in range(i + 1, self.row_size):
                pearsoncorrs[i][j], pearsonpvalues[i][j] = scp.pearsonr(self.maindata[i], self.maindata[j])

        return pearsoncorrs, pearsonpvalues

    # Calcular el NMI
    def nmi_corr(self, precission):
        nmi_values = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(self.row_size)):
            for j in range(i + 1, self.row_size):
                v1 = list(map(lambda x: round(x, precission), self.maindata[i]))
                v2 = list(map(lambda x: round(x, precission), self.maindata[j]))
                nmi_values[i][j] = normalized_mutual_info_score(v1, v2)
        return nmi_values

    # Calcula el NMI entre dos vectores
    def single_nmi(self, arr1, arr2, precission=1):
        """
        Calcula el NMI entre dos vectores y devuelve
            ans -> 1 si pasa el umbral, 0 en caso contrario
            corr -> El valor resultante de aplicar Normalized Mutual Info

        :param arr1:
        :param arr2:
        :param precission:
        :return ans, corr:
        """
        ans = 0
        rnd = lambda x: round(x, precission)
        v1 = np.array(list(map(rnd, arr1)))
        v2 = np.array(list(map(rnd, arr2)))
        corr = normalized_mutual_info_score(v1, v2)
        if corr >= self.nmi_threshold:
            ans = 1
        return ans, corr

    def single_spearman(self, arr1, arr2):
        """
        Recibe dos vectores y calcula el coeficiente de correlación de spearman entre ellos, devuelve:
            ans -> 1 si pasa el umbral, 0 en caso contrario
            corr -> El coeficiente de correlación de spearman en valor absoluto
        :param arr1:
        :param arr2:
        :return:
        """
        ans = 0
        corr, pv = scp.spearmanr(arr1, arr2)
        corr = abs(corr)
        if corr >= self.spearman_threshold:
            ans = 1
        return ans, corr

    def single_pearson(self, arr1, arr2):
        """
        Recibe dos vectores y calcula el coeficiente de correlación de spearman entre ellos, devuelve:
            ans -> 1 si pasa el umbral, 0 en caso contrario
            corr -> El coeficiente de correlación de spearman en valor absoluto
        :param arr1:
        :param arr2:
        :return:
        """
        ans = 0
        corr, pv = scp.pearsonr(arr1, arr2)
        corr = abs(corr)
        if corr >= self.pearson_threshold:
            ans = 1
        return ans, corr

    def single_kendall(self, arr1, arr2):
        """
        Recibe dos vectores y calcula el coeficiente de correlación de Kendall entre ambos
            ans -> 1 si pasa el umbral, 0 en caso contrario
            corr -> El valor resultante, en valor absoluto, del coeficiente de correlación de kendall
        :param arr1:
        :param arr2:
        :return:
        """
        ans = 0
        corr, pv = scp.kendalltau(arr1, arr2)
        corr = abs(corr)
        if corr >= self.kendall_threshold:
            ans = 1
        return ans, corr

    def validate_corr(self, i, j, accepted_values):
        major_voting = 0

        # Las dos filas que vamos a utilizar
        v = self.maindata[i]
        w = self.maindata[j]

        # Agregamos las respuestas de los tests a una lista que servirá para el calculo de los pesos
        tests = [self.single_pearson(v, w), self.single_kendall(v, w), self.single_spearman(v, w)]

        for test in tests:
            major_voting += test[0]

        if major_voting >= 2:
            accepted_values.append(
                [i, j, {'weight': self.calculate_weight(tests)}])

    def calculate_weight(self, tests):
        weight = []
        for test in tests:
            if bool(test[0]):
                weight.append(test[1])
        return np.mean(weight)

    def edge_corr_validation(self, start, end):
        accepted = []
        for i in tqdm(range(start, end)):
            for j in range(i + 1, self.row_size):
                self.validate_corr(i, j, accepted)
        return accepted

    def mainmethod(self):
        intervalos = intervals(len(self.maindata), self.ncores)
        edges = []

        with ProcessPoolExecutor(max_workers=self.ncores) as executor:
            results = []
            for rango in intervalos:
                start = rango[0]
                end = rango[1]
                results.append(executor.submit(self.edge_corr_validation, start, end))

        for f in concurrent.futures.as_completed(results):
            for val in f.result():
                edges.append(val)

        return edges

    def engnet_1_0(self):
        oedges = self.mainmethod()
        #G = nx.Graph()
        #G.add_edges_from(oedges)
        #G2 = nx.maximum_spanning_tree(G, weight='weight', algorithm="kruskal")

        #G3 = self.readd_edges(G, G2)
        #fedges = nx.to_edgelist(G3)

        #return G3, fedges

    def readd_edges(self, graph0, graph2):

        origin_edges = nx.to_edgelist(graph0)
        tree_edges = nx.to_edgelist(graph2)

        # Obtain eliminated edges
        eliminated_edges = [edge for edge in origin_edges if edge not in tree_edges]

        # Obtain hubs
        detected_hubs = [graph2.degree(node) for node in graph2.nodes if graph2.degree(node) > 2]

        if len(detected_hubs) != 0:
            mean_degree = round(np.mean([graph2.degree(node) for node in graph2.nodes if graph2.degree(node) > 2]))
            hubs = []
            for node in graph2.nodes:
                if graph2.degree(node) >= mean_degree:
                    hubs.append(node)

            readded = []
            for node in hubs:
                for e in eliminated_edges:
                    if (node in e) and (e not in readded) and (e[2]['weight'] > self.readded_edges_threshold):
                        readded.append(e)
            G3 = nx.Graph()
            G3.add_edges_from(nx.to_edgelist(graph2))
            G3.add_edges_from(readded)

        return G3


    def __str__(self):
        return f"PyEnGNet Object with shape ({self.row_size},{self.column_size})"


if __name__ == "__main__":
    df = pd.read_csv("/home/daiego/PycharmProjects/pyEnGNet/pyEnGNet/Notebooks/Data/113_exp_mat_cond_1.csv")
    df = df.drop(df.columns[[0, 2]], axis=1)
    data = df.to_numpy()
    peg = PyEnGNet(nparr=data)
    G, aristas = peg.engnet_1_0()


