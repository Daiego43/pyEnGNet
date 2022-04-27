import pandas as pd
import scipy.stats as scp
import numpy as np
from tqdm import tqdm
from sklearn.metrics.cluster import normalized_mutual_info_score
import warnings

warnings.filterwarnings('ignore')

sameval = 0.5

class PyEnGNet:
    def __init__(self, nparr=None, nmi_th=sameval, spearman_th=sameval, kendall_th=sameval):
        if nparr is not None:
            self.maindata = nparr
        else:
            raise Exception("No data given")

        self.row_size = len(self.maindata)
        self.column_size = len(self.maindata[0])

        self.spearman_threshold = spearman_th
        self.kendall_threshold = kendall_th
        self.nmi_threshold = nmi_th

    # Calcular el coeficiente de correlación de spearman (Slow but correct way)
    def spearman_test(self):
        spearmancorrs = np.zeros((self.row_size, self.row_size))
        spearmanpvalues = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(len(self.maindata))):
            for j in range(len(self.maindata)):
                spearmancorrs[i][j], spearmanpvalues[i][j] = scp.spearmanr(self.maindata[i], self.maindata[j])
        return spearmancorrs, spearmanpvalues

    # Calcular el coeficiente de correlación de Kendall
    def kendall_test(self):
        kendallcorrs = np.zeros((self.row_size, self.row_size))
        kendallpvalues = np.zeros((self.row_size, self.row_size))
        for i in tqdm(range(len(self.maindata))):
            for j in range(len(self.maindata)):
                kendallcorrs[i][j], kendallpvalues[i][j] = scp.kendalltau(self.maindata[i], self.maindata[j])

        return kendallcorrs, kendallpvalues

    # Calcular el NMI
    def nmi_test(self, precission):
        nmi_values = np.zeros((self.row_size, self.row_size))
        for i in range(len(self.maindata)):
            for j in range(len(self.maindata)):
                rnd = lambda x: round(x, precission)
                v1 = list(map(rnd, self.maindata[i]))
                v2 = list(map(rnd, self.maindata[j]))
                nmi_values[i][j] = normalized_mutual_info_score(v1, v2)
        return nmi_values

    def single_nmi(self, arr1, arr2, precission=1):
        ans = 0
        rnd = lambda x: round(x, precission)
        v1 = np.array(list(map(rnd, arr1)))
        v2 = np.array(list(map(rnd, arr2)))
        corr = normalized_mutual_info_score(v1, v2)
        if corr >= self.nmi_threshold:
            ans = 1
        return ans, corr

    def single_spearman(self, arr1, arr2):
        ans = 0
        corr, pv = scp.spearmanr(arr1, arr2)
        if corr >= self.spearman_threshold:
            ans = 1
        return ans, abs(corr)

    def single_kendall(self, arr1, arr2):
        ans = 0
        corr, pv = scp.kendalltau(arr1, arr2)
        if corr >= self.kendall_threshold:
            ans = 1
        return ans, abs(corr)

    def engnet_1_0(self):
        engnet_accepted_values = []
        major_voting = 0
        for i in range(self.row_size):
            for j in range(i+1, self.row_size):
                # Las dos filas que vamos a utilizar
                v = self.maindata[i]
                w = self.maindata[j]
                tests = []

                # Aplicamos nmi y kendall y sumamos al major voting
                nmi = self.single_nmi(v, w)
                kend = self.single_kendall(v, w)

                major_voting += nmi[0] + kend[0]

                # Agregamos las respuestas de ambos tests a una lista que servirá para el calculo de los pesos
                tests.append(nmi)
                tests.append(kend)

                # Si los resultados anteriores no son concluyentes se procede a una tercera prueba
                if major_voting >= 1:
                    spear = self.single_spearman(v, w)
                    major_voting += spear[0]
                    tests.append(spear)

                # Si se obtiene una mayoría, entonces la relacion se considera apta, se guarda la arista que los
                # representa y se calcula el peso asociado a dicha arista
                if major_voting >= 2:
                    engnet_accepted_values.append((i, j, {'weight': self.calculate_weight(tests)}))

                major_voting = 0

        return engnet_accepted_values

    def calculate_weight(self, tests):
        weight = []
        for test in tests:
            if bool(test[0]):
                weight.append(test[1])
        return np.mean(weight)

    def __str__(self):
        return f"PyEnGNet Object with shape ({self.row_size},{self.column_size})"


if __name__ == "__main__":
    peg = PyEnGNet(extract_nparray("datasample_shittys.csv"))
    peg.engnet_1_0()
    pd.DataFrame()
