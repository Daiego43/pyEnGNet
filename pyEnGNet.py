import pandas as pd
import scipy.stats as scp
import numpy as np
from tqdm import tqdm
from numba import jit


def extract_nparray(filename):
    df = pd.read_csv(filename, delimiter=',')
    df = df.loc[:, df.columns != 'genes']
    cleandataset = df.to_numpy()
    return cleandataset


class PyEnGNet:
    def __init__(self, nparr=None):
        if nparr is not None:
            self.maindata = nparr
        else:
            raise Exception("No data given")

        self.row_size = len(self.maindata)
        self.column_size = len(self.maindata[0])

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

    def __str__(self):
        return f"PyEnGNet Object with shape ({self.row_size},{self.column_size})"
    # Calcular el coeficiente de correlación de spearman (Slow but correct way)


if __name__ == "__main__":
    peg = PyEnGNet(extract_nparray("datasample_big.csv"))
    peg.spearman_test()
    peg.kendall_test()
