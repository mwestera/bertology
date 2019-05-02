import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
import pickle

import tree_utils


def main():
    """
    For checking the correlation between attention and gradients.
    """

    raw1 = 'output/auxiliary/en_gum-ud-dev_POS_attention_500.pkl'
    raw2 = 'output/auxiliary/en_gum-ud-dev_POS_gradient_500.pkl'

    with open(raw1, 'rb') as file:
        data1 = pickle.load(file)
        print('BERTs raw outputs loaded from', raw1)

    with open(raw2, 'rb') as file:
        data2 = pickle.load(file)
        print('BERTs raw outputs loaded from', raw2)

    pearsons = []
    pvalues = []
    for d1,d2 in zip(data1, data2):
        pearson = tree_utils.pearson_correlation(d1, d2)
        pearsons.append(pearson[0])
        pvalues.append(pearson[1])

    print(np.nanmean(pearsons), min(pearsons), max(pearsons), np.var(pearsons))
    print(np.nanmean(pvalues), min(pvalues), max(pvalues), np.var(pvalues))


if __name__ == "__main__":
    main()

