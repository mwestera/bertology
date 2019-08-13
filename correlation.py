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
    data1 = np.concatenate([d1.reshape(12,-1) for d1 in data1], axis=-1)
    data2 = np.concatenate([d2.reshape(12, -1) for d2 in data2], axis=-1)

    pearson = tree_utils.correlation(data1, data2)
    print(pearson[0], pearson[1])

    for n, (d1, d2) in enumerate(zip(data1, data2)):
        pearson = tree_utils.correlation(d1, d2)
        print(n, pearson[0], pearson[1])


    # for d1,d2 in zip(data1, data2):
    #     print(d1.reshape(12,-1).shape, d2.reshape(12,-1).shape)
    #     quit()
    #     pearson = tree_utils.pearson_correlation(d1, d2)
    #     pearsons.append(pearson[0])
    #     pvalues.append(pearson[1])

    # print(np.nanmean(pearsons), min(pearsons), max(pearsons), np.var(pearsons))
    # print(np.nanmean(pvalues), min(pvalues), max(pvalues), np.var(pvalues))


if __name__ == "__main__":
    main()

