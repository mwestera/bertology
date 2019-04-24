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
        print('BERTs raw outputs loaded from', raw1)
        data1 = pickle.load(file)

    with open(raw2, 'rb') as file:
        print('BERTs raw outputs loaded from', raw2)
        data2 = pickle.load(file)

    pearsons = []
    pvalues = []
    for d1,d2 in zip(data1, data2):
        pearson = tree_utils.pearson_correlation(d1, d2)
        pearsons.append(pearson[0])
        pvalues.append(pearson[1])

    print(np.nanmean(pearsons), min(pearsons), max(pearsons), np.var(pearsons))
    print(np.nanmean(pvalues), min(pvalues), max(pvalues), np.var(pvalues))



def plot_pearson_scores(scores_df, args):

    ## Stack for plotting
    scores_df = scores_df[['pearson']].stack().stack().stack().reset_index(level=[1, 2, 3])
    scores_df = scores_df[scores_df.measure != 'p-value']

    plt.figure(figsize=(10, 8))
    plt.ylim(0.0, 1.0)
    ax = sns.lineplot(x='layer', y='pearson', hue='relations', data=scores_df)

    ax.set_title("Pearson coefficient of dependency tree with {} ({}{}, {}{})".format(args.method,
                               args.combine,
                               ', norm' if args.method == 'attention' and args.normalize_heads else '',
                               args.group_merger,
                               ', transpose' if args.transpose else '',))


    out_filepath = '{}/pearson_{}{}{}{}{}{}.png'.format(args.out,
                                                           args.method,
                                                           "_" + args.combine if args.combine != "no" else "",
                                                           '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                           ('_' + str(
                                                               args.n_items)) if args.n_items is not None else '',
                                                           '_' + args.group_merger,
                                                           '_transpose' if args.transpose else '',
                                                           )


    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)

    return out_filepath

if __name__ == "__main__":
    main()

