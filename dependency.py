from pytorch_pretrained_bert import BertTokenizer

import numpy as np
import pandas as pd
import csv

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
import seaborn as sns
import matplotlib.pylab as pylab
import imageio

import os
import warnings

import argparse

import pickle

import interface_BERT
import data_utils
import tree_utils

parser = argparse.ArgumentParser(description='e.g., experiment.py data/example.csv')
parser.add_argument('data', type=str,
                    help='Path to data file (typically .csv).')
parser.add_argument('--n_items', type=int, default=None,
                    help='Max number of items from dataset to consider.')
parser.add_argument('--out', type=str, default=None,
                    help='Output directory for plots (default: creates a new /temp## folder)')
parser.add_argument('--raw_out', type=str, default=None,
                    help='Output directory for raw BERT outputs, pickled for efficient reuse.')
parser.add_argument('--method', type=str, default='gradient', choices=["gradient", "attention"],
                    help='attention or gradient (default)')
parser.add_argument('--combine', type=str, default='no', choices=["chain", "cumsum", "no"],
                    help='how to combine layers: chain, cumsum, or no (default)')
parser.add_argument('--normalize_heads', action="store_true",
                    help='To apply normalization per attention head (only used for "attention" method).')
parser.add_argument('--ignore_groups', action="store_true",
                    help='To ignore groupings of tokens in the input data, and compute/plot per token. NOTE: POTENTIALLY BUGGY.')
parser.add_argument('--group_merger', type=str, default='mean', choices=["mean", "sum"],
                    help='how to combine the weights of tokens (and token pieces) within a token group: sum or mean (default)')
## Disabled, as axis labels etc. would be incorrect:
# parser.add_argument('--transpose', action="store_true",
#                     help='To transpose the plots; by default they read like "rows influenced by cols" (otherwise: rows influencing cols).')
parser.add_argument('--no_diff_plots', action="store_true",
                    help='To NOT plot the differences between levels of a given factor.')
parser.add_argument('--gif', action="store_true",
                    help='To create animated gif of plots across layers.')
parser.add_argument('--bert', type=str, default='bert-base-cased',
                    help='Which BERT model to use (default bert-base-cased; not sure which are available)')
parser.add_argument('--factors', type=str, default=None,
                    help='Which factors to plot, comma separated like "--factors reflexivity,gender"; default: first 2 factors in the data')
parser.add_argument('--no_global_colormap', action="store_true",
                    help='Whether to standardize plot coloring across plots ("global"); otherwise only per plot (i.e., per layer)')
parser.add_argument('--balance', action="store_true",
                    help='To compute and plot balances, i.e., how much a token influences minus how much it is influenced.')
parser.add_argument('--cuda', action="store_true",
                    help='To use cuda.')

# TODO Make sure to try merging token pieces with summing...

# TODO: perhaps it's useful to allow plotting means over layers; sliding window-style? or chaining but with different starting points?
# TODO: Is attention-chain bugged? Plots are uninterpretable; without normalization super high values only at layer 10-11... with normalization... big gray mess.
# TODO: Should I take sum influence per group of tokens, or mean? E.g., with averaging, "a boy" will be dragged down by uninformative "a"...
# TODO: Check why attention-chain doesn't yield good pictures; does normalization even make sense? What about normalizing the whole matrix just for the sake of comparability across layers?

# TODO: I got an error when running on example.csv with --n_items 1 or even 2.

def main():
    """
    To run this code with default settings and example data, do
       $ python experiment.py data/example.csv
    """

    ## Argument parsing
    args = parser.parse_args()
    if args.factors is not None:
        args.factors = args.factors.split(",")
        if len(args.factors) > 2:
            print("WARNING: Cannot plot more than 2 factors at a time. Trimming to", args.factors[:2])
            args.factors = args.factors[:2]
    if args.out is not None:
        if os.path.exists(args.out):
            if input('Output directory {} already exists. Risk overwriting files? N/y'.format(args.out)) != 'y':
                quit()

    if args.raw_out is None:
        args.raw_out = 'data/auxiliary/{}_{}{}{}{}.pkl'.format(os.path.basename(args.data)[:-4],
                                                         args.method,
                                                         '_'+args.combine if args.combine != 'no' else '',
                                                         '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                         ('_'+str(args.n_items)) if args.n_items is not None else '')
    need_BERT = True
    if os.path.exists(args.raw_out):
        if input('Raw output file exists. Overwrite? (N/y)') != "y":
            need_BERT = False

    ## Set up tokenizer, data
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=("uncased" in args.bert))
    items = data_utils.parse_data(args.data, tokenizer, max_items=args.n_items, words_as_groups=True, as_dependency='dependencies')

    # print(len(items), 'items')
    # with pd.option_context('display.max_rows', 10, 'display.max_columns', None):
    #     print(items)

    ## Store for convenience
    args.factors = args.factors or items.factors[:2]    # by default use the first two factors from the data

    ## Now that args.factors is known, finally choose output directory
    if args.out is None:
        dirname = 'temp'
        out_idx = 0
        if not os.path.exists("output"):
            os.mkdir('output')
        if not os.path.exists("data/auxiliary/"):
            os.mkdir('data/auxiliary')
        while any(x.startswith(dirname) for x in os.listdir('output')):
            out_idx += 1
            dirname = 'temp{}'.format(out_idx)
        dirname += "_{}{}{}{}".format(args.method,
                                       "-" + args.combine if args.combine != "no" else "",
                                       "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                       '_' + '-x-'.join(args.factors) if len(args.factors) > 0 else '')
        args.out = os.path.join("output", dirname)
        os.mkdir(args.out)


    ## Apply BERT or, if available, load results saved from previous run
    if need_BERT:
        data_for_all_items = interface_BERT.apply_bert(items, tokenizer, args)
        with open(args.raw_out, 'wb') as file:
            pickle.dump(data_for_all_items, file)
            print('BERTs raw outputs saved as',args.raw_out)
    else:
        with open(args.raw_out, 'rb') as file:
            print('BERTs raw outputs loaded from', args.raw_out)
            data_for_all_items = pickle.load(file)
    n_layers = data_for_all_items[0].shape[0] # for convenience
    # The list data_for_all_items now contains, for each item, weights (n_layers, n_tokens, n_tokens)


    ## Take averages over groups of tokens
    if not args.ignore_groups and not len(items.groups) == 0:
        data_for_all_items = data_utils.merge_grouped_tokens(items, data_for_all_items, method=args.group_merger)
        # list with, for each item, weights (n_layers, n_groups, n_groups)


    ## Compute balances (though whether they will be plotted depends on args.balance)
    # (Re)compute balance: how much token influences minus how much is influenced
    balance_for_all_items = []
    for data_for_item in data_for_all_items:
        balance_for_item = []
        for data_for_layer in data_for_item:
            balance = np.nansum(data_for_layer - data_for_layer.transpose(), axis=1)
            balance_for_item.append(balance)
        balance_for_all_items.append(np.stack(balance_for_item))
    # At this point we have two lists of numpy arrays: for each item, the weights & balance across layers.


    ## TODO The following applies only if there are groups of tokens.
    ## TODO Otherwise, perhaps have option of plotting individual sentences + balance, but no comparison?

    ## Store the weights in dataframe together with original data
    # TODO All of this feels terribly hacky...
    # First flatten the numpy array per item
    data_for_all_items = [data.reshape(-1).tolist() for data in data_for_all_items]
    balance_for_all_items = [data.reshape(-1).tolist() for data in balance_for_all_items]
    # And then concatenate them (still per item per layer)
    data_and_balance_for_all_items = [array1 + array2 for array1, array2 in zip(data_for_all_items, balance_for_all_items)]
    # Concatenate onto original data rows (with each row repeated n_layers times)
    # original_items_times_nlayers = [a for l in [[i.to_list()] * n_layers for (_, i) in items.iterrows()] for a in l]
    data_for_dataframe = [a + b for a, b in zip([i.to_list() for (_, i) in items.iterrows()], data_and_balance_for_all_items)]
    # Multi-column to represent the (flattened) numpy arrays in a structured way
    multi_columns = pd.MultiIndex.from_tuples([(c, '', '', '') for c in items.columns] + [('weights', l, g1, g2) for l in range(n_layers) for g1 in items.groups for g2 in items.groups] + [('balance', l, g, '') for l in range(n_layers) for g in items.groups])

    df = pd.DataFrame(data_for_dataframe, index=items.index, columns=multi_columns)
    # Dataframe with three sets of columns: columns from original dataframe, weights (as extracted from BERT), and the balance computed from them

    ## Compute means over attention weights across all conditions (easy because they're flattened)
    # df_means = df.groupby(items.factors).mean()
    # print(df.groupby(items.factors).describe()) # TODO group columns?

    print(df)

    ## Restrict attention to the factors of interest:
    df_means = df.groupby(args.factors).mean()

    print(df_means)

    ## Print a quick text summary of main results, significance tests, etc.
    # TODO implement this here :)


    ## Time to create some plots!

    # Compute a list that contains, for each layer, a list of lists of matrices to be plotted.
    weights_to_plot_per_layer = create_dataframes_for_plotting(items, df_means, n_layers, args)

    # Compute and set global min/max to have same colormap extremes within or even across layers
    calibrate_for_colormap(weights_to_plot_per_layer, not args.no_global_colormap)

    # Create a plot for each layer (collect file paths)
    out_filepaths = []
    for weights_to_plot in weights_to_plot_per_layer:
        out_filepaths.append(plot(weights_to_plot, args))

    # Optionally, an animated gif :)
    if args.gif:
        out_filepath = "{}/{}{}{}{}.gif".format(args.out, args.method,
                                                        "-" + args.combine if args.combine != "no" else "",
                                                        "_normalized" if (
                                                                args.method == "attention" and args.normalize_heads) else "",
                                                        '_' + '-x-'.join(args.factors) if len(args.factors) > 0 else '')
        images = []
        for filename in out_filepaths:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_filepath, images, format='GIF', duration=.5)
        print("Saving movie:", out_filepath)




if __name__ == "__main__":
    main()

