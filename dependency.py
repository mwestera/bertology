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
import utils

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
                                                         ('_'+args.n_items) if args.n_items is not None else '')
    need_BERT = True
    if os.path.exists(args.raw_out):
        if input('Raw output file exists. Overwrite? (N/y)') != "y":
            need_BERT = False

    ## Set up tokenizer, data
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=("uncased" in args.bert))
    items = parse_data(args.data, tokenizer, max_items=args.n_items)

    print(len(items), 'items')

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
        data_for_all_items = average_for_token_groups(items, data_for_all_items)
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


def parse_data(data_path, tokenizer, max_items=None):
    """
    Turns a .csv file with some special markup of 'token groups' into a dataframe.
    :param data_path:
    :param tokenizer: BERT's own tokenizer
    :return: pandas DataFrame with different factors, the sentence, tokenized sentence, and token group indices as columns
    """

    items = []
    num_factors = None
    max_group_id = 0

    # Manual checking and parsing of first line (legend)
    with open(data_path) as f:
        legend = f.readline()
        if not legend.startswith("#"):
            print("WARNING: Legend is missing from the data. Using boring group and factor labels instead.")
            group_legend = None
            factor_legend = None
        else:
            legend = [l.strip() for l in legend.strip('#').split(',')]
            group_legend = {}
            factor_legend = {}
            for term in legend:
                if term.startswith('|'):
                    ind, term = term.strip('|').split(' ')
                    group_legend[int(ind)] = term
                else:
                    factor_legend[len(factor_legend)] = term

    # Now read the actual data
    reader = csv.reader(open(data_path), skipinitialspace=True)
    for row in filter(lambda row: not row[0].startswith('#'), reader):
        num_factors = len(row)-1    # Num rows minus the sentence itself
        group_to_token_ids = {}  # Map token-group numbers to token positions
        sentence = ""
        total_len = 1   # Take mandatory CLS symbol into account

        # go through the sentence group by group (separated by | )
        for each_part in (' '+row[-1]).strip('|').split('|'):   # Cheap fix to avoid unintended groups for sentence starting with number
            first_char = each_part[0]
            if first_char.isdigit():
                group_id = int(first_char)
                each_part = each_part[1:].strip()
                max_group_id = max(max_group_id, group_id)
            tokens = tokenizer.tokenize(each_part)
            # If group has a number, remember this group for plotting etc.
            if first_char.isdigit():
                if group_id in group_to_token_ids:
                    group_to_token_ids[group_id].extend(list(range(total_len, total_len + len(tokens))))
                else:
                    group_to_token_ids[group_id] = list(range(total_len, total_len + len(tokens)))
            total_len += len(tokens)
            sentence += each_part.strip() + ' '

        # collect token group ids in a list instead of dict, for inclusion in the final DataFrame

        if len(group_to_token_ids) == 0:
            token_ids_list = []
        else:
            token_ids_list = [[] for _ in range(max(group_to_token_ids)+1)]
            for key in group_to_token_ids:
                token_ids_list[key] = group_to_token_ids[key]

        # create data row
        items.append(row[:-1] + [sentence.strip()] + [' '.join(['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]'])] + token_ids_list)

        if max_items is not None and len(items) >= max_items:
            break

    # Make all rows the same length
    row_length = num_factors + 2 + max_group_id + 1
    items = [item + [[] for _ in range(row_length - len(item))] for item in items]

    # If no legend was given, infer legends with boring names from the data itself
    if group_legend is None:
        group_names = ['g{}'.format(i) for i in range(max_group_id + 1)]
    else:
        group_names = [group_legend[key] for key in group_legend]
    if factor_legend is None:
        factor_names = ['f{}'.format(i) for i in range(num_factors)]
    else:
        factor_names = [factor_legend[key] for key in factor_legend]

    # Remove empty list of groups if there are no groups
    if len(group_names) == 0:
        items = [item[:-1] for item in items]

    # Create dataframe with nice column names
    columns = factor_names + ['sentence'] + ['tokenized'] + group_names
    items = pd.DataFrame(items, columns=columns)

    # Add a bunch of useful metadata to the DataFrame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        items.num_factors = num_factors
        items.factors = factor_names
        items.num_groups = max_group_id + 1
        items.groups = group_names
        items.levels = {factor: items[factor].unique().tolist() for factor in items.factors}
        # # following is bugged: not all combination needs to exist in the data
        # items.conditions = list(itertools.product(*[items.levels[factor] for factor in items.factors]))
        items.conditions = list(set([tuple(l) for l in items[items.factors].values]))

    return items


def average_for_token_groups(items, data_for_all_items):
    """
    Takes weights matrix per item per layer, and averages rows and columns based on desired token groups.
    :param items: dataframe as read from example.csv
    :param data_for_all_items: list of numpy arrays with attention/gradients extracted from BERT
    :return: list of (for each item) a numpy array layer x num_groups x num_groups
    """
    data_for_all_items2 = []

    for (_, each_item), weights_per_layer in zip(items.iterrows(), data_for_all_items):

        # TODO Ideally this would be done still on cuda
        data_per_layer = []
        for m in weights_per_layer:
            # Group horizontally
            grouped_weights_horiz = []
            for group in items.groups:
                grouped_weights_horiz.append(m[each_item[group]].mean(axis=0))
            grouped_weights_horiz = np.stack(grouped_weights_horiz)

            # Group the result also vertically
            grouped_weights = []
            for group in items.groups:
                grouped_weights.append(grouped_weights_horiz[:, each_item[group]].mean(axis=1))
            grouped_weights = np.stack(grouped_weights).transpose()  # transpose to restore original order

            # store
            data_per_layer.append(grouped_weights)

        data_for_all_items2.append(np.stack(data_per_layer))

    return data_for_all_items2


if __name__ == "__main__":
    main()

