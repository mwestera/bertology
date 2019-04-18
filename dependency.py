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

from tqdm import tqdm

from scipy.stats.stats import pearsonr
from scipy.stats import combine_pvalues

parser = argparse.ArgumentParser(description='e.g., experiment.py data/example.csv')
parser.add_argument('data', type=str,
                    help='Path to data file (typically .csv).')
parser.add_argument('--n_items', type=int, default=None,
                    help='Max number of items from dataset to consider.')
parser.add_argument('--out', type=str, default=None,
                    help='Output directory for plots (default: creates a new /temp## folder)')
parser.add_argument('--raw_out', type=str, default=None,
                    help='Output directory for raw BERT outputs, pickled for efficient reuse.')
parser.add_argument('--trees_out', type=str, default=None,
                    help='Output directory for constructed spanning trees, pickled for efficient reuse.')
parser.add_argument('--pearson_out', type=str, default=None,
                    help='Output directory for computed pearson coefficients and p-values, pickled for efficient reuse.')
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
parser.add_argument('--transpose', action="store_true",
                    help='To transpose the matrix before computing spaning trees, i.e., assume info flows from children to head.')
parser.add_argument('--bert', type=str, default='bert-base-cased',
                    help='Which BERT model to use (default bert-base-cased; not sure which are available)')
parser.add_argument('--factors', type=str, default=None,
                    help='Which factors to plot, comma separated like "--factors reflexivity,gender"; default: first 2 factors in the data')
parser.add_argument('--balance', action="store_true",
                    help='To compute and plot balances, i.e., how much a token influences minus how much it is influenced.')
parser.add_argument('--cuda', action="store_true",
                    help='To use cuda.')
parser.add_argument('--no_overwrite', action="store_true",
                    help='To not overwrite existing files.')

# TODO Move all intermediate auxiliary files to output/aux/ directory.
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
            pass
        #     if args.no_overwrite:
        #         quit()
        #     if input('Output directory {} already exists. Risk overwriting files? N/y'.format(args.out)) != 'y':
        #         quit()
        else:
            os.mkdir(args.out)

    if args.raw_out is None:
        args.raw_out = 'data/auxiliary/{}_{}{}{}{}.pkl'.format(os.path.basename(args.data)[:-4],
                                                         args.method,
                                                         "_chain" if args.combine == "chain" else "", # cumsum can use same as no
                                                         '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                         ('_'+str(args.n_items)) if args.n_items is not None else '')

    if args.trees_out is None:
        args.trees_out = 'data/auxiliary/{}_SPANNING_{}{}{}{}{}{}.pkl'.format(os.path.basename(args.data)[:-4],
                                                         args.method,
                                                         '_' + args.combine if args.combine != "no" else "",
                                                         '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                         ('_'+str(args.n_items)) if args.n_items is not None else '',
                                                          '_'+args.group_merger,
                                                           '_' + 'transpose' if args.transpose else '',
                                                          )

    if args.pearson_out is None:
        args.pearson_out = 'data/auxiliary/{}_PEARSON_{}{}{}{}{}{}.pkl'.format(os.path.basename(args.data)[:-4],
                                                                           args.method,
                                                                           '_' + args.combine if args.combine != "no" else "",
                                                                           '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                                           ('_' + str(args.n_items)) if args.n_items is not None else '',
                                                                           '_' + args.group_merger,
                                                                           '_' + 'transpose' if args.transpose else '',
                                                                           )

    print(args.trees_out)
    print(args.pearson_out)

    ## Do we need to apply BERT (anew)?
    apply_BERT = True
    if os.path.exists(args.raw_out):
        if args.no_overwrite:
            apply_BERT = False
        elif input('Raw output file exists. Overwrite? (N/y)') != "y":
            apply_BERT = False

    ## Do we need to compute spanning trees (anew)?
    compute_trees = True
    if os.path.exists(args.trees_out):
        if args.no_overwrite:
            compute_trees = False
        elif input('Trees output file exists. Overwrite? (N/y)') != "y":
            compute_trees = False

    ## Do we need to compute pearson coefficients (anew)?
    compute_pearson = True
    if os.path.exists(args.pearson_out):
        if args.no_overwrite:
            compute_pearson = False
        elif input('Pearson output file exists. Overwrite? (N/y)') != "y":
            compute_pearson = False


    ## Set up tokenizer, data
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=("uncased" in args.bert))
    items, dependency_trees = data_utils.parse_data(args.data, tokenizer, max_items=args.n_items, words_as_groups=True, dependencies=True)

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
        dirname += "_{}{}{}{}{}".format(args.method,
                                       "-" + args.combine if args.combine != "no" else "",
                                       "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                       '_' + '-x-'.join(args.factors) if len(args.factors) > 0 else '',
                                      "_transposed" if args.transpose else "",)
        args.out = os.path.join("output", dirname)
        os.mkdir(args.out)


    ## Apply BERT or, if available, load results saved from previous run
    if apply_BERT:
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


    ## If some computation needs to be done, we need to process the BERT outputs a bit
    if compute_trees or compute_pearson:

        print("Processing the data from BERT...")

        ## Take cumsum if needed (placed outside the foregoing, to avoid having to save/load separate file for this
        if args.combine == "cumsum":
            for i in range(len(data_for_all_items)):
                data_for_all_items[i] = np.cumsum(data_for_all_items[i], axis=0)

        ## Take averages over groups of tokens
        if not args.ignore_groups and not len(items.groups) == 0:
            data_for_all_items = data_utils.merge_grouped_tokens(items, data_for_all_items, method=args.group_merger)

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
        multi_columns = pd.MultiIndex.from_tuples([(c, '', '', '') for c in items.columns] + [('weights', l, g1, g2) for l in range(n_layers) for g1 in items.groups for g2 in items.groups] + [('balance', l, g, '') for l in range(n_layers) for g in items.groups], names=['', 'layer', 'in', 'out'])

        df = pd.DataFrame(data_for_dataframe, index=items.index, columns=multi_columns)
        # Dataframe with three sets of columns: columns from original dataframe, weights (as extracted from BERT & grouped), and the balance computed from them


    ## Apply BERT or, if available, load results saved from previous run
    if compute_trees:
        trees_df = analyze_by_spanning_trees(df, items, dependency_trees, n_layers, args)
        with open(args.trees_out, 'wb') as file:
            pickle.dump(trees_df, file)
            print('Trees and scores saved as', args.trees_out)
    else:
        with open(args.trees_out, 'rb') as file:
            print('Trees and scores loaded from', args.trees_out)
            trees_df = pickle.load(file)

    plot_tree_scores(trees_df, args)

    if compute_pearson:
        pearson_df = analyze_by_pearson_correlation(df, items, dependency_trees, n_layers, args)
        with open(args.pearson_out, 'wb') as file:
            pickle.dump(pearson_df, file)
            print('Pearson coefficients and p-values saved as', args.pearson_out)
    else:
        with open(args.pearson_out, 'rb') as file:
            print('Pearson coefficients and p-values loaded from', args.pearson_out)
            pearson_df = pickle.load(file)

    plot_pearson_scores(pearson_df, args)

    ## TODO write basic statistics to file?   using combine_pvalues

def analyze_by_pearson_correlation(df, items, dependency_trees, n_layers, args):

    pearson_for_all_items = []

    print("Computing pearson correlations...")

    for i, item in tqdm(df.iterrows(), total=len(df)):
        pearson_for_item = []

        dtree = dependency_trees[i]

        dmatrix = - tree_utils.arcs_to_distance_matrix(tree_utils.conllu_to_arcs(dtree.to_tree()))
        dmatrix_bidirectional = - tree_utils.arcs_to_distance_matrix(tree_utils.conllu_to_arcs(dtree.to_tree()), bidirectional=True)

        n_tokens = len(item['balance'][0])

        for layer in range(n_layers):

            matrix = item['weights'][layer].values.reshape(n_tokens, n_tokens).astype(float)
            # remove all nans
            matrix = matrix[~np.all(np.isnan(matrix), axis=1), :][:, ~np.all(np.isnan(matrix), axis=0)]
            if args.transpose:
                # Without transpose, the matrix[i,j] represents how much i influences j.
                # In a dependency tree arrows flow away from the root.
                # Hence, a maximum spanning tree on the matrix maximizes information flowing away from the root (= head).
                # Use args.transpose to hypothesize that info flows towards the root instead.
                matrix = matrix.transpose()

            ## Flatten all matrices for comparison # TODO Not sure if necessary
            matrix = matrix.reshape(-1)
            dmatrix = dmatrix.reshape(-1)
            dmatrix_bidirectional = dmatrix_bidirectional.reshape(-1)

            ## Compute pearson  # TODO Compute for interesting subsets of nodes too
            coef1, p1 = pearsonr(matrix[~np.isnan(dmatrix)], dmatrix[~np.isnan(dmatrix)])
            coef2, p2 = pearsonr(matrix, dmatrix_bidirectional)

            pearson_for_item.extend([coef1, p1, coef2, p2])

        pearson_for_all_items.append(pearson_for_item)

    ## Put results into a new dataframe
    columns = [('pearson', l, direct, x) for l in range(n_layers) for direct in ['directed', 'undirected'] for x in ['coefficient', 'p-value']]

    columns = pd.MultiIndex.from_tuples(columns, names=['result', 'layer', 'relations', 'measure'])
    pearson_df = pd.DataFrame(pearson_for_all_items, index=items.index, columns=columns)

    return pearson_df


def analyze_by_spanning_trees(df, items, dependency_trees, n_layers, args):
    scores = []
    trees = []

    print("Computing spanning trees...")

    for i, item in tqdm(df.iterrows(), total=len(df)):
        dtree = dependency_trees[i]
        n_tokens = len(item['balance'][0])
        scores.append([])
        trees.append([])
        for layer in range(n_layers):
            matrix = item['weights'][layer].values.reshape(n_tokens,n_tokens)
            if args.transpose:
                # Without transpose, the matrix[i,j] represents how much i influences j.
                # In a dependency tree arrows flow away from the root.
                # Hence, a maximum spanning tree on the matrix maximizes information flowing away from the root (= head).
                # Use args.transpose to hypothesize that info flows towards the root instead.
                matrix = matrix.transpose()
            # TODO cut the matrix to size
            arcs = tree_utils.matrix_to_arcs(matrix)

            # Obtain tree and compute scores
            wtree, wtree_value = tree_utils.max_sa_from_nodes(arcs, list(range(n_tokens)))
            wtree, wtree_value = tree_utils.arcs_to_tuples(wtree.values())
            # dtree_value = tree_utils.tree_value_from_matrix(dtree, matrix)
            scores[i].append(tree_utils.get_scores(wtree, dtree))
            trees[i].append(wtree)

    ## Put results into a new dataframe
    rows = []
    columns = [s for scorelist in [[('score', l, cat, measure) for cat in score_for_layer for measure in score_for_layer[cat]] for l, score_for_layer in enumerate(scores[0])] for s in scorelist]
    columns.extend([('tree', l, '', '') for l in range(0, n_layers)])
    for i, score in enumerate(scores):
        row = [s for scorelist in [[score_for_layer[cat][measure] for cat in score_for_layer for measure in score_for_layer[cat]] for score_for_layer in score] for s in scorelist]
        row.extend(trees[i])
        rows.append(row)
    columns = pd.MultiIndex.from_tuples(columns, names=['result', 'layer', 'relations', 'measure'])
    scores_df = pd.DataFrame(rows, index=items.index, columns=columns)

    return scores_df


def plot_tree_scores(scores_df, args):

    ## Stack for plotting
    scores_df = scores_df[['score']].stack().stack().stack().reset_index(level=[1, 2, 3])
    scores_df = scores_df[scores_df.measure != 'num_rels']

    plt.figure(figsize=(10, 8))
    sns.lineplot(x='layer', y='score', style='measure', hue='relations', data=scores_df)

    out_filepath = '{}/treescores_{}{}{}{}{}{}.png'.format(args.out,
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



def plot_pearson_scores(scores_df, args):

    ## Stack for plotting
    scores_df = scores_df[['pearson']].stack().stack().stack().reset_index(level=[1, 2, 3])
    scores_df = scores_df[scores_df.measure != 'p-value']

    plt.figure(figsize=(10, 8))
    sns.lineplot(x='layer', y='pearson', hue='relations', data=scores_df)

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

