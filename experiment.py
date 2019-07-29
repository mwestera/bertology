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

import statsmodels.api as sm

import os
import warnings

import argparse

import pickle

import interface_BERT

import data_utils

from scipy.stats import ttest_ind, ttest_rel

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
parser.add_argument('--estimator', type=str, default="mean", choices=["none", "mean"],
                    help='for lineplots')
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
parser.add_argument('--track', type=str, default=None,
                    help='Which tokens/token groups to track; single tokens to track balance; pairs (,) to track their weight. Items separated by ;, e.g., term1,term2;term3. Default tracks all groups.')
parser.add_argument('--no_global_colormap', action="store_true",
                    help='Whether to standardize plot coloring across plots ("global"); otherwise only per plot (i.e., per layer)')
parser.add_argument('--balance', action="store_true",
                    help='To compute and plot balances, i.e., how much a token influences minus how much it is influenced.')
parser.add_argument('--cuda', action="store_true",
                    help='To use cuda.')
parser.add_argument('--no_overwrite', action="store_true",
                    help='To not overwrite existing files.')
parser.add_argument('--heatmap', action="store_true",
                    help='To output pretty heat maps.')
parser.add_argument('--prefix', type=str, default='',
                    help='Prefix for saved (non-auxiliary) output files like plots.')

# TODO: Allow a more 'localist' analysis mode? See if there are any single neurons disproportionately reactive to a distinction, by looking at gradients.

# TODO: perhaps it's useful to allow plotting means over layers; sliding window-style? or chaining but with different starting points?
# TODO: I got an error when running on example.csv with --n_items 1 or even 2.

# TODO: Compose output file basename more centrally; add PREFIX to the folder name.

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
            # if input('Output directory {} already exists. Risk overwriting files? N/y'.format(args.out)) != 'y':
            #    quit()
            pass
        else:
            os.mkdir(args.out)

    if args.prefix != '':
        args.prefix += '_'

    args.estimator = None if args.estimator.lower() == "none" else args.estimator

    if args.raw_out is None:
        args.raw_out = 'output/auxiliary/{}_{}{}{}{}.pkl'.format(os.path.basename(args.data)[:-4],
                                                         args.method,
                                                         "_chain" if args.combine == "chain" else "", # cumsum can use same as no
                                                         '_norm' if args.method == 'attention' and args.normalize_heads else '',
                                                         ('_'+str(args.n_items)) if args.n_items is not None else '')

    ## Do we need to apply BERT (anew)?
    apply_BERT = True
    if os.path.exists(args.raw_out):
        if args.no_overwrite:
            apply_BERT = False
        elif input('Raw output file exists. Overwrite? (N/y)') != "y":
            apply_BERT = False


    ## Set up tokenizer, data
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=("uncased" in args.bert))
    items = data_utils.parse_data(args.data, tokenizer, max_items=args.n_items)

    print(len(items), 'items')

    ## Store for convenience
    args.factors = args.factors or items.factors[:2]    # by default use the first two factors from the data

    # Fill in args.track default depending on data
    if args.track is None:
        args.track = [[g] for g in items.groups]
    elif args.track is not None:
        args.track = [[a.strip() for a in b.split(',')] for b in args.track.split(";")]

    ## Now that args.factors is known, finally choose output directory
    if args.out is None:
        dirname = 'temp'
        out_idx = 0
        if not os.path.exists("output"):
            os.mkdir('output')
        if not os.path.exists("output/auxiliary/"):
            os.mkdir('output/auxiliary')
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

    ## Take cumsum if needed (placed outside the foregoing, to avoid having to save/load separate file for this
    if args.combine == "cumsum":
        for i in range(len(data_for_all_items)):
            data_for_all_items[i] = np.cumsum(data_for_all_items[i], axis=0)


    ## Compute balances (though whether they will be plotted depends on args.balance)   # TODO rename 'balance' something else, or allow switching between sum and balance...
    # (Re)compute balance: how much token influences minus how much is influenced
    balance_for_all_items = []
    for data_for_item in data_for_all_items:
        balance_for_item = []
        for data_for_layer in data_for_item:
            # I also tried setting the diagonal to np.nan; doesn't make a difference.
            balance = np.nansum(data_for_layer, axis=1) # - data_for_layer.transpose(), axis=1)
            balance_for_item.append(balance)
        balance_for_all_items.append(np.stack(balance_for_item))
    # At this point we have two lists of numpy arrays: for each item, the weights & balance across layers.


    ## Take averages over groups of tokens
    if not args.ignore_groups and not len(items.groups) == 0:
        data_for_all_items = data_utils.merge_grouped_tokens(items, data_for_all_items, method=args.group_merger)
        balance_for_all_items = data_utils.merge_grouped_tokens(items, balance_for_all_items, method=args.group_merger)
        # list with, for each item, weights (n_layers, n_groups, n_groups)

    ## TODO If no groups, perhaps have option of plotting individual sentences + balance, but no comparison?


    ## Store the weights and balance in dataframe      # TODO All of this feels terribly hacky...

    # First flatten the numpy array per item
    data_for_all_items = [data.reshape(-1).tolist() for data in data_for_all_items]
    balance_for_all_items = [data.reshape(-1).tolist() for data in balance_for_all_items]
    # And then concatenate them (still per item per layer)
    data_and_balance_for_all_items = [array1 + array2 for array1, array2 in zip(data_for_all_items, balance_for_all_items)]
    # Concatenate onto original data rows (with each row repeated n_layers times)
    # original_items_times_nlayers = [a for l in [[i.to_list()] * n_layers for (_, i) in items.iterrows()] for a in l]
    data_for_dataframe = data_and_balance_for_all_items # [a + b for a, b in zip([i.to_list() for (_, i) in items.iterrows()], data_and_balance_for_all_items)]
    # Multi-column to represent the (flattened) numpy arrays in a structured way
    multi_columns = pd.MultiIndex.from_tuples([('weights', l, g1, g2) for l in range(n_layers) for g1 in items.groups for g2 in items.groups] + [('balance', l, g, '') for l in range(n_layers) for g in items.groups], names=['result', 'layer', 'in', 'out'])
    # [('', '', '', c) for c in items.columns] +
    df_all_values = pd.DataFrame(data_for_dataframe, index=items.index, columns=multi_columns)
    # Dataframe with three sets of columns: columns from original dataframe, weights (as extracted from BERT), and the balance computed from them


    ## Track the behavior of particular (groups of) tokens across conditions and layers

    df_for_tracking = create_dataframe_for_tracking(items, df_all_values, n_layers, args)
    # this df contains n_layers rows per item, with as columns the original factors and the scores for all items to track.

    # create and save a single plot for all factors and token groups
    plot_tracked_tokens(df_for_tracking, args)    # TODO maybe allow for multiple plots, e.g., args.plot_per_factor?)

    # compute and write some statistics to a file
    stats_tracked_tokens(items, df_for_tracking, args)


    ## Optionally output heat maps for all tokens/token groups across conditions and layers:

    if args.heatmap:

        df_means = (df_all_values.groupby(args.factors) if args.factors else df_all_values).mean()

        # Compute a list that contains, for each layer, a list of lists of matrices to be plotted.
        weights_to_plot_per_layer = create_dataframes_for_heatmap(items, df_means, n_layers, args)

        # Compute and set global min/max to have same colormap extremes within or even across layers
        calibrate_for_colormap(weights_to_plot_per_layer, not args.no_global_colormap)

        # Create a plot for each layer (collect file paths)
        out_filepaths = []
        for weights_to_plot in weights_to_plot_per_layer:
            out_filepaths.append(heatmap(weights_to_plot, args))

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
            print("Saving award-winning movie:", out_filepath)


##################################################
### A bunch of methods for the 'tracking' particular tokens/token groups.
##################################################

def create_dataframe_for_tracking(items, df, n_layers, args):
    # First copy the 'factors' columns from the original data, each row repeated n_layers times to accommodate scores per layer.
    tracking_df = pd.DataFrame([a for b in [[item[items.factors]] * n_layers for _, item in items.iterrows()] for a in b])

    for i,to_track in enumerate(args.track):

        # Tokens (or token groups) to track can be pairs (weight between them) or single tokens (balance)
        if len(to_track) == 1:
            score = 'balance'
            data = df.loc[:, ('balance', slice(None), to_track[0])]
        else:
            score = 'weights'
            data = df.loc[:, ('weights', slice(None), to_track[0], to_track[1])]

        # Move layers from column index to row index
        data = data.stack(level=1, dropna=False).reset_index(level=[1])

        # Remove unused levels in columns multiindex
        data.columns = data.columns.droplevel([1, 2])

        # Rename 'weight' or 'balance' to the tokens to track
        data.rename(columns={score: '>'.join(to_track)}, inplace=True)

        # concatenate with columns already in tracking_df (initially: factors from original items):
        tracking_df = pd.concat([tracking_df, data[(['layer'] if i==0 else []) + ['>'.join(to_track)]]], axis=1)

    return tracking_df


def plot_tracked_tokens(df, args):


    plt.figure(figsize=(20, 12))

    df['item'] = df.index
    kwargs = {'units': 'item', 'alpha': .3} if args.estimator is None else {}

    for to_track in args.track:
        ax = sns.lineplot(x="layer", y='>'.join(to_track),
                      hue=None if len(args.track) > 1 else args.factors[0] if len(args.factors) > 0 else None,
                      style=args.factors[0] if len(args.factors) > 0 and len(args.track) > 1 else args.factors[1] if len(args.factors) > 1 else None,
                      data=df, label='>'.join(to_track), estimator=args.estimator, **kwargs)
    # ax = sns.lineplot(x="layer", y=score, hue=args.factors[0] if len(args.factors) > 0 else None, style=args.factors[1] if len(args.factors) > 1 else None, data=data, label=to_track)
    ax.set_title("Tracking {}{} across layers".format(args.method, (" (" + args.combine + ")") if args.combine is not "no" else ""))
    ax.set_ylabel("{}{}".format(args.method, (" (" + args.combine + ")") if args.combine is not "no" else ""))

    # TODO Fix the legend.
    ax.legend()
    handles, labels = ax.get_legend_handles_labels()
    lgd = dict(zip(labels, handles))
    ax.legend(lgd.values(), lgd.keys())
    # plt.legend()

    # TODO Also an overall mean plot on the side

    out_filepath = "{}/{}track_{}{}{}.png".format(args.out,
                                                args.prefix,
                                                  args.method,
                                                 "-"+args.combine if args.combine != "no" else "",
                                                 "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                                )
                                                 # ';'.join([','.join(to_track) for to_track in args.track]))
    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)

    return out_filepath


def stats_tracked_tokens(items, tracking_df, args):
    """
    Applies statistical test per factor level (if any) and per token group:
    TODO What about interactions of factors into account? Do a multivariate regression instead?
    TODO T-test? Can I assume normal distributions? What about Spearman correlation across layers instead?
    TODO What about model-fitting based on hypotheses about computation?
    :param items:
    :param tracking_df:
    :param args:
    :return:
    """

    out_statspath = "{}/{}stats_{}{}{}.tsv".format(args.out,
                                                  args.prefix,
                                                  args.method,
                                                  "-" + args.combine if args.combine != "no" else "",
                                                  "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                                  )

    with open(out_statspath, 'w+') as file:

        if len(args.factors) == 0:

            pass

            ## If no factors are given, let X be the tracked token? Nah, this doesn't make sense: OLS for a categorical X.
            # print(tracking_df[:10])
            #
            # tracking_df['weight'] = np.nan
            # tracking_df['tracked'] = ""
            # for to_track in args.track:
            #     tracking_df['weight'] = tracking_df['weight'].fillna(tracking_df[to_track])
            #     tracking_df['']
            #
            #
            # tracking_df[].fillna()
            #
            # X = tracking_df[['>'.join(to_track) for to_track in args.track]]
            # X = sm.add_constant(X)

        else:
            for to_track in args.track:
                X = tracking_df[args.factors]
                X = sm.add_constant(X)
                if 'coref' in X:
                    X['distance'] = tracking_df['distance']
                    X['coref'] = (X['coref'] == 'coref').astype(int)
                    X['coref*distance'] = X['coref'] * X['distance']

                y = tracking_df['>'.join(to_track)]

                model = sm.OLS(y, X).fit()
                predictions = model.predict()
                with open(out_statspath, 'a') as file:
                    file.write('\n\n'+str(model.summary())+'\n\n')


    ttest_results = []
    for factor in args.factors if args.factors else [None]:
        for l1, level1 in enumerate(items.levels[factor] if args.factors else [None]):
            for l2, level2 in enumerate(items.levels[factor] if args.factors else [None]):
                if l1 <= l2:
                    for g1, group1 in enumerate(args.track) if args.track else [None]:
                        for g2, group2 in enumerate(args.track) if args.track else [None]:
                            if g1 <= g2 and not (l1 == l2 and g1 == g2):
                                if factor is not None and group1 is not None:
                                    t_tested = tracking_df.dropna().groupby('layer').apply(
                                        lambda df: ttest_ind(df.where(df[factor] == level1).dropna()['>'.join(group1)],
                                                             df.where(df[factor] == level2).dropna()['>'.join(group2)],
                                                             equal_var=False))
                                    t_tested.loc['overall'] = ttest_ind(
                                        tracking_df.where(tracking_df[factor] == level1).dropna()['>'.join(group1)],
                                        tracking_df.where(tracking_df[factor] == level2).dropna()['>'.join(group2)],
                                        equal_var=False)
                                elif group1 is not None:
                                    t_tested = tracking_df.dropna().groupby('layer').apply(
                                        lambda df: ttest_ind(df['>'.join(group1)],
                                                             df['>'.join(group2)],
                                                             equal_var=False))
                                    t_tested.loc['overall'] = ttest_ind(tracking_df['>'.join(group1)].dropna(),
                                                                        tracking_df['>'.join(group2)].dropna(),
                                                                        equal_var=False)
                                elif factor is not None:
                                    t_tested = tracking_df.dropna().groupby('layer').apply(
                                        lambda df: ttest_ind(df.where(df[factor] == level1).dropna(),
                                                             df.where(df[factor] == level2).dropna(),
                                                             equal_var=False))
                                    t_tested.loc['overall'] = ttest_ind(
                                        tracking_df.where(tracking_df[factor] == level1).dropna(),
                                        tracking_df.where(tracking_df[factor] == level2).dropna(),
                                        equal_var=False)

                                # Add some convenient metadata:
                                t_tested.a = ' '.join(([level1] if level1 is not None else []) + (
                                ['>'.join(group1)] if group1 is not None else []))
                                t_tested.b = ' '.join(([level2] if level2 is not None else []) + (
                                ['>'.join(group2)] if group2 is not None else []))
                                ttest_results.append(t_tested)

    with open(out_statspath, 'a+') as file:
        for t_tested in ttest_results:
            file.write('\n\nt-test ' + t_tested.a + ' vs. ' + t_tested.b + '\n')
            for l, row in t_tested.iteritems():
                file.write('\t'.join([str(l), str(row[0].round(2)), str(row[1].round(5))])+'\n')

    print("Stats written to", out_statspath)


##################################################
### A bunch of methods for the heatmap
##################################################

def create_dataframes_for_heatmap(items, df_means, n_layers, args):
    """
    :param items: as read from data/example.csv
    :param df_means: means as resulting from .groupby() the relevant factors.
    :param n_layers: TODO can be inferred from df_means... omit here?
    :param args: command line arguments; contain some useful things
    :return:
    """
    ## Prepare for plotting
    # Determine overall layout of the plots, adding <DIFF> in case difference plot is to be included   # TODO I'm probably mixing up vertical and horizontal here (though plots are labels so interpretation is ok)
    levels_horiz = items.levels[args.factors[0]] if len(args.factors) >= 1 else [None]
    levels_vert = items.levels[args.factors[1]] if len(args.factors) >= 2 else [None]
    if len(levels_horiz) == 2 and not args.no_diff_plots:  # if two levels, also compute difference       # TODO These <DIFF>s are ugly... though at least it works.
        levels_horiz.append('<DIFF>')
    if len(levels_vert) == 2 and not args.no_diff_plots:  # if two levels, also compute difference
        levels_vert.append('<DIFF>')

    # This list will contain all weights matrices to-be-plotted (computed altogether, prior to plotting, in order to compute global max/min for colormap...)
    weights_to_plot_per_layer = []
    for l in range(n_layers):

        # For each layer, there will be multiple weights matrices due to different levels per factor
        data_to_plot = [[[] for _ in levels_vert] for _ in levels_horiz]

        # Loop through rows and columns of the multiplot-to-be:
        for h, level_horiz in enumerate(levels_horiz):
            for v, level_vert in enumerate(levels_vert):
                # Things are easy if it's a difference plot of plots we've already computed before:
                if level_horiz != "<DIFF>" and level_vert == "<DIFF>":
                    data = data_to_plot[h][0] - data_to_plot[h][1]
                    data.difference = True
                elif level_horiz == "<DIFF>" and level_vert != "<DIFF>":
                    data = data_to_plot[0][v] - data_to_plot[1][v]
                    data.difference = True
                elif level_horiz == "<DIFF>" and level_vert == "<DIFF>":
                    data = data_to_plot[0][0] - data_to_plot[1][1]
                    data.difference = True
                # More work if it's an actual weights plot:
                else:
                    data = df_means.loc[(level_horiz, level_vert),(slice(None),l)] if level_vert is not None else (df_means.loc[level_horiz,(slice(None),l)] if level_horiz is not None else df_means[(slice(None),l)])
                    data.difference = False

                # Some convenient metadata (used mostly when creating plots)
                # It's a lot safer that each dataframe carries its own details with it in this way.
                data.level_horiz = level_horiz
                data.level_vert = level_vert
                data.max_for_colormap = data['weights'].max().max()
                data.min_for_colormap = data['weights'].min().min()
                data.balance_max_for_colormap = data['balance'].max().max()
                data.balance_min_for_colormap = data['balance'].min().min()
                data.layer = l

                # Add dataframe to designated position
                data_to_plot[h][v] = data

        # Save the weights to plot for this particular layer
        weights_to_plot_per_layer.append(data_to_plot)

    # The list weights_to_plot_per_layer now contains, for each layer, a list of lists of weights matrices.
    return weights_to_plot_per_layer


def heatmap(data, args):
    """
    Output a single image file, typically containing several plots, depending on which factors to cross
    and whether to include a difference plot.
    :param data: list of lists of weights dataframes; each DF will be plotted as a single heatmap.
    :param args: the command line arguments; they contain some further settings.
    :return: output file path
    """
    # Let's plot!
    f = plt.figure(figsize=(4 * len(data) + 1, 4 * len(data[0])))
    gs0 = gridspec.GridSpec(len(data[0]), len(data), figure=f, hspace=.6, wspace=.6)

    # plt.subplots_adjust(wspace=.6, top=.9)
    f.suptitle("{}{} {}layer {}".format(args.method, " ("+args.combine+")" if args.combine != "no" else "", "up to " if args.combine != "no" else "", data[0][0].layer), size=16)

    for c, col in enumerate(data):

        for r, weights in enumerate(col):

            tokens = weights['balance'].index.get_level_values(0)

            axis = gs0[c, r] if weights.level_vert is not None else gs0[c] if weights.level_horiz is not None else gs0[0]

            subgs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axis, height_ratios=(len(tokens), 1 if args.balance else .0001), width_ratios=(.95, .05))

            ax_main = plt.Subplot(f, subgs[0,0])
            f.add_subplot(ax_main)
            ax_main_cbar = plt.Subplot(f, subgs[0,1])
            f.add_subplot(ax_main_cbar)
            if args.balance:
                ax_balance = plt.Subplot(f, subgs[1, 0])
                f.add_subplot(ax_balance)
                ax_balance_cbar = plt.Subplot(f, subgs[1,1])
                f.add_subplot(ax_balance_cbar)

            sns.heatmap(weights['weights'].values.reshape(len(tokens), len(tokens)),
                         xticklabels=tokens,
                         yticklabels=tokens,
                         vmin=weights.min_for_colormap,
                         vmax=weights.max_for_colormap,
                         center=0 if weights.difference else None,
                         linewidth=0.5,
                         ax=ax_main,
                         cbar=True,
                         cbar_ax=ax_main_cbar,
                         cmap="RdBu" if weights.difference else "Greys",
                         square=False,      # TODO See if I can get the square to work...
#                        cbar_kws={'shrink': .5},
                         label='small')
            # ax_main.set_xlabel('...to layer {}'.format(weights.layer))
            # ax_main.set_ylabel('From previous layer...' if args.combine == "no" else "From initial embeddings...")
            if weights.difference:
                ax_main.set_title("difference")
            else:
                ax_main.set_title('{} & {}'.format(weights.level_horiz, weights.level_vert) if weights.level_vert is not None else (weights.level_horiz or ""))

            map_img = mpimg.imread('figures/arrow-down.png')

            ax_main.imshow(map_img,
                        aspect=ax_main.get_aspect(),
                        extent=ax_main.get_xlim() + ax_main.get_ylim(),
                        zorder=3,
                        alpha=.2)

            plt.setp(ax_main.get_yticklabels(), rotation=0)

            if args.balance:
                sns.heatmap(weights['balance'].values.reshape(1, len(tokens)),
                        xticklabels=["" for _ in tokens],
                        yticklabels=['Balance'],
                        ax=ax_balance,
                        center=0,
                        vmin = -round(weights.balance_max_for_colormap, 2),
                        vmax = round(weights.balance_max_for_colormap, 2),
                        linewidth=0.5,
                        cmap="PiYG" if weights.difference else "PiYG",
                        cbar=True,
                        cbar_ax=ax_balance_cbar,
                        # cbar_kws={'shrink': .5}, # makes utterly mini...
                        label='small',
                        cbar_kws=dict(ticks=[-round(weights.balance_max_for_colormap, 2), 0, round(weights.balance_max_for_colormap, 2)], format="%.2f")
                        )
                plt.setp(ax_balance.get_yticklabels(), rotation=0)
                ax_balance.xaxis.tick_top()
                # ax_main.set_xlabel('...to layer {}'.format(weights.layer))

    gs0.tight_layout(f, rect=[0, 0.03, 1, 0.95])
#    f.subplots_adjust(top=1.0-(1.0 / (4 * len(weights_to_plot) + 1)))

    out_filepath = "{}/{}{}{}{}_layer{}.png".format(args.out, args.method,
                                                     "-"+args.combine if args.combine != "no" else "",
                                                     "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                                     '_'+'-x-'.join(args.factors) if len(args.factors) > 0 else '', data[0][0].layer)
    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)
    # pylab.show()

    return out_filepath


def calibrate_for_colormap(weights_to_plot_per_layer, global_colormap):
    """
    Calibrates meta-info of weights dataframes, namely, the max_for_plot and min_for_plot, which will determine the coloration.
    If colormap == "global", global max and min are computed across all layers; if "layer", max and min are computed within layer.
    :param weights_to_plot_per_layer: list of (list of lists of dataframes) to be plotted.
    :param colormap: either "layer" or "global"
    :return: nothing; two fields of the weights dataframes are modified in-place.
    """
    layer_maxes = []
    layer_mins = []
    layer_max_diffs = []

    layer_balance_maxes = []
    layer_balance_mins = []
    layer_balance_max_diffs = []

    ## First compute and set max and min per layer
    for l in range(len(weights_to_plot_per_layer)):
        weights = [w for row in weights_to_plot_per_layer[l] for w in row]

        layer_max = max([w.max_for_colormap for w in weights if not w.difference])
        layer_min = min([w.min_for_colormap for w in weights if not w.difference])
        layer_max_diff = max([max(abs(w.max_for_colormap), abs(w.min_for_colormap)) for w in weights if w.difference] + [0])

        layer_balance_max = max([w.balance_max_for_colormap for w in weights if not w.difference])
        layer_balance_min = min([w.balance_min_for_colormap for w in weights if not w.difference])
        layer_balance_max_diff = max([max(abs(w.balance_max_for_colormap), abs(w.balance_min_for_colormap)) for w in weights if w.difference] + [0])

        # Default: "layer" calibration, set the new max and min values.
        for w in weights:
            if w.difference:
                w.max_for_colormap = layer_max_diff
                w.min_for_colormap = -layer_max_diff
                w.balance_max_for_colormap = layer_balance_max_diff
                w.balance_min_for_colormap = -layer_balance_max_diff
            else:
                w.max_for_colormap = layer_max
                w.min_for_colormap = layer_min
                w.balance_max_for_colormap = layer_balance_max
                w.balance_min_for_colormap = layer_balance_min

        layer_maxes.append(layer_max)
        layer_mins.append(layer_min)
        layer_max_diffs.append(layer_max_diff)

        layer_balance_maxes.append(layer_balance_max)
        layer_balance_mins.append(layer_balance_min)
        layer_balance_max_diffs.append(layer_balance_max_diff)


    ## If global_colormap requested, compute overall max and min (across layers) and add these values to dataframes
    if global_colormap:
        overall_max = max(layer_maxes)
        overall_min = min(layer_mins)
        overall_max_diff = max(layer_max_diffs)

        overall_balance_max = max(layer_balance_maxes)
        overall_balance_min = min(layer_balance_mins)
        overall_balance_max_diff = max(layer_balance_max_diffs)

        weights = [w for layer in weights_to_plot_per_layer for row in layer for w in row]
        for w in weights:
            if w.difference:
                w.max_for_colormap = overall_max_diff
                w.min_for_colormap = -overall_max_diff
                w.balance_max_for_colormap = overall_balance_max_diff
                w.balance_min_for_colormap = -overall_balance_max_diff
            else:
                w.max_for_colormap = overall_max
                w.min_for_colormap = overall_min
                w.balance_max_for_colormap = overall_balance_max
                w.balance_min_for_colormap = overall_balance_min


def suplabel(axis, label,label_prop={"size": 16},
             labelpad=5,
             ha='center',va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    Thanks to user KYC on StackOverflow.com. https://stackoverflow.com/questions/6963035/pyplot-axes-labels-for-subplots
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = pylab.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin,ymin = min(xmin),min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation=90.
        x = xmin-float(labelpad)/dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad)/dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    pylab.text(x,y,label,rotation=rotation,
               transform=fig.transFigure,
               ha=ha,va=va,
               **label_prop)


if __name__ == "__main__":
    main()

