from bertviz import attention, visualization
from bertviz.pytorch_pretrained_bert import BertModel, BertTokenizer

import numpy as np
import pandas as pd
import csv
import itertools

import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pylab as pylab
import imageio

import os
import warnings

import argparse


parser = argparse.ArgumentParser(description='e.g., main.py data/example.csv')
parser.add_argument('data', type=str,
                    help='Path to data file (typically .csv).')
parser.add_argument('--out', type=str, default=None,
                    help='Output directory for plots (default: creates a new /temp## folder)')
parser.add_argument('--method', type=str, default='MAT',
                    help='MAT (mean attention per token) or PAT (percolated attention per token); default: MAT')
parser.add_argument('--no_layernorm', action="store_true",
                    help='To prevent applying normalization per attention head.')
parser.add_argument('--no_groups', action="store_true",
                    help='To ignore groupings of tokens in the input data, and compute/plot per token.')
parser.add_argument('--no_transpose', action="store_true",
                    help='To NOT transpose the weights matrix for plotting; by default it is transposed, plotting as "rows influenced by cols" (otherwise: rows influencing cols).')
parser.add_argument('--no_difs', action="store_true",
                    help='To NOT plot the differences between levels of a given factor.')
parser.add_argument('--gif', action="store_true",
                    help='To create animated gif of plots across layers.')
parser.add_argument('--bert', type=str, default='bert-base-cased',
                    help='Which BERT model to use (default bert-base-cased; not sure which are available)')
parser.add_argument('--factors', type=str, default=None,
                    help='Which factors to plot, comma separated like "--factors reflexivity,gender"; default: first 2 factors in the data')

# TODO Allow cumulative MAT too... CAT? cMAT?

def main():
    """
    To run this code with default settings and example data, do
       $ python main.py data/example.csv

    For more interpretable plots, look only at factor reflexivity (ignore gender) by doing:
      $ python main.py data/example.csv --factors reflexivity

    And for a little bonus, add --gif .

    This applies BERT to the data, extracts attention weights, and creates a number of plots.
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
    else:  # else write to /temp folder, though with increasing numeral to avoid overwriting
        args.out = 'output/temp'
        args.out_idx = 0
        while os.path.exists(args.out):
            args.out_idx += 1
            args.out = 'output/temp_{}'.format(args.out_idx)
        os.mkdir(args.out)


    ## Set up tokenizer, data and model
    tokenizer = BertTokenizer.from_pretrained(args.bert)  # TODO The tokenizer seems to get rid of casing; why? Is this an older BERT version?
    items = parse_data(args.data, tokenizer)
    model = BertModel.from_pretrained(args.bert)
    attention_visualizer = visualization.AttentionVisualizer(model, tokenizer)    # TODO Bypass the AttentionVisualizer code altogether and remove from repo; I'm not really using it.


    ## Store for convenience
    args.factors = args.factors or items.factors[:2]    # by default use the first two factors from the data
    n_layers = len(model.encoder.layer)


    ## Compute attention weights, one item at a time
    weights_for_all_items = []
    for _, each_item in items.iterrows():

        tokens_a, tokens_b, attention = attention_visualizer.get_viz_data(each_item['sentence'])
        all_tokens = tokens_a + tokens_b
        attention = attention.squeeze()

        weights_per_layer = (compute_PAT if args.method == "PAT" else compute_MAT)(attention, layer_norm=not args.no_layernorm)

        # TODO Put the following outside the current loop, inside its own (anticipating intermediary writing of results to disk)
        # Take averages over groups of tokens
        if not args.no_groups:
            grouped_weights_per_layer = []
            for m in weights_per_layer:
                # Group horizontally
                grouped_weights_horiz = []
                for group in items.groups:
                    # TODO gives ERROR in case not all items have the same number of groups.
                    grouped_weights_horiz.append(m[each_item[group]].mean(axis=0))
                grouped_weights_horiz = np.stack(grouped_weights_horiz)

                # Group the result also vertically
                grouped_weights = []
                for group in items.groups:
                    grouped_weights.append(grouped_weights_horiz[:, each_item[group]].mean(axis=1))
                grouped_weights = np.stack(grouped_weights).transpose()  # transpose to restore original order

                # store
                grouped_weights_per_layer.append(grouped_weights)

            # stack and flatten for much easier handling with pandas, computing means etc.
            weights_per_layer = np.stack(grouped_weights_per_layer).reshape(-1)

        weights_for_all_items.append(weights_per_layer)

    # At this point weights_for_all_items is a list containing, for each item, a numpy array with (flattened) attention weights averaged for token groups, for all layers


    ## Store the weights in dataframe together with original data
    weights_for_all_items = pd.DataFrame(weights_for_all_items)
    df = pd.concat([items, weights_for_all_items], axis=1)


    ## Compute means over attention weights across all conditions
    means = df.groupby(items.factors).mean().values
    means = means.reshape(len(items.conditions) * n_layers, -1)
    multi_index = pd.MultiIndex.from_product(
        [items.levels[factor] for factor in items.factors] + [list(range(n_layers))], names=items.factors + ['layer'])
    df_means = pd.DataFrame(means, index=multi_index)


    ## Print a quick text summary of main results, significance tests, etc.
    # TODO implement this :)


    ## Create plots!
    plot(df_means, items.levels, items.groups, n_layers, args)


def parse_data(data_path, tokenizer):
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
        for each_part in row[-1].strip('|').split('|'):
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
        token_ids_list = [[] for _ in range(max(group_to_token_ids)+1)]
        for key in group_to_token_ids:
            token_ids_list[key] = group_to_token_ids[key]

        # create data row
        items.append(row[:-1] + [sentence.strip()] + [' '.join(['[CLS]'] + tokenizer.tokenize(sentence) + ['[SEP]'])] + token_ids_list)

    # If no legend was given, infer legends with boring names from the data itself
    if group_legend is None:
        group_names = ['g{}'.format(i) for i in range(max_group_id + 1)]
    else:
        group_names = [group_legend[key] for key in group_legend]
    if factor_legend is None:
        factor_names = ['f{}'.format(i) for i in range(num_factors)]
    else:
        factor_names = [factor_legend[key] for key in factor_legend]

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
        items.conditions = list(itertools.product(*[items.levels[factor] for factor in items.factors]))

    return items


def normalize(v):
    """
    Divides a vector by its norm.
    :param v:
    :return:
    """
    norm = np.linalg.norm(v)
    if norm == 0:
       return v
    return v / norm


def compute_PAT(all_attention_weights, layer_norm=True):
    """
    Computes Percolated Attention per Token (PAT), through all layers.
    :param all_attention_weights: as retrieved from the attention visualizer
    :param layer_norm: whether to normalize the weights of each attention head
    :return: percolated activations up to every layer
    """
    # TODO: Think about this. What about normalizing per layer, instead of per head? Does that make any sense? Yes, a bit. However, since BERT has LAYER NORM in each attention head, outputs of all heads will have same mean/variance. Does this mean that all heads will contribute same amount of information? Yes, roughly.
    percolated_activations_per_layer = []
    percolated_activations = np.diag(np.ones(all_attention_weights.shape[-1]))      # n_tokens × n_tokens
    for layer in all_attention_weights:
        summed_activations = np.zeros_like(percolated_activations)
        for head in layer:      # n_tokens × n_tokens
            activations_per_head = np.matmul(head, percolated_activations)
            # (i,j) = how much (activations coming ultimately from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])
            summed_activations += activations_per_head
        # for the next layer, use summed_activations as the next input activations
        percolated_activations = summed_activations
        # I believe normalizing the activations (as a whole or per col) makes no difference.

        percolated_activations_per_layer.append(percolated_activations)

    return percolated_activations_per_layer


def compute_MAT(all_attention_weights, layer_norm=True):
    """
    Computes Mean Attention per Token (MAT), i.e,, mean across all attention heads, per layer.
    :param all_attention_weights: as retrieved from attention visualizer
    :param layer_norm: whether to normalize
    :return: mean attention weights (across heads) per layer
    """
    mean_activations_per_layer = []
    for heads_of_layer in all_attention_weights:
        summed_activations = np.zeros_like(all_attention_weights[0][1])
        for head in heads_of_layer:      # n_tokens × n_tokens
            activations_per_head = head.copy()
            # (i,j) = how much (activations coming from) token i influences token j
            if layer_norm:       # Normalize influence (across all tokens i) on each token j
                for j in range(0, len(activations_per_head)):
                    activations_per_head[:,j] = normalize(activations_per_head[:, j])
            summed_activations += activations_per_head

        mean_activations_per_layer.append(summed_activations / all_attention_weights.shape[1])

    return mean_activations_per_layer


def plot(df_means, levels, groups, n_layers, args):
    """
    Output a series of image files, one for each layer, and optionally an animated gif :).
    Each image typically contains several plots, depending on which factors to cross.
    :param df_means: dataframe containing the means per token group for each condition (factor * levels)
    :param levels: what levels are there per factor
    :param groups: what are the token groups called?
    :param n_layers: how many layers are there to plot?
    :param args: the command line arguments; they contain some further settings.
    :return:
    """
    # TODO: levels and n_layers can be inferred from df_means... Remove as arguments?
    # TODO: perhaps it's useful to allow plotting means over layers

    # Consider only those means needed to plot
    df_means = df_means.groupby(args.factors + ['layer']).mean()

    # Collect which plots to make (crossing factors + optional difference plot)
    levels_horiz = levels[args.factors[0]]
    levels_vert = levels[args.factors[1]] if len(args.factors) == 2 else [None]
    if len(levels_horiz) == 2 and not args.no_difs:  # if two levels, also compute difference
        levels_horiz.append('<DIFF>')
    if len(levels_vert) == 2 and not args.no_difs:  # if two levels, also compute difference
        levels_vert.append('<DIFF>')

    # Global min/max to have same color map everywhere
    vmin = df_means.min().min()
    vmax = df_means.max().max()

    # for keeping track to create gif
    out_filepaths = []

    # Let's plot!
    for l in range(n_layers):

        fig, axs = plt.subplots(ncols=len(levels_horiz), nrows=len(levels_vert),
                                figsize=(4 * len(levels_horiz), 4 * len(levels_vert)))
        plt.subplots_adjust(wspace=.6, top=.9)
        fig.suptitle("{}-scores given {} (layer {})".format(args.method, ' × '.join(args.factors), l))

        # Keep for computing differences
        weights_for_diff = [[None, None],
                            [None, None]]

        for h, level_horiz in enumerate(levels_horiz):

            for v, level_vert in enumerate(levels_vert):

                index = groups if not args.no_groups else None  # TODO "None" will give error; replace by exemplary tokens of a given type of item?  items.iloc[0].tokenized.split()

                is_difference_plot = True
                if level_horiz != "<DIFF>" and level_vert == "<DIFF>":
                    weights = weights_for_diff[h][0] - weights_for_diff[h][1]
                elif level_horiz == "<DIFF>" and level_vert != "<DIFF>":
                    weights = weights_for_diff[0][v] - weights_for_diff[1][v]
                elif level_horiz == "<DIFF>" and level_vert == "<DIFF>":
                    weights = weights_for_diff[0][0] - weights_for_diff[1][1]
                else:
                    is_difference_plot = False
                    weights = df_means.loc[(level_horiz, level_vert, l)] if level_vert is not None else df_means.loc[
                        (level_horiz, l)]
                    weights = weights.values
                    dim = int(np.sqrt(weights.shape[-1]))
                    weights = weights.reshape(dim, dim)
                    if not args.no_transpose:
                        weights = weights.transpose()
                    weights = pd.DataFrame(weights, index=index, columns=index)
                    # Keep for difference plot
                    weights_for_diff[h][v] = weights

                # TODO Consider setting global vmin/vmax only in case of MAT; in that case also for is_difference_plot.
                ax = sns.heatmap(weights,
                                 xticklabels=True,
                                 yticklabels=True,
                                 vmin=vmin if not is_difference_plot else None,
                                 vmax=vmax if not is_difference_plot else None,
                                 center=0 if is_difference_plot else None,
                                 linewidth=0.5,
                                 ax=axs[h, v] if level_vert is not None else axs[h],
                                 cbar=False,
                                 cmap="coolwarm_r" if is_difference_plot else "Blues",
                                 square=True,
                                 cbar_kws={'shrink': .5},
                                 label='small')
                if is_difference_plot:
                    ax.set_title('Difference')
                else:
                    ax.set_title('{} & {}'.format(level_horiz, level_vert) if level_vert is not None else level_horiz)
                # ax.xaxis.tick_top()
                plt.setp(ax.get_yticklabels(), rotation=0)

        out_filepath = "{}/{}_{}_layer{}.png".format(args.out, args.method, '-x-'.join(args.factors), l)
        print("Saving figure:", out_filepath)
        pylab.savefig(out_filepath)
        # pylab.show()

        out_filepaths.append(out_filepath)

    if args.gif:  # :)
        out_filepath = "{}/{}_{}_animated.gif".format(args.out, args.method, '-x-'.join(args.factors))
        images = []
        for filename in out_filepaths:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_filepath, images, format='GIF', duration=.5)
        print("Saving movie:", out_filepath)


if __name__ == "__main__":
    main()