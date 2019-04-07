import torch
from pytorch_pretrained_bert import BertModel, BertTokenizer

import numpy as np
import pandas as pd
import csv
import itertools

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.image as mpimg
from matplotlib import ticker
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

# TODO: perhaps it's useful to allow plotting means over layers; sliding window-style? or chaining but with different starting points?
# TODO: Is attention-chain bugged? Plots are uninterpretable; without normalization super high values only at layer 10-11... with normalization... big gray mess.
# TODO: Should I take sum influence per group of tokens, or mean? E.g., with averaging, "a boy" will be dragged down by uninformative "a"...

def main():
    """
    To run this code with default settings and example data, do
       $ python main.py data/example.csv
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

    ## Set up tokenizer, data and model
    tokenizer = BertTokenizer.from_pretrained(args.bert, do_lower_case=("uncased" in args.bert))
    items = parse_data(args.data, tokenizer)
    model = BertModel.from_pretrained(args.bert)


    ## Store for convenience
    args.factors = args.factors or items.factors[:2]    # by default use the first two factors from the data
    n_layers = len(model.encoder.layer)


    ## Now that args.factors is known, finally choose output directory
    if args.out is None:
        dirname = 'temp'
        out_idx = 0
        if not os.path.exists("output"):
            os.mkdir('output')
        while any(x.startswith(dirname) for x in os.listdir('output')):
            out_idx += 1
            dirname = 'temp{}'.format(out_idx)
        dirname += "_{}{}{}_{}".format(args.method,
                                       "-" + args.combine if args.combine != "no" else "",
                                       "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                       '-x-'.join(args.factors))
        args.out = os.path.join("output", dirname)
        os.mkdir(args.out)


    ## Compute attention weights, one item at a time
    weights_for_all_items = []
    for _, each_item in items.iterrows():

        if args.method == "attention":
            tokens_a, tokens_b, attention = apply_bert_get_attention(model, tokenizer, each_item['sentence'])
            attention = attention.squeeze()
            if args.combine == "chain":
                weights_per_layer = compute_pMAT(attention, layer_norm=args.normalize_heads)
            else:
                weights_per_layer = compute_MAT(attention, layer_norm=args.normalize_heads)
            ## Nope, instead of the following, transpose the gradients: from input_token to output_token
            # weights_per_layer = weights_per_layer.transpose(0,2,1)  # for uniformity with gradients: (layer, output_token, input_token)
        elif args.method == "gradient":
            tokens_a, tokens_b, weights_per_layer = apply_bert_get_gradients(model, tokenizer, each_item['sentence'], chain=args.combine=="chain")
            weights_per_layer = weights_per_layer.transpose(0,2,1)  # for uniformity with attention weights: (layer, input_token, output_token)

        if args.combine == "cumsum":
            weights_per_layer = np.cumsum(weights_per_layer, axis=0)

        # weights_per_layer now contains: (n_layers, n_tokens, n_tokens)

        # TODO Put the following outside the current loop, inside its own (anticipating intermediary writing of results to disk)
        # Take averages over groups of tokens
        if not args.ignore_groups:
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


    ## Compute means over attention weights across all conditions (easy because they're flattened)
    if len(items.factors) > 0:
        means = df.groupby(items.factors).mean().values
    else:
        means = df.mean().values
    means = means.reshape(len(items.conditions) * n_layers, -1)
    multi_index = pd.MultiIndex.from_product(
        [items.levels[factor] for factor in items.factors] + [list(range(n_layers))], names=items.factors + ['layer'])
    df_means = pd.DataFrame(means, index=multi_index)

    # TODO Now might be a good time to store intermediate results to disk?


    ## Restrict attention to the factors of interest:
    df_means = df_means.groupby(args.factors + ['layer']).mean()


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
        out_filepath = "{}/{}_{}_animated.gif".format(args.out, args.method, '-x-'.join(args.factors))
        images = []
        for filename in out_filepaths:
            images.append(imageio.imread(filename))
        imageio.mimsave(out_filepath, images, format='GIF', duration=.5)
        print("Saving movie:", out_filepath)


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


def tokenize_sequence(tokenizer, sequence):
    sequence = sequence.split(" \|\|\| ")
    if len(sequence) == 1:
        sentence_a = sequence[0]
        sentence_b = ""
    else:
        sentence_a = sequence[0]
        sentence_b = sequence[1]

    tokens_a = tokenizer.tokenize(sentence_a)
    tokens_b = tokenizer.tokenize(sentence_b)
    tokens_a_delim = ['[CLS]'] + tokens_a + ['[SEP]']
    tokens_b_delim = tokens_b + (['[SEP]'] if len(tokens_b) > 0 else [])
    token_ids = tokenizer.convert_tokens_to_ids(tokens_a_delim + tokens_b_delim)
    tokens_tensor = torch.tensor([token_ids])
    token_type_tensor = torch.LongTensor([[0] * len(tokens_a_delim) + [1] * len(tokens_b_delim)])

    return tokens_a_delim, tokens_b_delim, tokens_tensor, token_type_tensor


def apply_bert_get_attention(model, tokenizer, sequence):
    """
    Essentially isolated from jessevig/bertviz
    :param model: bert
    :param tokenizer: bert tokenizer
    :param sequence: single sentence
    :return:
    """

    model.eval()
    tokens_a, tokens_b, tokens_tensor, token_type_tensor = tokenize_sequence(tokenizer, sequence)
    _, _, attn_data_list = model(tokens_tensor, token_type_ids=token_type_tensor)
    attn_tensor = torch.stack([attn_data['attn_probs'] for attn_data in attn_data_list])
    attn = attn_tensor.data.numpy()

    return tokens_a, tokens_b, attn


def apply_bert_get_gradients(model, tokenizer, sequence, chain):
    """
    :param model: bert
    :param tokenizer: bert tokenizer
    :param sequence: single sentence
    :param chain: whether to compute the gradients all the way back, i.e., wrt the input embeddings
    :return:
    """
    model.train()   # Because I need the gradients
    model.zero_grad()

    tokens_a, tokens_b, tokens_tensor, token_type_tensor = tokenize_sequence(tokenizer, sequence)

    encoded_layers, _, _, embedding_output = model(tokens_tensor, token_type_ids=token_type_tensor, output_embedding=True)
#   [n_layers x [batch_size, seq_len, hidden]]      [seq_len x hidden]

    previous_activations = embedding_output
    gradients = []
    for layer in encoded_layers:        # layer: [batch_size, seq_len, hidden]
        target = embedding_output if chain else previous_activations
        gradients_for_layer = []

        for token_idx in range(layer.shape[1]): # loop over output tokens
            target.retain_grad()    # not sure if needed every iteration
            mask = torch.zeros_like(layer)
            mask[:,token_idx,:] = 1
            layer.backward(mask, retain_graph=True)
            gradient = target.grad.data    # [batch_size, seq_len, hidden]
            gradient = gradient.squeeze().clone().numpy()
            gradient_norm = np.linalg.norm(gradient, axis=-1)   # take norm per input token
            gradients_for_layer.append(gradient_norm)
            target.grad.data.zero_()
            previous_activations = layer

        gradients_for_layer = np.stack(gradients_for_layer)
        gradients.append(gradients_for_layer)

    gradients = np.stack(gradients)

    return tokens_a, tokens_b, gradients


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

# TODO Merge the following two functions and change names
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

    return np.stack(mean_activations_per_layer)


def compute_pMAT(all_attention_weights, layer_norm=True):
    """
    Computes Percolated Mean Attention per Token (pMAT), through all layers.
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

    return np.stack(percolated_activations_per_layer)


def create_dataframes_for_plotting(items, df_means, n_layers, args):
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
        weights_to_plot = [[[] for _ in levels_vert] for _ in levels_horiz]

        # Loop through rows and columns of the multiplot-to-be:
        for h, level_horiz in enumerate(levels_horiz):
            for v, level_vert in enumerate(levels_vert):
                # Things are easy if it's a difference plot of plots we've already computed before:
                if level_horiz != "<DIFF>" and level_vert == "<DIFF>":
                    weights = weights_to_plot[h][0] - weights_to_plot[h][1]
                    weights.difference = True
                elif level_horiz == "<DIFF>" and level_vert != "<DIFF>":
                    weights = weights_to_plot[0][v] - weights_to_plot[1][v]
                    weights.difference = True
                elif level_horiz == "<DIFF>" and level_vert == "<DIFF>":
                    weights = weights_to_plot[0][0] - weights_to_plot[1][1]
                    weights.difference = True
                # More work if it's an actual weights plot:
                else:
                    weights = df_means.loc[(level_horiz, level_vert, l)] if level_vert is not None else (df_means.loc[(level_horiz, l)] if level_horiz is not None else df_means.loc[l])
                    weights = weights.values
                    dim = int(np.sqrt(weights.shape[-1]))
                    weights = weights.reshape(dim, dim)
                    # if args.transpose:        ## Disabled; would yield incorrect axis labels etc.
                    #    weights = weights.transpose()
                    index = items.groups if not args.ignore_groups else None  # TODO "None" will give error; replace by exemplary tokens of a given type of item?  items.iloc[0].tokenized.split()
                    weights = pd.DataFrame(weights, index=index, columns=index)
                    weights.difference = False

                # Compute how much (more) each token influences vs. how much it is influenced:
                # TODO Not sure how to interpret this for DIFF plots...
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    weights.balance = pd.DataFrame({'balance': (weights.transpose()-weights.values).sum(axis=0)})

                # Some convenient metadata (used mostly when creating plots)
                # It's a lot safer that each dataframe carries its own details with it in this way.
                weights.level_horiz = level_horiz
                weights.level_vert = level_vert
                weights.max_for_colormap = weights.max().max()
                weights.min_for_colormap = weights.min().min()
                weights.layer = l

                # Add dataframe to designated position
                weights_to_plot[h][v] = weights

        # Save the weights to plot for this particular layer
        weights_to_plot_per_layer.append(weights_to_plot)

    # The list weights_to_plot_per_layer now contains, for each layer, a list of lists of weights matrices.
    return weights_to_plot_per_layer


def plot(weights_to_plot, args):

    return plot_new(weights_to_plot, args)

    """
    Output a single image file, typically containing several plots, depending on which factors to cross
    and whether to include a difference plot.
    :param weights_to_plot: list of lists of weights dataframes; each DF will be plotted as a single heatmap.
    :param args: the command line arguments; they contain some further settings.
    :return: output file path
    """
    # TODO Tweak side-title and main title font size, padding, which is ugly especially if there's only a single plot (i.e., if there are no factors).
    # TODO Tweak filename and title if there are no factors being crossed.

    # Let's plot!
    fig, axs = plt.subplots(ncols=len(weights_to_plot), nrows=len(weights_to_plot[0]),
                            figsize=(4 * len(weights_to_plot), 4 * len(weights_to_plot[0])))
    plt.subplots_adjust(wspace=.6, top=.9)
    fig.suptitle("{}{} given {} (layer {})".format(args.method, "-"+args.combine if args.combine != "no" else "", ' × '.join(args.factors), weights_to_plot[0][0].layer), size=20)

    if args.combine != "cumsum":
        suplabel("x", "...influenced by tokens at {}".format("embedding layer" if (args.combine != "no" or weights_to_plot[0][0].layer == 0) else "layer {}".format(weights_to_plot[0][0].layer - 1)), labelpad=3)
        suplabel("y", "Tokens at layer {}...".format(weights_to_plot[0][0].layer), labelpad=8)
    else:
        suplabel("x", "...influenced by tokens at previous layer", labelpad=3)
        suplabel("y", "Tokens up to layer {}...".format(weights_to_plot[0][0].layer), labelpad=8)
    for c, col in enumerate(weights_to_plot):

        for r, weights in enumerate(col):

            axis = axs[c, r] if weights.level_vert is not None else axs[c] if weights.level_horiz is not None else axs

            # TODO Consider setting global vmin/vmax only in case of MAT; in that case also for is_difference_plot.
            ax = sns.heatmap(weights,
                             xticklabels=True,
                             yticklabels=True,
                             vmin=weights.min_for_colormap,
                             vmax=weights.max_for_colormap,
                             center=0 if weights.difference else None,
                             linewidth=0.5,
                             ax=axis,
                             cbar=True,
                             cmap="coolwarm_r" if weights.difference else "Blues",
                             square=True,
                             cbar_kws={'shrink': .5},
                             label='small')
            if weights.difference:
                ax.set_title('Difference')
            else:
                ax.set_title('{} & {}'.format(weights.level_horiz, weights.level_vert) if weights.level_vert is not None else weights.level_horiz)
            # ax.xaxis.tick_top()
            plt.setp(ax.get_yticklabels(), rotation=0)

    out_filepath = "{}/{}{}{}_{}_layer{}.png".format(args.out, args.method,
                                                     "-"+args.combine if args.combine != "no" else "",
                                                     "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                                     '-x-'.join(args.factors), weights_to_plot[0][0].layer)
    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)
    # pylab.show()

    return out_filepath


def plot_new(weights_to_plot, args):

    """
    Output a single image file, typically containing several plots, depending on which factors to cross
    and whether to include a difference plot.
    :param weights_to_plot: list of lists of weights dataframes; each DF will be plotted as a single heatmap.
    :param args: the command line arguments; they contain some further settings.
    :return: output file path
    """
    # TODO Tweak side-title and main title font size, padding, which is ugly especially if there's only a single plot (i.e., if there are no factors).
    # TODO Tweak filename and title if there are no factors being crossed.

    # Let's plot!
    f = plt.figure(figsize=(4 * len(weights_to_plot) + 1, 4 * len(weights_to_plot[0])))
    gs0 = gridspec.GridSpec(len(weights_to_plot[0]), len(weights_to_plot), figure=f, hspace=.6, wspace=.6)

    # plt.subplots_adjust(wspace=.6, top=.9)
    f.suptitle("{}{} given {} (layer {})".format(args.method, "-"+args.combine if args.combine != "no" else "", ' × '.join(args.factors), weights_to_plot[0][0].layer), size=16)

    for c, col in enumerate(weights_to_plot):

        for r, weights in enumerate(col):

            axis = gs0[c, r] if weights.level_vert is not None else gs0[c] if weights.level_horiz is not None else gs0[0]

            subgs = gridspec.GridSpecFromSubplotSpec(2, 2, subplot_spec=axis, height_ratios=(len(weights), 1 if args.balance else .0001), width_ratios=(.95, .05))

            ax_main = plt.Subplot(f, subgs[0,0])
            f.add_subplot(ax_main)
            ax_main_cbar = plt.Subplot(f, subgs[0,1])
            f.add_subplot(ax_main_cbar)
            if args.balance:
                ax_balance = plt.Subplot(f, subgs[1, 0])
                f.add_subplot(ax_balance)
                ax_balance_cbar = plt.Subplot(f, subgs[1,1])
                f.add_subplot(ax_balance_cbar)

            # TODO Consider setting global vmin/vmax only in case of MAT; in that case also for is_difference_plot.
            sns.heatmap(weights,
                         xticklabels=not args.balance,
                         yticklabels=True,
                         vmin=weights.min_for_colormap,
                         vmax=weights.max_for_colormap,
                         center=0 if weights.difference else None,
                         linewidth=0.5,
                         ax=ax_main,
                         cbar=True,
                         cbar_ax=ax_main_cbar,
                         cmap="coolwarm_r" if weights.difference else "Blues",
                         square=False,      # TODO See if I can get the square to work...
#                        cbar_kws={'shrink': .5},
                         label='small')
            ax_main.set_xlabel('...to layer {}'.format(weights.layer))
            ax_main.set_ylabel('From previous layer...' if args.combine == "no" else "From initial embeddings...")
            if weights.difference:
                ax_main.set_title('Difference')
            else:
                ax_main.set_title('{} & {}'.format(weights.level_horiz, weights.level_vert) if weights.level_vert is not None else (weights.level_horiz or ""))

            map_img = mpimg.imread('arrow-down-white.png')

            ax_main.imshow(map_img,
                        aspect=ax_main.get_aspect(),
                        extent=ax_main.get_xlim() + ax_main.get_ylim(),
                        zorder=3,
                        alpha=.2)

            plt.setp(ax_main.get_yticklabels(), rotation=0)

#            print(weights.balance, weights.balance.index, weights.balance.values)
            if args.balance:
                sns.heatmap(weights.balance.transpose(),
                        xticklabels=True,
                        yticklabels=['Balance'],
                        ax=ax_balance,
                        center=0,
                        vmin = -.05,   # TODO compute vmin and vmax globally
                        vmax = .05,
                        linewidth=0.5,
                        cmap="coolwarm_r",
                        cbar=True,
                        cbar_ax=ax_balance_cbar,
                        # cbar_kws={'shrink': .5}, # makes utterly mini...
                        label='small',
                        cbar_kws=dict(ticks=[-.05, 0, .05])    # TODO Add global cmin and vmax here.
                        )
                plt.setp(ax_balance.get_yticklabels(), rotation=0)
                # ax_balance.xaxis.tick_top()
                # ax_main.set_xlabel('...to layer {}'.format(weights.layer))

    gs0.tight_layout(f, rect=[0, 0.03, 1, 0.95])
#    f.subplots_adjust(top=1.0-(1.0 / (4 * len(weights_to_plot) + 1)))

    out_filepath = "{}/{}{}{}_{}_layer{}.png".format(args.out, args.method,
                                                     "-"+args.combine if args.combine != "no" else "",
                                                     "_normalized" if (args.method == "attention" and args.normalize_heads) else "",
                                                     '-x-'.join(args.factors), weights_to_plot[0][0].layer)
    print("Saving figure:", out_filepath)
    pylab.savefig(out_filepath)
    # pylab.show()

    return out_filepath

# plt.subplots(1, 2, gridspec_kw={'width_ratios': [3, 1]})      # different size subplots


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

    ## First compute and set max and min per layer
    for l in range(len(weights_to_plot_per_layer)):
        weights = [w for row in weights_to_plot_per_layer[l] for w in row]

        layer_max = max([w.max_for_colormap for w in weights if not w.difference])
        layer_min = min([w.min_for_colormap for w in weights if not w.difference])
        layer_max_diff = max([max(abs(w.max_for_colormap), abs(w.min_for_colormap)) for w in weights if w.difference] + [0])

        # Default: "layer" calibration, set the new max and min values.
        for w in weights:
            if w.difference:
                w.max_for_colormap = layer_max_diff
                w.min_for_colormap = -layer_max_diff
            else:
                w.max_for_colormap = layer_max
                w.min_for_colormap = layer_min

        layer_maxes.append(layer_max)
        layer_mins.append(layer_min)
        layer_max_diffs.append(layer_max_diff)

    ## If global_colormap requested, compute overall max and min (across layers) and add these values to dataframes
    if global_colormap:
        overall_max = max(layer_maxes)
        overall_min = min(layer_mins)
        overall_max_diff = max(layer_max_diffs)

        weights = [w for layer in weights_to_plot_per_layer for row in layer for w in row]
        for w in weights:
            if w.difference:
                w.max_for_colormap = overall_max_diff
                w.min_for_colormap = -overall_max_diff
            else:
                w.max_for_colormap = overall_max
                w.min_for_colormap = overall_min


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

