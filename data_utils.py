from conllu import parse
from conllu import parse_incr
import os
import csv
import random
import warnings
import pandas as pd
import numpy as np

import regex

from nltk.tokenize import word_tokenize

"""
Mostly concerned with reading universal dependency format, connlu.
"""

path_to_conllu_file = "/home/u148187/datasets/Universal dependencies/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu"

UPOS_tags = {
    'ADJ': 'adjective',
    'ADP': 'adposition',
    'ADV': 'adverb',
    'AUX': 'auxiliary',
    'CCONJ': 'coordinating conjunction',
    'DET': 'determiner',
    'INTJ': 'interjection',
    'NOUN': 'noun',
    'NUM': 'numeral',
    'PART': 'particle',
    'PRON': 'pronoun',
    'PROPN': 'proper noun',
    'PUNCT': 'punctuation',
    'SCONJ': 'subordinating conjunction',
    'SYM': 'symbol',
    'VERB': 'verb',
    'X': 'other',
}

open_class_tags = ['ADJ', 'ADV', 'INTJ', 'NOUN', 'PROPN', 'VERB']
closed_class_tags = ['ADP', 'AUX', 'CCONJ', 'DET', 'NUM']

all_dependency_relations = {
    'acl': 'clausal modifier of noun(adjectival clause)',
    'advcl': 'adverbial clause modifier',
    'advmod': 'adverbial modifier',
    'amod': 'adjectival modifier',
    'appos': 'appositional modifier',
    'aux': 'auxiliary',
    'case': 'case marking',
    'cc': 'coordinating conjunction',
    'ccomp': 'clausal complement',
    'clf': 'classifier',
    'compound': 'compound',
    'conj': 'conjunct',
    'cop': 'copula',
    'csubj': 'clausal subject',
    'dep': 'unspecified dependency',
    'det': 'determiner',
    'discourse': 'discourse element',
    'dislocated': 'dislocated elements',
    'expl': 'expletive',
    'fixed': 'fixed multiword expression',
    'flat': 'flat multiword expression',
    'goeswith': 'goes with',
    'iobj': 'indirect object',
    'list': 'list',
    'mark': 'marker',
    'nmod': 'nominal modifier',
    'nsubj': 'nominal subject',
    'nummod': 'numeric modifier',
    'obj': 'object',
    'obl': 'oblique nominal',
    'orphan': 'orphan',
    'parataxis': 'parataxis',
    'punct': 'punctuation',
    'reparandum': 'overridden disfluency',
    'root': 'root',
    'vocative': 'vocative',
    'xcomp': 'open clausal complement'
}

nominal_core_arguments = ['nsubj', 'obj', 'iobj']
nominal_non_core_dependents = ['obl', 'vocative', 'expl', 'dislocated']
nominal_nominal_dependents = ['nmod', 'appos', 'nummod']

modifier_core_arguments = []
modifier_non_core_dependents = ['advmod', 'discourse']
modifier_nominal_dependents = ['amod']

clause_core_arguments = ['csubj', 'ccomp', 'xcomp']
clause_non_core_dependents = ['advcl']
clauser_nominal_dependents = ['acl']

function_core_arguments = []
function_non_core_dependents = ['aux', 'cop', 'mark']
function_nominal_dependents = ['det', 'clf', 'case']


# TODO Write to single file and adapt parse_data instead.
def write_file_plain_sentences(n, with_dependencies=False):
    """
    :param n: how many (random sample; fixed seed!)
    :param with_dependencies: whether to write dependencies to a column as well; currently as a list of pairs (no labels)
        WARNING: If not with dependencies, NO ALIGNMENT WITH DEPENDENCY RELATIONS IS GUARANTEED.
    :return:
    """
    out_file_path = os.path.basename(path_to_conllu_file)[:-7] + '{}.csv'.format(n)
    out_file_path_dep = os.path.basename(path_to_conllu_file)[:-7] + '{}-{}.csv'.format(n, 'dep')

    sentences = []

    for s in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
        sentences.append(s)

    random.seed(12345)

    indices = random.sample(list(range(len(sentences))), n)
    sentences = [sentences[i] for i in indices]

    if not with_dependencies:
        with open('data/' + out_file_path, 'w') as outfile:
            outfile.write('# index, id \n')
            writer = csv.writer(outfile)
            for i, s in zip(indices, sentences):
                row = [str(i), s.metadata['sent_id'], s.metadata['text']]
                writer.writerow(row)

    else:
        with open('data/' + out_file_path_dep, 'w') as outfile:
            outfile.write('# index, id, dependencies \n')
            writer = csv.writer(outfile)

            for i, s in zip(indices, sentences):
                nodes_to_explore = [s.to_tree()]
                arcs = []
                while len(nodes_to_explore) > 0:
                    node = nodes_to_explore.pop()
                    for c in node.children:
                        nodes_to_explore.append(c)
                        arcs.append((node.token['id']-1, c.token['id']-1))
                sentence = ' '.join([t["form"] for t in s])
                row = [str(i), s.metadata['sent_id'], ';'.join(['{}-{}'.format(a,b) for (a,b) in arcs]), sentence]
                writer.writerow(row)


def write_file_for_nominal_core_args():
    """
    Extract just the 'nominal core args', i.e., subject and objects.
    :return:
    """
    out_file_path = os.path.basename(path_to_conllu_file)[:-7]+'_nom_args.csv'

    with open('data/'+out_file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['{}|{} {}'.format('#' if i==0 else '', i, t) for i,t in enumerate(nominal_core_arguments)])

        for sentence in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
            token_forms = []
            for token in sentence:
                if token["deprel"] in nominal_core_arguments:
                    index = nominal_core_arguments.index(token["deprel"])
                    token_forms.append('|{} {} |'.format(index, token["form"].replace('|','')))
                else:
                    token_forms.append(token["form"])
            writer.writerow([' '.join(token_forms)])


def write_file_for_open_vs_closed_POS():
    """
    To extract sentences where open/closed class POS tags are grouped
    :return:
    """
    out_file_path = os.path.basename(path_to_conllu_file)[:-7]+'_open-closed.csv'

    with open('data/'+out_file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['{}|{} {}'.format('#' if i==0 else '', i, t) for i,t in enumerate(['open', 'closed'])])

        for sentence in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
            token_forms = []
            for token in sentence:
                if token["upostag"] in open_class_tags:
                    token_forms.append('|{} {} |'.format(0, token["form"].replace('|', '')))
                elif token["upostag"] in closed_class_tags:
                    token_forms.append('|{} {} |'.format(1, token["form"].replace('|', '')))
                else:
                    token_forms.append(token["form"])
            writer.writerow([' '.join(token_forms)])


def write_file_for_main_POS():
    """
    To extract sentences where all POS of a certain type are grouped.
    :return:
        """
    tags_of_interest = ['ADJ', 'ADV', 'NOUN', 'PRON', 'VERB', 'AUX', 'DET', 'PROPN']

    out_file_path = os.path.basename(path_to_conllu_file)[:-7]+'_POS.csv'

    with open('data/'+out_file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['{}|{} {}'.format('#' if i==0 else '', i, t) for i,t in enumerate(tags_of_interest)])

        for sentence in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
            token_forms = []
            for token in sentence:
                if token["upostag"] in tags_of_interest:
                    index = tags_of_interest.index(token["upostag"])
                    token_forms.append('|{} {} |'.format(index, token["form"].replace('|','')))
                else:
                    token_forms.append(token["form"])
            writer.writerow([' '.join(token_forms)])


def read_categories_from_rosch_etal():
    """
    Lazy sloppy code for selecting some categories.
    :return:
    """
    path = 'data/auxiliary/categories.txt'
    triples = []

    with open(path) as data:
        for line in data:
            if not line.startswith('#'):
                line = line.split('|')
                superordinate = None
                if line[0] != '':
                    superordinate = line[0].strip()
                if line[1] != '':
                    basics = [term.strip() for term in line[1].split(';')]
                else:
                    basics = [None]
                subordinates_per_basic = [[None] for _ in basics]
                if line[2] != '':
                    termslists = [[term.strip() for term in seq.split(',')] for seq in line[2].split(';')]
                    termslists = [[None if (a==[''] or a == [] or x == '') else x for x in a] for a in termslists]
                    for i, terms in enumerate(termslists):
                        subordinates_per_basic[i] = terms
                for i, basic in enumerate(basics):
                    triples.extend([(superordinate, basic, subordinate) for subordinate in subordinates_per_basic[i]])

    triples = [t for t in triples if None not in t]


    # At least add all single-word cases
    random.seed(1235)

    random.shuffle(triples)

    length_one = [t for t in triples if all(len(w.split()) == 1 for w in t)]

    selection = []
    for tuple in length_one:
        if not any(word in t for t in selection for word in tuple):
            selection.append(tuple)

    # Then keep adding 2-word cases until you have enough of 'em

    length_two = [t for t in triples if (all(len(w.split()) == 1 or len(w.split()) == 2 for w in t) and not all(len(w.split()) == 1 for w in t))]
    for tuple in length_two:
        if not any(word in t for t in selection for word in tuple):
            selection.append(tuple)

    # Add up to 10
    selection = selection[:10]

    print('\n\n'.join(['\n'.join(t) for t in selection]))

    return selection

# read_categories_from_rosch_etal()


def generate_sentences_from_categories():
    """
    Takes a file containing data like this:
    > super; weapon; He was arraigned Tuesday on nine charges -- two counts of aggravated murder, attempted aggravated murder, first-degree assault, two counts of intimidation and three counts of unlawful use of a weapon.
    > basic; gun; The other two roommates, William Calderon and Cory Lynch, each decided to "prank" Siela by pulling an unloaded gun on him and pretending to shoot when he returned to the living room.
    > sub; shotgun;  The door slowly creaked open, and Dan found himself nose to nose with the barrel of a shotgun.
    > [newline before next triple]
    and generates sentences by replacing each term by terms of other levels of categorization, written in a format readable by experiment.py
    :return:
    """
    sentences = []
    tuples_dict = {}
    tuples = []
    with open('data/auxiliary/category-sentences.txt') as file:
        for line in file:
            if not line.startswith('#') and not line.strip() == '':
                row = [s.strip() for s in line.split(';')]
                sentences.append(row)
    if len(sentences) %3  != 0:
        print("Something's wrong")
    for i in range(0, len(sentences), 3):
        tuple = [s[1] for s in sentences[i:i+3]]
        tuples.append(tuple)
        tuples_dict[tuple[0]] = tuples_dict[tuple[1]] = tuples_dict[tuple[2]] = tuple

    all_items = []
    for original in sentences:
        all_items.append((original[0], original[0], 'natural', original[1], original[2]))
        for i, level in enumerate(['super', 'basic', 'sub']):
            if level != original[0]:
                replacement = tuples_dict[original[1]][i]
                new_sentence = original[2].replace(original[1], replacement)
                # Quick dirty fix
                for vowel in ['a', 'e', 'i', 'o', 'u']:
                    new_sentence = new_sentence.replace(' a {}'.format(vowel), ' an {}'.format(vowel))
                new_item = (original[0], level, 'manipulated', replacement, new_sentence)
                all_items.append(new_item)

    artificial = 'I wonder if my sister will bring her {} to the meeting.'
    for tuple in tuples:
        for term, level in zip(tuple, ['super', 'basic', 'sub']):
            all_items.append((level, level, 'artificial', term, artificial.format(term)))

    # Write to csv including group tags right away
    with open('data/auxiliary/category-sentences.csv', 'w') as file:
        file.write('# original, level, source, |0 term, |1 rest \n')
        writer = csv.writer(file)
        for item in all_items:
            sent_with_groups = '|1 ' + item[4].replace(item[3], '|0 '+item[3] + ' |1')
            row = [item[0], item[1], item[2], sent_with_groups]
            writer.writerow(row)

    print('wrote {} items to {}'.format(len(all_items), 'data/category-sentences.csv'))


# generate_sentences_from_categories()


def parse_data(data_path, tokenizer, max_items=None, words_as_groups=False, as_dependency=None):
    """
    Turns a .csv file with some special markup of 'token groups' into a dataframe.
    :param data_path:
    :param tokenizer: BERT's own tokenizer
    :return: pandas DataFrame with different factors, the sentence, tokenized sentence, and token group indices as columns
    """

    items = []
    num_factors = None
    max_group_id = 0

    group_regex = regex.compile('\|(\d*) ([^\|]*)')

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
            if len(group_legend) == 0:
                group_legend = None

    # Now read the actual data
    reader = csv.reader(open(data_path), skipinitialspace=True)
    for row in filter(lambda row: not row[0].startswith('#'), reader):
        num_factors = len(row)-1    # Num rows minus the sentence itself
        group_to_token_ids = {}  # Map token-group numbers to token positions
        sentence = ""
        total_len = 1   # Take mandatory CLS symbol into account

        if words_as_groups:
            words = row[-1].split(' ') # WARNING This presumes tokenized lists from UD...
            row[-1] = ' '.join(['|{} {}'.format(i,w) for i,w in enumerate(words)])

        for digit, each_part in regex.findall(group_regex, row[-1]):   # Cheap fix to avoid unintended groups for sentence starting with number
            if digit != '':
                group_id = int(digit)
                each_part = each_part.strip()
                max_group_id = max(max_group_id, group_id)
            tokens = tokenizer.tokenize(each_part)
            # If group has a number, remember this group for plotting etc.
            if digit != '':
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

        # read dependency tree if necessary
        if as_dependency is not None:   # TODO replace rigid -2 by index depending on column label
            if row[-2] != '':
                row[-2] = [tuple([int(a) for a in s.split('-')]) for s in row[-2].split(';')]
            else:
                row[-2] = []

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
        if as_dependency in items.factors:
            items.factors.remove(as_dependency) # This is not a factor proper
        if 'index' in items.factors:
            items.factors.remove('index')  # This is not a factor proper either
        if 'id' in items.factors:
            items.factors.remove('id')  # This is not a factor proper either
        items.num_groups = max_group_id + 1
        items.groups = group_names
        items.levels = {factor: items[factor].unique().tolist() for factor in items.factors}
        # # following is bugged: not all combination needs to exist in the data
        # items.conditions = list(itertools.product(*[items.levels[factor] for factor in items.factors]))
        items.conditions = list(set([tuple(l) for l in items[items.factors].values]))

    return items



def merge_grouped_tokens(items, data_for_all_items, method="mean"):
    """
    Takes weights matrix per item per layer, and averages rows and columns based on desired token groups.
    :param items: dataframe as read from example.csv
    :param data_for_all_items: list of numpy arrays with attention/gradients extracted from BERT
    :param method: mean or sum
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
                if method == "mean":
                    grouped_weights_horiz.append(m[each_item[group]].mean(axis=0))
                elif method == "sum":
                    grouped_weights_horiz.append(m[each_item[group]].sum(axis=0))
            grouped_weights_horiz = np.stack(grouped_weights_horiz)

            # Group the result also vertically
            grouped_weights = []
            for group in items.groups:
                if method == "mean":
                    grouped_weights.append(grouped_weights_horiz[:, each_item[group]].mean(axis=1))
                if method == "sum":
                    grouped_weights.append(grouped_weights_horiz[:, each_item[group]].sum(axis=1))
            grouped_weights = np.stack(grouped_weights).transpose()  # transpose to restore original order

            # store
            data_per_layer.append(grouped_weights)

        data_for_all_items2.append(np.stack(data_per_layer))

    return data_for_all_items2

if __name__ == "__main__":
    write_file_plain_sentences(500, with_dependencies=True)
    # TODO Write similar sentences but with dependency structure.
