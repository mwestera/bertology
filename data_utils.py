from conllu import parse
from conllu import parse_incr
import os
import csv
import random
import warnings
import pandas as pd
import numpy as np

import tree_utils

import regex

from tqdm import tqdm

"""
Mostly concerned with reading universal dependency format, connlu.
"""

path_to_conllu_file = "/home/u148187/datasets/Universal dependencies/ud-treebanks-v2.3/UD_English-GUM/en_gum-ud-dev.conllu"
# path_to_conllu_file = "/home/u148187/datasets/Universal dependencies/ud-treebanks-v2.3/UD_English-EWT/en_ewt-ud-train.conllu"
# path_to_conllu_file = "/home/matthijs/Dropbox/en_ewt-ud-train.conllu"


class CONLLU_tags:

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

    nominal_dependents = nominal_core_arguments + nominal_non_core_dependents + nominal_nominal_dependents

    modifier_core_arguments = []
    modifier_non_core_dependents = ['advmod', 'discourse']
    modifier_nominal_dependents = ['amod']

    modifier_dependents = modifier_core_arguments + modifier_non_core_dependents + modifier_nominal_dependents

    clause_core_arguments = ['csubj', 'ccomp', 'xcomp']
    clause_non_core_dependents = ['advcl']
    clause_nominal_dependents = ['acl']

    clause_dependents = clause_core_arguments + clause_non_core_dependents + clause_nominal_dependents

    function_core_arguments = []
    function_non_core_dependents = ['aux', 'cop', 'mark']
    function_nominal_dependents = ['det', 'clf', 'case']


# TODO filter out uninteresting cases (e.g., 1 token, or no dependency rels)
def write_file_plain_sentences(n, with_dependencies=False):
    """
    :param n: how many (random sample; fixed seed!)
    :param with_dependencies: whether to write dependencies to a column as well; currently as a list of pairs (no labels)
        WARNING: If not with dependencies, NO ALIGNMENT WITH DEPENDENCY RELATIONS IS GUARANTEED.
    :return:
    """
    out_file_path = os.path.basename(path_to_conllu_file)[:-7] + '{}.csv'.format(n)

    sentences = []

    for s in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
        sentences.append(s)

    random.seed(12345)

    indices = random.sample(list(range(len(sentences))), n)
    proper_indices = []
    for i in indices:
        arcs = tree_utils.conllu_to_arcs(sentences[i].to_tree())
        if len(arcs) >= 2:
            proper_indices.append(i)
    while len(proper_indices) < n:
        i = random.choice(range(len(sentences)))
        if i not in proper_indices and len(tree_utils.conllu_to_arcs(sentences[i].to_tree())) >= 2:
            proper_indices.append(i)

    indices = proper_indices
    sentences = [sentences[i] for i in indices]

    with open('data/' + out_file_path, 'w') as outfile:
        outfile.write('# index, id \n')
        writer = csv.writer(outfile)
        for i, s in zip(indices, sentences):
            row = [str(i), s.metadata['sent_id'], ' '.join([token["form"] for token in s])]
            writer.writerow(row)

    if with_dependencies:
        out_file_path_dep = out_file_path[:-4] + '-dep.conllu'
        with open('data/' + out_file_path_dep, 'w') as outfile:

            for s in sentences:
                outfile.writelines(s.serialize())



def pearson_baseline(path):

    sentences = []

    for s in parse_incr(open(path, "r", encoding="utf-8")):
        sentences.append(s)

    baseline_left_score = []
    baseline_right_score = []
    gold_score = []

    for i, s in enumerate(sentences):
        arcs = tree_utils.conllu_to_arcs(s.to_tree())

        nodes = list(set([a[j] for j in [0,1] for a in arcs]))
        nodes.sort()

        baseline_left = [(nodes[i],nodes[i-1]) for i in range(1, len(nodes))]
        baseline_right = [(nodes[i-1], nodes[i]) for i in range(1, len(nodes))]

        baseline_left_matrix = - tree_utils.arcs_to_distance_matrix(baseline_left)
        baseline_right_matrix = - tree_utils.arcs_to_distance_matrix(baseline_right)
        baseline_left_matrix_bidir = - tree_utils.arcs_to_distance_matrix(baseline_left, bidirectional=True)
        baseline_right_matrix_bidir = - tree_utils.arcs_to_distance_matrix(baseline_right, bidirectional=True)
        gold_matrix = - tree_utils.arcs_to_distance_matrix(arcs)
        gold_matrix_bidir = - tree_utils.arcs_to_distance_matrix(arcs, bidirectional=True)

        pearson_left = tree_utils.pearson_scores(baseline_left_matrix, s)
        pearson_left_bidir = tree_utils.pearson_scores(baseline_left_matrix_bidir, s)
        pearson_right = tree_utils.pearson_scores(baseline_right_matrix, s)
        pearson_right_bidir = tree_utils.pearson_scores(baseline_right_matrix_bidir, s)
        pearson_gold = tree_utils.pearson_scores(gold_matrix, s)
        pearson_gold_bidir = tree_utils.pearson_scores(gold_matrix_bidir, s)

        baseline_left_score.append(pearson_left + pearson_left_bidir)
        baseline_right_score.append(pearson_right + pearson_right_bidir)
        gold_score.append(pearson_gold + pearson_gold_bidir)

    print("PEARSON BASELINES: (plain; irreflexive; bidirectional dep; bidir-irrefl) (same for bidirectional both sides)")
    for label, scores in zip(["LEFT", "RIGHT", "GOLD"], [baseline_left_score, baseline_right_score, gold_score]):
        print("  "+label)
        means = [np.nanmean(score) for score in zip(*scores)]
        print('   ' + '\n   '.join(['{} ({})'.format(a.round(2), b.round(2)) for a,b in zip(means[::2],means[1::2])]))


def dependency_baseline(path):

    sentences = []

    for s in parse_incr(open(path, "r", encoding="utf-8")):
        sentences.append(s)


    baseline_left_score = None
    baseline_right_score = None
    gold_score = None

    for i, s in enumerate(sentences):
        arcs = tree_utils.conllu_to_arcs(s.to_tree())

        nodes = list(set([a[j] for j in [0,1] for a in arcs]))
        nodes.sort()

        baseline_left = [(nodes[i],nodes[i-1]) for i in range(1, len(nodes))]
        baseline_right = [(nodes[i-1], nodes[i]) for i in range(1, len(nodes))]

        scores_left = tree_utils.get_scores(baseline_left, s)
        scores_right = tree_utils.get_scores(baseline_right, s)
        gold_scores = tree_utils.get_scores(arcs, s)

        if i == 0:
            baseline_left_score = scores_left
            baseline_right_score = scores_right
            gold_score = gold_scores
            for dict in [baseline_left_score, baseline_right_score, gold_score]:
                for key1 in dict:
                    for key2 in dict[key1]:
                        dict[key1][key2] = [dict[key1][key2]]
        else:
            for dict1, dict2 in zip([scores_left, scores_right, gold_scores], [baseline_left_score, baseline_right_score, gold_score]):
                for key1 in dict1:
                    for key2 in dict1[key1]:
                        dict2[key1][key2].append(dict1[key1][key2])

    for dict in [baseline_left_score, baseline_right_score, gold_score]:
        for key1 in dict:
            for key2 in dict[key1]:
                dict[key1][key2] = np.nanmean(dict[key1][key2])


    print("BASELINES:")
    for label, dict in zip(["LEFT", "RIGHT", "GOLD"], [baseline_left_score, baseline_right_score, gold_score]):
        print("  " + label)
        for key1 in dict:
            print("    " + key1 + ':   ' + '  '.join([(key2 + ":" + str(dict[key1][key2])[:5]) for key2 in dict[key1]]))


def write_file_for_nominal_core_args():
    """
    Extract just the 'nominal core args', i.e., subject and objects.
    :return:
    """
    out_file_path = os.path.basename(path_to_conllu_file)[:-7]+'_nom_args.csv'

    with open('data/'+out_file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['{}|{} {}'.format('#' if i==0 else '', i, t) for i,t in enumerate(CONLLU_tags.nominal_core_arguments)])

        for sentence in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
            token_forms = []
            for token in sentence:
                if token["deprel"] in CONLLU_tags.nominal_core_arguments:
                    index = CONLLU_tags.nominal_core_arguments.index(token["deprel"])
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
                if token["upostag"] in CONLLU_tags.open_class_tags:
                    token_forms.append('|{} {} |'.format(0, token["form"].replace('|', '')))
                elif token["upostag"] in CONLLU_tags.closed_class_tags:
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



def write_file_for_coreference(n):

    path = 'data/raw/ontonotes_dev_info.tsv'
    out_file_path = os.path.basename(path)[:-4] + '_NN-PRP.csv'

    discourses = []
    with open(path) as data:
        discourse = []
        current_discourse_id = None
        sentence = []
        for i, line in enumerate(data):
            if i != 0:
                row = line.split()
                row[2] = [int(s) for s in row[2].strip('[]').split(',') if s != '']
                if row[0] != current_discourse_id:
                    if len(sentence) > 0:
                        discourse.append(sentence)
                        sentence = []
                    discourses.append(discourse)
                    discourse = []
                    current_discourse_id = row[0]
                if row[1] == '<eos>':
                    if len(sentence) > 0:
                        discourse.append(sentence)
                        sentence = []
                else:
                    sentence.append((row[0],row[1],row[2],row[3]))

    items_coref = []
    items_nocoref = []

    # tags:
    consequent_tags = ['PRP', 'PRP$']
    antecedent_tags = ['NN', 'NNS']
    for d_idx, discourse in enumerate(discourses):
        # TODO Consider longer stretches too?
        for s_idx, sentence in enumerate(discourse):
            antecedents = []    # store as (idx, span)
            consequents = []
            for idx, token in enumerate(sentence):
                if token[3] in antecedent_tags:
                    # for span in token[2][:1]:# lowest-level span only
                    if len(token[2]) > 0:
                        antecedents.append((idx, token[2][0]))
                if token[3] in consequent_tags:
                    # for span in token[2][:1]:# lowest-level span only
                    #     if span in [ante[1] for ante in antecedents]:
                    if len(token[2]) > 0:
                        consequents.append((idx, token[2][0]))
            for consequent in consequents:
                for antecedent in antecedents:
                    if antecedent[0] < consequent[0]:
                        if consequent[1] == antecedent[1]:
                            ## Check if material intervenes
                            intervener = False
                            for j in range(antecedent[0]+1, consequent[0]):
                                if consequent[1] not in sentence[j][2]:
                                    # print(consequent[1], sentence[j], sentence[j][1])
                                    intervener = True
                                    break
                            if intervener:
                                row = []
                                for idx, token in enumerate(sentence):
                                    if idx == antecedent[0]:
                                        row.append('|0 '+ token[1] + ' |')
                                    elif idx == consequent[0]:
                                        row.append('|1 ' + token[1] + ' |')
                                    # elif token[2] in antecedent_tags and consequent[1] not in token[1]:
                                    #     row.append('|2 ' + token[0] + ' |') # competing nouns get nr 2
                                    # elif token[2] in consequent_tags and consequent[1] not in token[1]:
                                    #     row.append('|3 ' + token[0] + ' |') # competing pronouns get nr 3
                                    else:
                                        row.append(token[1])
                                items_coref.append([sentence[0][0], 'coref', consequent[0] - antecedent[0], ' '.join(row).replace('| |', '|')])
                        else:
                            row = []
                            for idx, token in enumerate(sentence):
                                if idx == antecedent[0]:
                                    row.append('|0 ' + token[1] + ' |')
                                elif idx == consequent[0]:
                                    row.append('|1 ' + token[1] + ' |')
                                # elif token[2] in antecedent_tags and consequent[1] not in token[1]:
                                #     row.append('|2 ' + token[0] + ' |') # competing nouns get nr 2
                                # elif token[2] in consequent_tags and consequent[1] not in token[1]:
                                #     row.append('|3 ' + token[0] + ' |') # competing pronouns get nr 3
                                else:
                                    row.append(token[1])
                            items_nocoref.append([sentence[0][0], 'nocoref', consequent[0] - antecedent[0], ' '.join(row).replace('| |', '|')])

    # TODO Should I not randomize this?
    items_coref = items_coref[:n/2]
    items_nocoref = items_nocoref[:n-len(items_coref)]
    items = [pair[i] for pair in zip(items_coref, items_nocoref) for i in [0,1]]

    items = items[:n]

    print("Mean distance/variance:")
    print("coref:", len(items_coref), np.mean([item[2] for item in items_coref]), np.var([item[2] for item in items_coref]))
    print("nocoref:", len(items_nocoref), np.mean([item[2] for item in items_nocoref]), np.var([item[2] for item in items_nocoref]))

    print("Writing", len(items), "items to file", out_file_path)

    with open('data/'+out_file_path, 'w') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['# id, coref, distance, |0 noun,|1 pronoun,|2 other_noun,|3 other_pronoun'])

        for item in items:
            writer.writerow(item)





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



def colorless_sentences_from_categories(n):

    tuples = [('weapon', 'gun', 'shotgun'), ('clothing', 'pants', 'jeans'), ('footgear', 'shoes', 'sandals'),
              ('tool', 'saw', 'chainsaw'), ('animal', 'dog', 'poodle')]

    templates = [  # {0} noun; {1} noun, {2} noun, {3} verb, {4} adjective; {5} adverb
        'The {0} {5} {3} some {2} from the {4} {1}.',
        'The {4} {2} {5} {3} some {1} the {0}.',
        '{5} every {2}, the {4} {1} {3} some {0}.',
        'My {4} {2} {5} {3} the {0} or the {1}.',
        'When his {1} {3}, every {1} {3} {5} and his {4} {0} too.',
        'Given the {4} {1} in the {2}, the {0} {5} {3}.',
        'Because the {0} and the {4} {1} {3}, my {2} {5} {3} it.',
    ]
    # TODO Add more templates
    # TODO Add larger vocabularies
    nouns = ["apple", "economy", "book", "cover"]
    verbs = ["bought", "sold", "saw", "witnessed", "liked"]
    adjectives = ["horrible", "fun", "tall", "old"]
    adverbs = ["gently", "always", "probably"]

    colorless = []
    while len(colorless) < n:
        for template in templates:
            noun1 = random.choice(nouns)
            noun2 = random.choice(nouns)
            nounchoice = ["{0}", noun1, noun2]
            random.shuffle(nounchoice)
            verb = random.choice(verbs)
            adjective = random.choice(adjectives)
            adverb = random.choice(adverbs)
            filled = template.format(*nounchoice, verb, adjective, adverb)
            for tuple in tuples:
                for term, level in zip(tuple, ['super', 'basic', 'sub']):
                    sentence = filled.format(term)
                    # Quick dirty fix
                    for vowel in ['a', 'e', 'i', 'o', 'u']:
                        sentence = sentence.replace(' a {}'.format(vowel), ' an {}'.format(vowel))
                        sentence = sentence.replace('A {}'.format(vowel), 'An {}'.format(vowel))
                    colorless.append([level, term, sentence.capitalize()])


    # Write to csv including group tags right away
    with open('data/category-sentences-colorless.csv', 'w') as file:
        file.write('# level, |0 term, |1 rest \n')
        writer = csv.writer(file)
        for item in colorless:
            sent_with_groups = '|1 ' + item[2].replace(item[1], '|0 '+item[1] + ' |1')
            row = [item[0], sent_with_groups]
            writer.writerow(row)
    print('wrote {} items to {}'.format(len(colorless), 'data/category-sentences-colorless.csv'))



def read_all_categories():
    path = 'data/raw/sub-basic-super_triples_v0.tsv'
    categories = []
    with open(path) as file:
        for i, line in enumerate(file):
            if i != 0:
                line = line.split()
                lists = ' '.join(line[4:-1]).replace('"[','[').replace(']"', ']').split(']')
                lists = [l + ']' for l in lists if l != ""]
                lists = [eval(l) for l in lists]
                lists = [[s.replace('_', ' ') for s in l] for l in lists]
                line = line[:3] + lists + [line[-1]]
                categories.append(line)

    ## Got tons of triples now to work with

    # TODO: Build colorless sentences; simple CFG; maybe using https://github.com/gabrielilharco/sentence-generator

# def generate_sentences_from_categories_and_conllu(n, n_templates):
#
#     ## Read dictioary with replacements
#     dictionary = {
#         'NOUN': {"sing": ['apple', 'pear']},
#         'VERB': {"sing": ['sees', 'runs']},
#         'ADJ': ["funny", "stupid"]
#     }
#
#     ## Get sentences from conllu
#     out_file_path = os.path.basename('category-sentences-colorless.csv')
#
#     sentences = []
#
#     for s in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
#         sentences.append(s)
#
#     random.seed(12345)
#
#     ## Get only a sample, and only those containing singular nouns
#     def has_singular_noun(s):
#         for token in s:
#             if token["upostag"] == "NOUN" and token["feats"] is not None and "Number" in token["feats"] and token["feats"]["Number"] == "Sing":
#                 return True
#         return False
#
#     indices = []
#     examples = []
#     while len(examples) < n_templates:
#         i = random.choice(list(range(len(sentences))))
#         if i not in indices and has_singular_noun(sentences[i]) and len(sentences[i]) > 8:
#             indices.append(i)
#             examples.append(sentences[i])
#
#     ## Generate templates from the sample
#     templates = []
#     for ex in examples:
#         template = []
#         for token in ex:
#             replace = False
#             if token["upostag"] in dictionary:
#                 replace = True
#             template.append((token["upostag"], token["feats"]) if replace else token["form"])
#         templates.append(template)
#
#     for t in templates:
#         print(t)
#         print('---------')
#
#     quit()
#
#     random.choice(dictionary[token["upostag"]][token["feats"]["Number"]])
#
#
#     ## Read tuples of categories
#     rows = []
#     tuples_dict = {}
#     tuples = []
#     with open('data/auxiliary/category-sentences.txt') as file:
#         for line in file:
#             if not line.startswith('#') and not line.strip() == '':
#                 row = [s.strip() for s in line.split(';')]
#                 rows.append(row)
#     if len(rows) %3  != 0:
#         print("Something's wrong")
#     for i in range(0, len(rows), 3):
#         tuple = [s[1] for s in rows[i:i+3]]
#         tuples.append(tuple)
#         tuples_dict[tuple[0]] = tuples_dict[tuple[1]] = tuples_dict[tuple[2]] = tuple
#
#
#     ## Combine and write
#     # Write to csv including group tags right away
#     with open(out_file_path, 'w') as file:
#         file.write('# original, level, source, |0 term, |1 rest \n')
#         writer = csv.writer(file)
#         for sentence in sentences:
#             tokens = []
#             noun_ids = []
#
#             for i, token in enumerate(sentence):
#                 if token["upostag"] == "NOUN" and token["feats"]["Number"] == "sing":
#                     noun_ids.append(i)
#                 tokens.append(token)
#             to_be_replaced_idx = random.choice(noun_ids)
#
#             sent_with_groups = '|1 ' + item[4].replace(item[3], '|0 ' + item[3] + ' |1')
#             row = [item[0], item[1], item[2], sent_with_groups]
#             writer.writerow(row)
#
#     with open('data/'+out_file_path, 'w') as outfile:
#         writer = csv.writer(outfile)
#         writer.writerow(['{}|{} {}'.format('#' if i==0 else '', i, t) for i,t in enumerate(tags_of_interest)])
#
#         for sentence in parse_incr(open(path_to_conllu_file, "r", encoding="utf-8")):
#             token_forms = []
#             for token in sentence:
#                 print(token("feats"))
#                 if token["upostag"] == "NOUN" and token["feats"]["Number"] == "sing":
#                     token_forms.append('|{} {} |'.format(index, token["form"].replace('|','')))
#                 else:
#                     token_forms.append(token["form"])
#             writer.writerow([' '.join(token_forms)])
#
#
#         artificial = 'I wonder if my sister will bring her {} to the meeting.'
#         for tuple in tuples:
#             for term, level in zip(tuple, ['super', 'basic', 'sub']):
#                 all_items.append((level, level, 'artificial', term, artificial.format(term)))


def parse_data(data_path, tokenizer, max_items=None, words_as_groups=False, dependencies=False):
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

        for digit, each_part in regex.findall(group_regex, row[-1]):
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

    # TODO change the way groups are represented, just a single 'groups' column containing lists.

    # Create dataframe with nice column names
    columns = factor_names + ['sentence'] + ['tokenized'] + group_names

    items = pd.DataFrame(items, columns=columns)

    # Add a bunch of useful metadata to the DataFrame
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        items.num_factors = num_factors
        items.factors = factor_names
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

    ## If no dependencies requested, you're done.
    if not dependencies:
        return items

    ## Otherwise read dependencies
    dependency_path = data_path[:-4] + "-dep.conllu"
    dependencies = []
    for sentence in parse_incr(open(dependency_path, "r", encoding="utf-8")):
        dependencies.append(sentence)
        if len(dependencies) == len(items):
            break

    if len(dependencies) != len(items):
        print("WARNING: Something's wrong.")

    return items, dependencies



def merge_grouped_tokens(items, data_for_all_items, method="mean"):
    """
    Takes weights matrix per item per layer, and averages rows and columns based on desired token groups.
    :param items: dataframe as read from example.csv
    :param data_for_all_items: list of numpy arrays with attention/gradients extracted from BERT
    :param method: mean or sum
    :return: list of (for each item) a numpy array layer x num_groups x num_groups
    """
    # TODO Ideally this would be done still on cuda
    data_for_all_items2 = []

    for (_, each_item), weights_per_layer in tqdm(zip(items.iterrows(), data_for_all_items), total=len(items)):

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

            if len(data_for_all_items[0][0].shape) > 1:
                # Group the result also vertically
                grouped_weights = []
                for group in items.groups:
                    if method == "mean":
                        grouped_weights.append(grouped_weights_horiz[:, each_item[group]].mean(axis=1))
                    if method == "sum":
                        grouped_weights.append(grouped_weights_horiz[:, each_item[group]].sum(axis=1))
                grouped_weights = np.stack(grouped_weights).transpose()  # transpose to restore original order
            else:
                grouped_weights = grouped_weights_horiz

            # store
            data_per_layer.append(grouped_weights)

        data_for_all_items2.append(np.stack(data_per_layer))

    return data_for_all_items2



if __name__ == "__main__":
    # write_file_plain_sentences(500, with_dependencies=True)
    # write_file_for_open_vs_closed_POS()
    # write_file_for_main_POS()

    # write_file_for_coreference(500)

    read_all_categories()
    quit()

    pass
    # colorless_sentences_from_categories(500)
    # generate_sentences_from_categories_and_conllu(100, 20)

    # dependency_baseline("data/en_gum-ud-dev500-dep.conllu")
    pearson_baseline("data/en_gum-ud-dev500-dep.conllu")
    # dependency_baseline("data/en_ewt-ud-train500-dep.conllu")
    # write_file_plain_sentences(500, with_dependencies=True)
