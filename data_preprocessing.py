from conllu import parse
from conllu import parse_incr
import os
import csv
import random

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
    and generates sentences by replacing each term by terms of other levels of categorization, written in a format readable by main.py
    :return:
    """
    sentences = []
    tuples = {}
    with open('data/auxiliary/category-sentences.txt') as file:
        for line in file:
            if not line.startswith('#') and not line.strip() == '':
                row = [s.strip() for s in line.split(';')]
                sentences.append(row)
    if len(sentences) %3  != 0:
        print("Something's wrong")
    for i in range(0, len(sentences), 3):
        tuple = [s[1] for s in sentences[i:i+3]]
        tuples[tuple[0]] = tuples[tuple[1]] = tuples[tuple[2]] = tuple

    all_items = []
    for original in sentences:
        all_items.append((original[0], original[0], original[1], original[2]))
        for i, level in enumerate(['super', 'basic', 'sub']):
            if level != original[0]:
                replacement = tuples[original[1]][i]
                new_sentence = original[2].replace(original[1], replacement)
                # Quick dirty fix
                for vowel in ['a', 'e', 'i', 'o', 'u']:
                    new_sentence = new_sentence.replace(' a {}'.format(vowel), ' an {}'.format(vowel))
                new_item = (original[0], level, replacement, new_sentence)
                all_items.append(new_item)

    # Write to csv including group tags right away
    with open('data/auxiliary/category-sentences.csv', 'w') as file:
        file.write('# original, level, |0 term, |1 rest \n')
        writer = csv.writer(file)
        for item in all_items:
            sent_with_groups = '|2 ' + item[3].replace(item[2], '|1 '+item[2] + ' |2 ')
            row = [item[0], item[1], sent_with_groups]
            writer.writerow(row)

    print('wrote {} items to {}'.format(len(all_items), 'data/category-sentences.csv'))


generate_sentences_from_categories()