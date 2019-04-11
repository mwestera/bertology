from conllu import parse
from conllu import parse_incr
import os
import csv

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


