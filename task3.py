from nltk.parse.viterbi import ViterbiParser
from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk import PCFG
from nltk.grammar import ProbabilisticProduction
from nltk import Tree, Nonterminal
from nltk.treetransforms import chomsky_normal_form


def simplify_functional_tag(tag):
    if '-' in tag:
        tag = tag.split('-')[0]
    return tag


def get_tag(tree):
    if isinstance(tree, Tree):
        return Nonterminal(simplify_functional_tag(tree.label()))
    else:
        return tree


def tree_to_production(tree):
    return ProbabilisticProduction(get_tag(tree), [get_tag(child) for child in tree], **{'prob': 0})


def tree_to_productions(tree):
    yield tree_to_production(tree)
    for child in tree:
        if isinstance(child, Tree):
            for prod in tree_to_productions(child):
                yield prod


treebank = LazyCorpusLoader('treebank/combined', BracketParseCorpusReader, r'wsj_.*\.mrg')


def get_productions(productions):

    probabilities = dict()
    productions_to_return = list(set(productions))

    for prod in productions:
        if str(prod) in probabilities:
            probabilities[str(prod)] += 1
        else:
            probabilities[str(prod)] = 1

    lhs_of_prods = set([prod.lhs() for prod in productions])

    for lhs in lhs_of_prods:
        number_of_occurrences = 0
        for prob in probabilities:
            if prob.startswith(str(lhs) + " "):
                number_of_occurrences += probabilities[prob]
        for prob in probabilities:
            if prob.startswith(str(lhs) + " "):
                probabilities[prob] = probabilities[prob] / number_of_occurrences

    for index in range(len(productions_to_return)):
        prod = productions_to_return[index]
        productions_to_return[index] = ProbabilisticProduction(prod.lhs(), prod.rhs(),
                                                               **{'prob': probabilities[str(prod)]})

    return productions_to_return


def pcfg_learn(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            chomsky_normal_form(tree, factor='right', horzMarkov=1, vertMarkov=1, childChar='|', parentChar='^')
            prod_gen = tree_to_productions(tree)
            tree_to_append = next(prod_gen)
            while tree_to_append:
                productions.append(tree_to_append)
                try:
                    tree_to_append = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions = get_productions(productions)
    return PCFG(Nonterminal('S'), productions)


pcfg_training = pcfg_learn(treebank, 10)
parser = ViterbiParser(pcfg_training)


parse_tree = parser.parse_all('Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29'
                              ' .'.split())


def find_first_index(parse_tree):
    if isinstance(parse_tree[0], Tree):
        return find_first_index(parse_tree[0])
    return parse_tree[0]


def find_last_index(parse_tree):
    if isinstance(parse_tree[-1], Tree):
        return find_last_index(parse_tree[-1])
    return parse_tree[-1][0]


def get_list_of_labelled_constituents(parse_tree):
    lst_to_return = list()
    label = parse_tree.label()
    first_index = find_first_index(parse_tree)
    last_index = find_last_index(parse_tree)
    lst_to_return.append((label, first_index, last_index))
    for child in parse_tree:
        if isinstance(child, Tree):
            lst_to_return.append(get_list_of_labelled_constituents(child))
    return lst_to_return


print(get_list_of_labelled_constituents(parse_tree[0]))
