from nltk.corpus import LazyCorpusLoader, BracketParseCorpusReader
from nltk import PCFG
from nltk.grammar import ProbabilisticProduction
from nltk import Tree, Nonterminal
from nltk.probability import FreqDist


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
    dist = FreqDist(probabilities)
    dist.plot(len(probabilities))

    return productions_to_return


def pcfg_learn1(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree)
            tree_to_append = next(prod_gen)
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP'):
                    productions.append(tree_to_append)
                try:
                    tree_to_append = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions)


def pcfg_learn2(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree)
            tree_to_append = next(prod_gen)
            father_title = Nonterminal('BOT')
            # todo i am taking the father title in the wrong way because i am not getting vp as father of np -
            #  looked on the method and understood why very easily - need to maybe modify the method. i can yield the
            #  parent's label as well and then little modifications will be required
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP') and father_title == Nonterminal('S'):
                    productions.append(tree_to_append)
                try:
                    father_title = tree_to_append.lhs()
                    tree_to_append = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions)


def pcfg_learn3(treebank, n):
    productions = list()
    for i in range(n):
        for tree in treebank.parsed_sents()[:i+1]:
            prod_gen = tree_to_productions(tree)
            tree_to_append = next(prod_gen)
            father_title = Nonterminal('BOT')
            while tree_to_append:
                if tree_to_append.lhs() == Nonterminal('NP') and father_title == Nonterminal('VP'):
                    productions.append(tree_to_append)
                try:
                    father_title = tree_to_append.lhs()
                    tree_to_append = next(prod_gen)
                except Exception as e:
                    tree_to_append = False
    productions = get_productions(productions)
    return PCFG(Nonterminal('NP'), productions)


print(pcfg_learn3(treebank, 200))
