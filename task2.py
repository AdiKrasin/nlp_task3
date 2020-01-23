from nltk.grammar import Nonterminal
from nltk.grammar import toy_pcfg2
from nltk.probability import DictionaryProbDist
from nltk.tree import Tree
import os
from nltk.grammar import ProbabilisticProduction, PCFG

'''
def pcfg_generate(grammar):

    def non_terminal_into_terminal(non_terminal):
        nt_productions = grammar.productions(Nonterminal(str(non_terminal)))
        my_dict = dict()
        for pr in nt_productions: my_dict[pr.rhs()] = pr.prob()
        nt_productions_probDist = DictionaryProbDist(my_dict)
        genereted = nt_productions_probDist.generate()
        return list(genereted)

    def nts_into_ts(genereted_nts):
        for index in range(len(genereted_nts)):
            old_nt = genereted_nts[index]
            try:
                t = non_terminal_into_terminal(genereted_nts[index])
            except Exception as e:
                continue
            genereted_nts[index] = nts_into_ts(Tree(old_nt, t))
        return genereted_nts

    productions = grammar.productions()
    dic = dict()
    for pr in productions: dic[pr.rhs()] = pr.prob()
    productions_probDist = DictionaryProbDist(dic)
    genereted = productions_probDist.generate()
    genereted = Tree('S', [genereted[0], genereted[1]])
    return nts_into_ts(genereted)


file_content = ""
for i in range(1000):
    res_tree = pcfg_generate(toy_pcfg2)
    file_content += str(res_tree) + "\n"

if os.path.exists(".\\toy_pcfg2.gen"):
    os.remove(".\\toy_pcfg2.gen")
else:
    with open(".\\toy_pcfg2.gen", "w+") as f:
        f.write(file_content)
'''


def induce_pcfg(start, productions):
    """
    Induce a PCFG grammar from a list of productions.

    The probability of a production A -> B C in a PCFG is:

    |                count(A -> B C)
    |  P(B, C | A) = ---------------       where \* is any right hand side
    |                 count(A -> \*)

    :param start: The start symbol
    :type start: Nonterminal
    :param productions: The list of productions that defines the grammar
    :type productions: list(Production)
    """
    # Production count: the number of times a given production occurs
    pcount = {}

    # LHS-count: counts the number of times a given lhs occurs
    lcount = {}

    for prod in productions:
        lcount[prod.lhs()] = lcount.get(prod.lhs(), 0) + 1
        pcount[prod] = pcount.get(prod, 0) + 1

    prods = [
        ProbabilisticProduction(p.lhs(), p.rhs(), prob=pcount[p] / lcount[p.lhs()])
        for p in pcount
    ]
    return PCFG(start, prods)


lst_of_nts = list()
productions = toy_pcfg2.productions()
for production in productions:
    lst_of_nts.append(production.lhs())
for nt in lst_of_nts:
    fd = induce_pcfg(nt, productions)
    print(fd)
