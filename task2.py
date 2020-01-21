from nltk.grammar import Nonterminal
from nltk.grammar import toy_pcfg2
from nltk.probability import DictionaryProbDist



'''
productions = toy_pcfg2.productions()
# Get all productions with LHS=NP
np_productions = toy_pcfg2.productions(Nonterminal('NP'))

dict = {}
for pr in np_productions: dict[pr.rhs()] = pr.prob()
np_probDist = DictionaryProbDist(dict)

# Each time you call, you get a random sample
print(np_probDist.generate())
'''


def pcfg_generate(grammar):

    def non_terminal_into_terminal(non_terminal):
        nt_productions = grammar.productions(Nonterminal(str(non_terminal)))
        my_dict = dict()
        for pr in nt_productions: my_dict[pr.rhs()] = pr.prob()
        nt_productions_probDist = DictionaryProbDist(my_dict)
        genereted = nt_productions_probDist.generate()
        return genereted

    # todo it does not work right recursivly when i move to the next nt i remove the old one somehow
    def nts_into_ts(genereted_nts):
        for index in range(len(genereted_nts)):
            old_nt = genereted_nts[index]
            try:
                t = non_terminal_into_terminal(genereted_nts[index])
            except Exception as e:
                ans_genereted_nts = genereted_nts
                continue
            ans_genereted_nts = genereted_nts[:index] + ((old_nt, t),) + genereted_nts[index+1:]
            ans_genereted_nts = ans_genereted_nts[:index] + ((old_nt, nts_into_ts(t)),) + ans_genereted_nts[index+1:]
        return ans_genereted_nts

    productions = grammar.productions()
    dic = dict()
    for pr in productions: dic[pr.rhs()] = pr.prob()
    productions_probDist = DictionaryProbDist(dic)
    genereted = productions_probDist.generate()
    return nts_into_ts(genereted)


for i in range(1000):
    print(pcfg_generate(toy_pcfg2))
