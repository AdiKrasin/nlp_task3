import nltk

sg = """
S -> NP VP | S NP | S PREP NP | S CONJ NP
VP -> IV | TV NP
NP -> NP CONJ NP | SNNom SN | PNNom PN | SNDet SN | PNDet PN | 'John' | 'bread' | 'Mary' | 'They' | 'her' | 'She' | 'them' | 'Everybody' | 'it' | 'butter' | 'men' | 'women' | 'children' | 'men,' | 'cheese'
SNNom -> SNDet Adj
PNNom -> PNDet Adj
Adj -> 'heavy'
PREP -> 'to' | 'with' |'on'
SNDet -> 'A' | 'The' | 'a' | 'the'
PNDet -> 'many' | 'The' | 'Some' | 'the'
SN -> 'boy' | 'chair' | 'book' | 'man' | 'telescope' | 'hill'
PN -> 'boys'
IV -> 'left'
TV -> 'eats' | 'loves' | 'love' | 'gave' | 'likes' | 'moves' | 'saw' | 'knows' | 'eat'
CONJ -> 'and'
"""

g = nltk.CFG.fromstring(sg)

# Bottom-up  parser
sr_parser = nltk.ShiftReduceParser(g, trace=2)


# Parse sentences and observe the behavior of the parser
def parse_sentence(sent):
    tokens = sent.split()
    trees = sr_parser.parse(tokens)
    for tree in trees:
        print(tree)

'''
parse_sentence("John left")
# should be:
# (S (NP John) (VP (IV left)))
parse_sentence("John eats bread")
# should be:
# (S (NP John) (VP (TV eats) (NP bread)))
parse_sentence("John loves Mary")
parse_sentence("They love Mary")
parse_sentence("They love her")
parse_sentence("She loves them")
parse_sentence("Everybody loves John")
parse_sentence("A boy loves Mary")
parse_sentence("The boy loves Mary")
parse_sentence("Some boys love Mary")
parse_sentence("John gave Mary a heavy book")
parse_sentence("John gave it to Mary")
parse_sentence("John likes butter")
parse_sentence("John moves a chair")


parse_sentence('John moves Mary to Mary')
parse_sentence('John eats John')
'''
'''
parse_sentence('John saw a man with a telescope')
parse_sentence('John saw a man on the hill with a telescope')
parse_sentence('Mary knows men and women')
parse_sentence('Mary knows men, children and women')
parse_sentence('John and Mary eat bread')
parse_sentence('John and Mary eat bread with cheese')
'''

parse_sentence('John saw a man with a telescope a telescope')
