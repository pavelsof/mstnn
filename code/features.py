from collections import OrderedDict



"""
Constants
"""

"""
Tuple listing all possible universal POS tags
http://universaldependencies.org/u/pos/index.html
"""
POS_TAGS = (
	'ADJ',  # adjective
	'ADP',  # adposition
	'ADV',  # adverb
	'AUX',  # auxiliary
	'CCONJ',  # coordinating conjunction
	'DET',  # determiner
	'INTJ',  # interjection
	'NOUN',  # noun
	'NUM',  # numeral
	'PART',  # particle
	'PRON',  # pronoun
	'PROPN',  # proper noun
	'PUNCT',  # punctuation
	'SCONJ',  # subordinating conjunction
	'SYM',  # symbol
	'VERB',  # verb
	'X',  # other
)

"""
Tuple listing all possible universal dependency relations
http://universaldependencies.org/u/dep/index.html
"""
DEP_RELS = (
	'acl',  # clausal modifier of noun (adjectival clause)
	'advcl',  # adverbial clause modifier
	'advmod',  # adverbial modifier
	'amod',  # adjectival modifier
	'appos',  # appositional modifier
	'aux',  # auxiliary
	'case',  # case marking
	'cc',  # coordinating conjunction
	'ccomp',  # clausal complement
	'clf',  # classifier
	'compound',  # compound
	'conj',  # conjunct
	'cop',  # copula
	'csubj',  # clausal subject
	'dep',  # unspecified dependency
	'det',  # determiner
	'discourse',  # discourse element
	'dislocated',  # dislocated elements
	'expl',  # expletive
	'fixed',  # fixed multiword expression
	'flat',  # flat multiword expression
	'goeswith',  # goes with
	'iobj',  # indirect object
	'list',  # list
	'mark',  # marker
	'nmod',  # nominal modifier
	'nsubj',  # nominal subject
	'nummod',  # numeric modifier
	'obj',  # object
	'obl',  # oblique nominal
	'orphan',  # orphan
	'parataxis',  # parataxis
	'punct',  # punctuation
	'reparandum',  # overridden disfluency
	'root',  # root
	'vocative',  # vocative
	'xcomp',  # open clausal complement
)


"""
Dictionary where the keys are the possible universal features and the values
are the respective features' possible values.

http://universaldependencies.org/u/feat/index.html
"""
MORPH_FEATURES = {
	'Abbr': ('Yes',),
	'Animacy': ('Anim', 'Hum ', 'Inan ', 'Nhum'),
	'Aspect': ('Hab', 'Imp ', 'Iter', 'Perf', 'Prog', 'Prosp'),
	'Case': ('Abs', 'Acc', 'Erg', 'Nom',
		'Abe', 'Ben', 'Cau', 'Cmp', 'Com', 'Dat', 'Dis',
		'Equ', 'Gen', 'Ins', 'Par', 'Tem', 'Tra', 'Voc',
		'Abl', 'Add', 'Ade', 'All', 'Del', 'Ela', 'Ess',
		'Ill', 'Ine', 'Lat', 'Loc', 'Sub', 'Sup', 'Ter'),
	'Definite': ('Com', 'Cons', 'Def', 'Ind', 'Spec'),
	'Degree': ('Abs', 'Cmp', 'Equ', 'Pos', 'Sup'),
	'Evident': ('Fh', 'Nfh'),
	'Foreign': ('Yes',),
	'Gender': ('Com', 'Fem', 'Masc', 'Neut'),
	'Mood': ('Adm', 'Cnd', 'Des', 'Imp', 'Ind', 'Jus',
		'Nec', 'Opt', 'Pot', 'Prp', 'Qot', 'Sub'),
	'NumType': ('Card', 'Dist', 'Frac', 'Mult', 'Ord', 'Range', 'Sets'),
	'Number': ('Coll', 'Count', 'Dual', 'Grpa', 'Grpl',
		'Inv', 'Pauc', 'Plur', 'Ptan', 'Sing', 'Tri'),
	'Person': ('0', '1', '2', '3', '4'),
	'Polarity': ('Neg', 'Pos'),
	'Polite': ('Elev', 'Form', 'Humb', 'Infm'),
	'Poss': ('Yes',),
	'PronType': ('Art', 'Dem', 'Emp', 'Exc', 'Ind', 'Int',
		'Neg', 'Prs', 'Rcp', 'Rel', 'Tot'),
	'Reflex': ('Yes',),
	'Tense': ('Fut', 'Imp', 'Past', 'Pqp', 'Pres'),
	'VerbForm': ('Conv', 'Fin', 'Gdv', 'Ger', 'Inf', 'Part', 'Sup', 'Vnoun'),
	'Voice': ('Act', 'Antip', 'Cau', 'Dir', 'Inv', 'Mid', 'Pass', 'Rcp')
}

MORPH_FEATURES = OrderedDict(sorted(MORPH_FEATURES.items(), key=lambda x: x[0]))



"""
Functions
"""

def featurise_pos_tag(pos_tag):
	"""
	Returns a one-hot vector for the given POS tag. Raises a ValueError if the
	given string is not a universal POS tag.
	"""
	vector = [0] * len(POS_TAGS)
	vector[POS_TAGS.index(pos_tag)] = 1
	
	return vector


def featurise_dep_rel(dep_rel):
	"""
	Returns a one-hot vector for the given dependency relation. Raises a
	ValueError if the given string is not a universal dependency relation.
	"""
	vector = [0] * len(DEP_RELS)
	vector[DEP_RELS.index(dep_rel)] = 1
	
	return vector


def featurise_morph(morph):
	"""
	Returns the feature vector corresponding to the given FEATS string. Raises
	an Exception if the string does not conform to the rules.
	"""
	try:
		morph = {
			key: value.split(',')
			for key, value in map(lambda x: x.split('='), morph.split('|'))}
	except:
		if not morph:
			morph = {}
		else:
			raise
	
	vector = []
	
	for feature, poss_values in MORPH_FEATURES.items():
		small_vec = [0] * len(poss_values)
		
		if feature in morph:
			for index, value in enumerate(poss_values):
				if value in morph[feature]:
					small_vec[index] = 1
		
		vector += small_vec
	
	return vector
