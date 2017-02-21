import networkx as nx
import numpy as np



"""
Errors
"""

class FeatureError(ValueError):
	"""
	Raised when an unrecognised input (i.e. non-existant POS tag) is given to
	any of the featurise_* functions. Inited with a user-friendly message.
	"""
	pass



"""
Functions
"""

def featurise_pos_tag(pos_tag):
	"""
	Returns the feature vector for the given POS tag. Raises a FeatureError if
	the given string is not a universal POS tag or 'ROOT'.
	
	The vector is a numpy array of zeroes and a single 1, the latter being at
	the index in POS_TAGS that corresponds to the given tag.
	"""
	try:
		vector = [0] * len(POS_TAGS)
		vector[POS_TAGS.index(pos_tag)] = 1
	except ValueError:
		raise FeatureError('Unknown POS tag: {}'.format(pos_tag))
	
	return np.array(vector)



def featurise_dep_rel(dep_rel):
	"""
	Returns the feature vector for the given dependency relation. Raises a
	FeatureError if the given string is not a universal dependency relation.
	
	The vector is a numpy array of zeroes and a single 1, the latter being at
	the index in DEP_RELS that corresponds to the given dependency relation.
	"""
	try:
		vector = [0] * len(DEP_RELS)
		vector[DEP_RELS.index(dep_rel)] = 1
	except ValueError:
		raise FeatureError('Unknown dependency relation: {}'.format(dep_rel))
	
	return np.array(vector)



def featurise_morph(morph):
	"""
	Returns the feature vector corresponding to the given FEATS string. Raises
	a FeatureError if the string does not conform to the rules.
	
	The vector is a numpy array of zeroes and ones with each element
	representing a possible value of the MORPH_FEATURES ordered dict. E.g. the
	output for "Animacy=Anim" should be a vector with its second element 1 and
	all the other elements zeroes.
	"""
	try:
		morph = {
			key: value.split(',')
			for key, value in map(lambda x: x.split('='), morph.split('|'))}
	except ValueError:
		if morph == '_':
			morph = {}
		else:
			raise FeatureError('Bad FEATS format: {}'.format(morph))
	
	vector = []
	
	for feature, poss_values in MORPH_FEATURES.items():
		small_vec = [0] * len(poss_values)
		
		if feature in morph:
			for index, value in enumerate(poss_values):
				if value in morph[feature]:
					small_vec[index] = 1
		
		vector += small_vec
	
	return np.array(vector)



def featurise_edge(graph, edge):
	"""
	Returns the feature vector for the edge in the given graph that is defined
	by the given nodes tuple. Non-edges can also be featurised.
	
	The return value is a numpy array that is the result of concatenating the
	parent and child's POS tag and morphology feature vectors.
	"""
	parent = graph.node[edge[0]]
	child = graph.node[edge[1]]
	
	return np.concatenate([
		featurise_pos_tag(parent['UPOSTAG']),
		featurise_morph(parent['FEATS']),
		featurise_pos_tag(child['UPOSTAG']),
		featurise_morph(child['FEATS'])])



def featurise_graph(graph):
	"""
	Returns the 3D feature matrix extracted from the given nx.DiGraph instance.
	The latter is expected to be of the type that conllu.Dataset.gen_graphs()
	produces.
	
	The matrix has the width and height equal to the number of words in the
	sentence (including the imaginary root word). The depth consists of (1) the
	adjacency 2D matrix, (2) the POS tag feature vectors, and (3) the
	morphology feature vectors, stacked in this order.
	
	The return value itself is a numpy array of shape (depth, height, width).
	"""
	num_nodes = graph.number_of_nodes()
	
	adj_mat = np.expand_dims(nx.adjacency_matrix(graph).todense(), axis=0)
	
	pos_mat = np.array([featurise_pos_tag(graph.node[node]['UPOSTAG'])
		for node in range(num_nodes)])
	pos_mat = np.tile(pos_mat, (num_nodes, 1, 1))
	pos_mat = np.swapaxes(np.swapaxes(pos_mat, 0, 2), 1, 2)
	
	morph_mat = np.array([featurise_morph(graph.node[node]['FEATS'])
		for node in range(num_nodes)])
	morph_mat = np.tile(morph_mat, (num_nodes, 1, 1))
	morph_mat = np.swapaxes(np.swapaxes(morph_mat, 0, 2), 1, 2)
	
	return np.concatenate((adj_mat, pos_mat, morph_mat), axis=0)
