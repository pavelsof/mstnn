from collections import defaultdict

import json

import h5py

import networkx as nx
import numpy as np

from code import ud



class FeatureError(ValueError):
	"""
	Raised when an unrecognised input (i.e. non-existant POS tag) is given to
	any of the featurise_* methods. Inited with a user-friendly message.
	"""
	pass



class Extractor:
	"""
	Makes the first pass over the data, collecting what is necessary in order
	to determine the dimensions of the neural network. Provides the featurise_*
	methods (which would otherwise need to expect the UD version as an arg).
	"""
	
	def __init__(self, ud_version=2):
		"""
		Constructor. The keyword argument specifies the UD version to use when
		featurising the POS tags, dependency relations, and morphology. Raises
		a ValueError if the UD version is unknown.
		"""
		if ud_version == 1:
			self.POS_TAGS = ud.POS_TAGS_V1
			self.DEP_RELS = ud.DEP_RELS_V1
			self.MORPH_FEATURES = ud.MORPH_FEATURES_V1
		elif ud_version == 2:
			self.POS_TAGS = ud.POS_TAGS_V2
			self.DEP_RELS = ud.DEP_RELS_V2
			self.MORPH_FEATURES = ud.MORPH_FEATURES_V2
		else:
			raise ValueError('Unknown UD version: {}'.format(ud_version))
		
		self.ud_version = ud_version
		self.lemmas = defaultdict(lambda: len(self.lemmas))
	
	
	@classmethod
	def create_from_model_file(cls, model_fp):
		"""
		Returns a new Extractor instance with self.ud_version and self.lemmas
		loaded from the specified model file. The latter is expected to be a
		hdf5 file written or appended to by the method below.
		
		Raises an OSError if the file does not exist or cannot be read.
		"""
		f = h5py.File(model_fp, 'r')
		
		ud_version = f['extractor'].attrs['ud_version']
		lemmas = json.loads(f['extractor'].attrs['lemmas'])
		
		f.close()
		
		extractor = Extractor(ud_version)
		
		for key, value in lemmas.items():
			extractor.lemmas[key] = value
		
		return extractor
	
	
	def write_to_model_file(self, model_fp):
		"""
		Appends to the specified hdf5 file, storing the UD version and the
		extracted lemmas. Thus, an identical Extractor can be later restored
		using the above class method.
		
		Raises an OSError if the file cannot be written.
		"""
		f = h5py.File(model_fp, 'a')
		
		group = f.create_group('extractor')
		group.attrs['ud_version'] = self.ud_version
		group.attrs['lemmas'] = json.dumps(dict(self.lemmas), ensure_ascii=False)
		
		f.flush()
		f.close()
	
	
	def read(self, dataset):
		"""
		Reads the data provided by the given conllu.Dataset instance and
		compiles the self.lemmas dict.
		"""
		self.lemmas['_']
		
		for sent in dataset.gen_sentences():
			[self.lemmas[word.LEMMA] for word in sent]
	
	
	def get_lemma_vocab_size(self):
		"""
		"""
		return len(self.lemmas)
	
	
	def featurise_lemma(self, lemma):
		"""
		Returns an integer uniquely identifying the given lemma. Raises a
		FeatureError if the lemma has not been found during reading.
		"""
		if lemma not in self.lemmas:
			raise FeatureError('Unknown lemma: {}'.format(lemma))
		
		return self.lemmas[lemma]
	
	
	def featurise_pos_tag(self, pos_tag):
		"""
		Returns the feature vector for the given POS tag. Raises a FeatureError
		if the given string is not a universal POS tag or 'ROOT'.
		
		The vector is a numpy array of zeroes and a single 1, the latter being
		at the index in POS_TAGS that corresponds to the given tag.
		"""
		try:
			vector = [0] * len(self.POS_TAGS)
			vector[self.POS_TAGS.index(pos_tag)] = 1
		except ValueError:
			raise FeatureError('Unknown POS tag: {}'.format(pos_tag))
		
		return np.array(vector)
	
	
	def featurise_dep_rel(self, dep_rel):
		"""
		Returns the feature vector for the given dependency relation. Raises a
		FeatureError if the given string is not a universal dependency
		relation.
		
		The vector is a numpy array of zeroes and a single 1, the latter being
		at the index in DEP_RELS that corresponds to the given dependency
		relation.
		"""
		try:
			vector = [0] * len(self.DEP_RELS)
			vector[self.DEP_RELS.index(dep_rel)] = 1
		except ValueError:
			raise FeatureError('Unknown dependency relation: {}'.format(dep_rel))
		
		return np.array(vector)
	
	
	def featurise_morph(self, morph):
		"""
		Returns the feature vector corresponding to the given FEATS string.
		Raises a FeatureError if the string does not conform to the rules.
		
		The vector is a numpy array of zeroes and ones with each element
		representing a possible value of the MORPH_FEATURES ordered dict. E.g.
		the output for "Animacy=Anim" should be a vector with its second
		element 1 and all the other elements zeroes.
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
		
		for feature, poss_values in self.MORPH_FEATURES.items():
			small_vec = [0] * len(poss_values)
			
			if feature in morph:
				for index, value in enumerate(poss_values):
					if value in morph[feature]:
						small_vec[index] = 1
			
			vector += small_vec
		
		return np.array(vector)
	
	
	def featurise_edge(self, graph, edge):
		"""
		Returns the feature vector for the edge in the given graph that is
		defined by the given nodes tuple. Non-edges can also be featurised.
		
		The return value is a numpy array that is the result of concatenating
		the parent and child's POS tag and morphology feature vectors.
		"""
		parent = graph.node[edge[0]]
		child = graph.node[edge[1]]
		
		return np.concatenate([
			self.featurise_pos_tag(parent['UPOSTAG']),
			self.featurise_morph(parent['FEATS']),
			self.featurise_pos_tag(child['UPOSTAG']),
			self.featurise_morph(child['FEATS'])])
	
	
	def featurise_graph(self, graph):
		"""
		Returns the 3D feature matrix extracted from the given nx.DiGraph
		instance.  The latter is expected to be of the type that
		conllu.Dataset.gen_graphs() produces.
		
		The matrix has the width and height equal to the number of words in the
		sentence (including the imaginary root word). The depth consists of (1)
		the adjacency 2D matrix, (2) the POS tag feature vectors, and (3) the
		morphology feature vectors, stacked in this order.
		
		The return value itself is a numpy array of shape (depth, height,
		width).
		"""
		num_nodes = graph.number_of_nodes()
		
		adj_mat = np.expand_dims(nx.adjacency_matrix(graph).todense(), axis=0)
		
		pos_mat = np.array([self.featurise_pos_tag(graph.node[node]['UPOSTAG'])
			for node in range(num_nodes)])
		pos_mat = np.tile(pos_mat, (num_nodes, 1, 1))
		pos_mat = np.swapaxes(np.swapaxes(pos_mat, 0, 2), 1, 2)
		
		morph_mat = np.array([self.featurise_morph(graph.node[node]['FEATS'])
			for node in range(num_nodes)])
		morph_mat = np.tile(morph_mat, (num_nodes, 1, 1))
		morph_mat = np.swapaxes(np.swapaxes(morph_mat, 0, 2), 1, 2)
		
		return np.concatenate((adj_mat, pos_mat, morph_mat), axis=0)
