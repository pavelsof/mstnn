"""
This module knows how to convert the output of the conllu module into feature
vectors understood by the nn module.
"""

from collections import defaultdict

import json

import h5py

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
		
		The lemmas dict provides unique IDs to the lemmas found in the dataset
		that the features are extracted from. ID 0 is used for unrecognised
		lemmas, hence the underscore. ID 1 is used for the non-standard root
		node lemma (__root__).
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
		self.lemmas['_']
		self.lemmas['__root__']
	
	
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
		for sent in dataset.gen_sentences():
			[self.lemmas[word.LEMMA] for word in sent]
	
	
	def get_vocab_sizes(self):
		"""
		Returns a {} containing: (*) the number of lemma IDs, i.e. the number
		of lemmas found during reading + 1 (for the unrecognised lemmas ID);
		(*) the number of POS tags + 1 (for the tags of the padding); (*) the
		size of the vectors returned by the featurise_morph method.
		
		These are used as parameters the POS and lemma embedding and the
		morphology input layers of the neural network.
		"""
		return {
			'lemmas': len(self.lemmas),
			'morph': 135 if self.ud_version == 2 else 104,
			'pos_tags': len(self.POS_TAGS) + 1}
	
	
	def featurise_lemma(self, lemma):
		"""
		Returns an integer uniquely identifying the given lemma. If the lemma
		has not been found during reading, returns 0.
		"""
		if lemma not in self.lemmas:
			return 0
		
		return self.lemmas[lemma]
	
	
	def featurise_pos_tag(self, pos_tag):
		"""
		Returns a positive integer uniquely identifying the POS tag. Raises a
		FeatureError if the given string is neither a valid universal POS tag
		nor 'ROOT'.
		"""
		try:
			return self.POS_TAGS.index(pos_tag) + 1
		except ValueError:
			raise FeatureError('Unknown POS tag: {}'.format(pos_tag))
	
	
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
		Returns the features for the edge (defined by the given tuple) in the
		given graph. Non-edges can also be featurised.
		
		The return value is a dict the keys of which should be the same as the
		ones listed in code.nn.EDGE_FEATURES.
		"""
		a, b = edge
		d = {}
		
		d['pos_A-1'] = 0 if a-1 < 0 else self.featurise_pos_tag(graph.node[a-1]['UPOSTAG'])
		d['pos_A'] = self.featurise_pos_tag(graph.node[a]['UPOSTAG'])
		d['pos_A+1'] = 0 if a+1 >= len(graph) else self.featurise_pos_tag(graph.node[a+1]['UPOSTAG'])
		
		d['pos_B-1'] = 0 if b-1 < 0 else self.featurise_pos_tag(graph.node[b-1]['UPOSTAG'])
		d['pos_B'] = self.featurise_pos_tag(graph.node[b]['UPOSTAG'])
		d['pos_B+1'] = 0 if b+1 >= len(graph) else self.featurise_pos_tag(graph.node[b+1]['UPOSTAG'])
		
		d['morph_A-1'] = self.featurise_morph('_' if a-1 < 0 else graph.node[a-1]['FEATS'])
		d['morph_A'] = self.featurise_morph(graph.node[a]['FEATS'])
		d['morph_A+1'] = self.featurise_morph('_' if a+1 >= len(graph) else graph.node[a+1]['FEATS'])
		
		d['morph_B-1'] = self.featurise_morph('_' if b-1 < 0 else graph.node[b-1]['FEATS'])
		d['morph_B'] = self.featurise_morph(graph.node[b]['FEATS'])
		d['morph_B+1'] = self.featurise_morph('_' if b+1 >= len(graph) else graph.node[b+1]['FEATS'])
		
		d['lemma_A'] = self.featurise_lemma(graph.node[a]['LEMMA'])
		d['lemma_B'] = self.featurise_lemma(graph.node[b]['LEMMA'])
		
		d['B-A'] = b - a
		
		return d
