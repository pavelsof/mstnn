"""
This module knows how to convert the output of the conllu module into feature
vectors understood by the nn module.
"""

from collections import defaultdict

import itertools
import json

import h5py

import numpy as np

from code.nn import EDGE_FEATURES
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
	
	Then makes the second pass over the data, collecting the samples (and
	possibly the targets) to be fed into the neural network.
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
	
	
	def extract(self, dataset, include_targets=False):
		"""
		Extracts and returns the samples, and possibly also the targets, from
		the given conllu.Dataset instance.
		
		Assumes that the latter is already read, i.e. this Extractor instance
		has already populated its self.lemmas dict.
		
		The samples comprise a dict where the keys are nn.EDGE_FEATURES and the
		values are numpy arrays. The targets are a numpy array of 0s and 1s.
		"""
		samples = {key: [] for key in EDGE_FEATURES}
		targets = []
		
		for graph in dataset.gen_graphs():
			nodes = graph.nodes()
			edges = graph.edges()
			
			pos_tags = {i: self.featurise_pos_tag(graph.node[i]['UPOSTAG']) for i in nodes}
			pos_tags[-1] = 0
			pos_tags[len(nodes)] = 0
			
			morph = {i: self.featurise_morph(graph.node[i]['FEATS']) for i in nodes}
			morph[-1] = self.featurise_morph('_')
			morph[len(nodes)] = self.featurise_morph('_')
			
			lemmas = {i: self.featurise_lemma(graph.node[i]['LEMMA']) for i in nodes}
			
			for a, b in itertools.permutations(nodes, 2):
				samples['pos A-1'].append(pos_tags[a-1])
				samples['pos A'].append(pos_tags[a])
				samples['pos A+1'].append(pos_tags[a+1])
				
				samples['pos B-1'].append(pos_tags[b-1])
				samples['pos B'].append(pos_tags[b])
				samples['pos B+1'].append(pos_tags[b+1])
				
				samples['morph A-1'].append(morph[a-1])
				samples['morph A'].append(morph[a])
				samples['morph A+1'].append(morph[a+1])
				
				samples['morph B-1'].append(morph[b-1])
				samples['morph B'].append(morph[b])
				samples['morph B+1'].append(morph[b+1])
				
				samples['lemma A'].append(lemmas[a])
				samples['lemma B'].append(lemmas[b])
				
				samples['B-A'].append(b-a)
				
				if include_targets:
					targets.append((a, b) in edges)
		
		samples_ = {}
		for key, value in samples.items():
			if key.startswith('lemma'):
				samples_[key] = np.array(value, dtype='uint16')
			else:
				samples_[key] = np.array(value, dtype='uint8')
		del samples
		
		targets = np.array(targets, dtype='uint8')
		
		if include_targets:
			return samples_, targets
		else:
			return samples_
