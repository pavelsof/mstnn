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



class Extractor:
	"""
	Makes the first pass over the data, collecting what is necessary in order
	to determine the dimensions of the neural network. Provides the featurise_*
	methods (which would otherwise need to expect the UD version as an arg).
	
	Then makes the second pass over the data, collecting the samples (and
	possibly the targets) to be fed into the neural network.
	"""
	
	def __init__(self):
		"""
		Constructor. The pos_tags tuple lists all the possible POS tags that a
		node could belong to.
		
		The morph dict comprises the morphological features, both keys and
		values, found in the dataset during the reading phase.
		
		The lemmas dict provides unique IDs to the lemmas found in the dataset
		during the reading phase. ID 0 is used for unrecognised lemmas, hence
		the underscore. ID 1 is used for the non-standard root node lemma
		(__root__).
		"""
		self.pos_tags = ('ROOT',)  # tuple of possible pos tags
		
		self.morph = {}  # key: tuple of possible values
		
		self.lemmas = defaultdict(lambda: len(self.lemmas))
		self.lemmas['_']
		self.lemmas['__root__']
	
	
	@classmethod
	def create_from_model_file(cls, model_fp):
		"""
		Returns a new Extractor instance with self.pos_tags, self.morph and
		self.lemmas loaded from the specified model file. The latter is
		expected to be a hdf5 file written or appended to by the method below.
		
		Raises an OSError if the file does not exist or cannot be read.
		"""
		f = h5py.File(model_fp, 'r')
		
		pos_tags = json.loads(f['extractor'].attrs['pos_tags'])
		morph = json.loads(f['extractor'].attrs['morph'])
		lemmas = json.loads(f['extractor'].attrs['lemmas'])
		
		f.close()
		
		extractor = Extractor()
		
		for key, value in lemmas.items():
			extractor.lemmas[key] = value
		
		extractor.morph = {key: tuple(value) for key, value in morph.items()}
		extractor.pos_tags = tuple(pos_tags)
		
		return extractor
	
	
	def write_to_model_file(self, model_fp):
		"""
		Appends to the specified hdf5 file, storing self.pos_tag, self.morph
		and self.lemmas.  Thus, an identical Extractor can be later restored
		using the above class method.
		
		Raises an OSError if the file cannot be written.
		"""
		f = h5py.File(model_fp, 'a')
		
		group = f.create_group('extractor')
		group.attrs['lemmas'] = json.dumps(dict(self.lemmas), ensure_ascii=False)
		group.attrs['morph'] = json.dumps(dict(self.morph), ensure_ascii=False)
		group.attrs['pos_tags'] = json.dumps(list(self.pos_tags), ensure_ascii=False)
		
		f.flush()
		f.close()
	
	
	def read(self, dataset):
		"""
		Reads the data provided by the given conllu.Dataset instance and
		populates self.pos_tag, self.morph, and self.lemmas.
		"""
		pos_tags = set(self.pos_tags)
		morph = defaultdict(set)
		
		for sent in dataset.gen_sentences():
			for word in sent:
				pos_tags.add(word.UPOSTAG)
				self.lemmas[word.LEMMA]
				for key, values in word.FEATS.items():
					for value in values:
						morph[key].add(value)
		
		self.pos_tags = tuple(sorted(pos_tags))
		self.morph = {key: tuple(sorted(value)) for key, value in morph.items()}
	
	
	def get_vocab_sizes(self):
		"""
		Returns a {} containing: (*) the number of lemma IDs, i.e. the number
		of lemmas found during reading + 1 (for the unrecognised lemmas ID);
		(*) the number of POS tags + 1 (for the tags of the padding); (*) the
		size of the vectors returned by the featurise_morph method.
		
		These are used as parameters the POS and lemma embedding and the
		morphology input layers of the neural network.
		
		This method assumes that the reading phase is already completed.
		"""
		morph_size = sum([len(value) for value in self.morph.values()])
		
		return {
			'lemmas': len(self.lemmas),
			'morph': morph_size,
			'pos_tags': len(self.pos_tags) + 1}
	
	
	def featurise_lemma(self, lemma):
		"""
		Returns a positive integer uniquely identifying the given lemma. If the
		lemma has not been found during the reading phase, returns 0.
		"""
		if lemma not in self.lemmas:
			return 0
		
		return self.lemmas[lemma]
	
	
	def featurise_pos_tag(self, pos_tag):
		"""
		Returns a positive integer uniquely identifying the given POS tag. If
		the POS tag has not been found during the reading phase, returns 0.
		"""
		try:
			return self.pos_tags.index(pos_tag) + 1
		except ValueError:
			return 0
	
	
	def featurise_morph(self, morph):
		"""
		Returns the feature vector corresponding to the given morphological
		features dict.
		
		The vector is a numpy array of 0s and 1s where each element represents
		a morphological feature. E.g. the output for "Animacy=Anim" should be a
		vector with its second element 1 and all the other elements zeroes.
		"""
		vector = []
		
		for feature, poss_values in sorted(self.morph.items()):
			small_vec = [0] * len(poss_values)
			
			if feature in morph:
				for index, value in enumerate(poss_values):
					if value in morph[feature]:
						small_vec[index] = 1
			
			vector += small_vec
		
		return np.array(vector, dtype='uint8')
	
	
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
			pos_tags[-2] = 0
			pos_tags[-1] = 0
			pos_tags[len(nodes)] = 0
			pos_tags[len(nodes)+1] = 0
			
			morph = {i: self.featurise_morph(graph.node[i]['FEATS']) for i in nodes}
			morph[-1] = self.featurise_morph('_')
			morph[len(nodes)] = self.featurise_morph('_')
			
			lemmas = {i: self.featurise_lemma(graph.node[i]['LEMMA']) for i in nodes}
			
			for a, b in itertools.permutations(nodes, 2):
				samples['pos A-2'].append(pos_tags[a-2])
				samples['pos A-1'].append(pos_tags[a-1])
				samples['pos A'].append(pos_tags[a])
				samples['pos A+1'].append(pos_tags[a+1])
				samples['pos A+2'].append(pos_tags[a+2])
				
				samples['pos B-2'].append(pos_tags[b-2])
				samples['pos B-1'].append(pos_tags[b-1])
				samples['pos B'].append(pos_tags[b])
				samples['pos B+1'].append(pos_tags[b+1])
				samples['pos B+2'].append(pos_tags[b+2])
				
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
