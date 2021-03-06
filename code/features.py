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
	
	def __init__(self, ignore_forms=False, ignore_lemmas=False, ignore_morph=False):
		"""
		Constructor. The boolean flags could be used to disable the extraction
		of certain features.
		
		The pos_tags tuple lists all the possible POS tags that a node could
		belong to.
		
		The morph dict comprises the morphological features, both keys and
		values, found in the dataset during the reading phase.
		
		The lemmas dict provides unique IDs to the lemmas found in the dataset
		during the reading phase. ID 0 is used for unrecognised lemmas, hence
		the underscore. ID 1 is used for the non-standard root node lemma
		(\xa0). The same applies for the forms dict.
		"""
		self.ignore_forms = ignore_forms
		self.ignore_lemmas = ignore_lemmas
		self.ignore_morph = ignore_morph
		
		self.forms = defaultdict(lambda: len(self.forms))
		self.forms['_']  # unknown
		self.forms['\xa0']  # root
		
		self.lemmas = defaultdict(lambda: len(self.lemmas))
		self.lemmas['_']  # unknown
		self.lemmas['\xa0']  # root
		
		self.pos_tags = ('ROOT',)  # tuple of possible pos tags
		
		self.morph = {}  # key: tuple of possible values
	
	
	@classmethod
	def create_from_model_file(cls, model_fp):
		"""
		Returns a new Extractor instance with self.pos_tags, self.morph and
		self.lemmas loaded from the specified model file. The latter is
		expected to be a hdf5 file written or appended to by the method below.
		
		Raises an OSError if the file does not exist or cannot be read.
		"""
		f = h5py.File(model_fp, 'r')
		
		forms = json.loads(f['extractor'].attrs['forms'])
		lemmas = json.loads(f['extractor'].attrs['lemmas'])
		pos_tags = json.loads(f['extractor'].attrs['pos_tags'])
		morph = json.loads(f['extractor'].attrs['morph'])
		
		ignore_forms = json.loads(f['extractor'].attrs['ignore_forms'])
		ignore_lemmas = json.loads(f['extractor'].attrs['ignore_lemmas'])
		ignore_morph = json.loads(f['extractor'].attrs['ignore_morph'])
		
		f.close()
		
		extractor = Extractor(ignore_forms, ignore_lemmas, ignore_morph)
		
		for key, value in forms.items():
			extractor.forms[key] = value
		
		for key, value in lemmas.items():
			extractor.lemmas[key] = value
		
		extractor.pos_tags = tuple(pos_tags)
		extractor.morph = {key: tuple(value) for key, value in morph.items()}
		
		return extractor
	
	
	def write_to_model_file(self, model_fp):
		"""
		Appends to the specified hdf5 file, storing the instance's properties.
		Thus, an identical Extractor can be later restored using the above
		class method.
		
		Raises an OSError if the file cannot be written.
		"""
		f = h5py.File(model_fp, 'a')
		
		group = f.create_group('extractor')
		
		group.attrs['forms'] = json.dumps(dict(self.forms), ensure_ascii=False)
		group.attrs['lemmas'] = json.dumps(dict(self.lemmas), ensure_ascii=False)
		group.attrs['pos_tags'] = json.dumps(list(self.pos_tags), ensure_ascii=False)
		group.attrs['morph'] = json.dumps(dict(self.morph), ensure_ascii=False)
		
		group.attrs['ignore_forms'] = json.dumps(self.ignore_forms, ensure_ascii=False)
		group.attrs['ignore_lemmas'] = json.dumps(self.ignore_lemmas, ensure_ascii=False)
		group.attrs['ignore_morph'] = json.dumps(self.ignore_morph, ensure_ascii=False)
		
		f.flush()
		f.close()
	
	
	def read(self, dataset):
		"""
		Reads the data provided by the given conllu.Dataset instance and
		populates self.forms, self.lemmas, self.pos_tag, and self.morph.
		"""
		form_counts = defaultdict(lambda: 0)
		lemma_counts = defaultdict(lambda: 0)
		pos_tags = set(self.pos_tags)
		morph = defaultdict(set)
		
		for sent in dataset.gen_sentences():
			for word in sent:
				form_counts[word.FORM] += 1
				lemma_counts[word.LEMMA] += 1
				
				pos_tags.add(word.UPOSTAG)
				
				for key, values in word.FEATS.items():
					for value in values:
						morph[key].add(value)
		
		if not self.ignore_forms:
			for form, count in form_counts.items():
				if count > 1:
					self.forms[form]
		
		if not self.ignore_lemmas:
			for lemma, count in lemma_counts.items():
				if count > 1:
					self.lemmas[lemma]
		
		self.pos_tags = tuple(sorted(pos_tags))
		
		if not self.ignore_morph:
			self.morph = {key: tuple(sorted(value)) for key, value in morph.items()}
			
			if not self.morph:
				raise ValueError((
					'Could not find morphological features in the dataset; '
					'please use the --ignore-morph flag'))
	
	
	def get_vocab_sizes(self):
		"""
		Returns a {} containing: the number of form IDs, the number of lemma
		IDs, the number of POS tags, and the size of the vectors returned by
		the featurise_morph method.
		
		These are used as parameters for the form, lemma, and POS tag embedding
		and the morphology input layers of the neural network.
		
		This method assumes that the reading phase is already completed.
		"""
		morph_size = sum([len(value) for value in self.morph.values()])
		
		return {
			'forms': 0 if self.ignore_forms else len(self.forms),
			'lemmas': 0 if self.ignore_lemmas else len(self.lemmas),
			'morph': morph_size,
			'pos_tags': len(self.pos_tags) + 1}
	
	
	def featurise_form(self, form):
		"""
		Returns a positive integer uniquely identifying the given form. If the
		form has not been found during the reading phase, returns 0.
		"""
		if form not in self.forms:
			return 0
		
		return self.forms[form]
	
	
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
		the given Dataset instance.
		
		Assumes that the latter is already read, i.e. this Extractor instance
		has already populated its pos_tags, lemmas, and morph properties.
		
		The samples comprise a dict where the keys are nn.EDGE_FEATURES (or a
		subset of those) and the values are numpy arrays. The targets are a
		numpy array of 0s and 1s.
		"""
		keys = list(EDGE_FEATURES)
		if self.ignore_forms:
			keys.remove('forms')
		if self.ignore_lemmas:
			keys.remove('lemmas')
		if self.ignore_morph:
			keys = itertools.filterfalse(lambda x: x.startswith('morph'), keys)
		
		samples = {key: [] for key in keys}
		targets = []
		
		for graph in dataset.gen_graphs():
			nodes = graph.nodes()
			edges = graph.edges()
			
			if not self.ignore_forms:
				forms = {i: self.featurise_form(graph.node[i]['FORM']) for i in nodes}
			
			if not self.ignore_lemmas:
				lemmas = {i: self.featurise_lemma(graph.node[i]['LEMMA']) for i in nodes}
			
			pos_tags = {i: self.featurise_pos_tag(graph.node[i]['UPOSTAG']) for i in nodes}
			pos_tags[-2] = 0
			pos_tags[-1] = 0
			pos_tags[len(nodes)] = 0
			pos_tags[len(nodes)+1] = 0
			
			if not self.ignore_morph:
				morph = {i: self.featurise_morph(graph.node[i]['FEATS']) for i in nodes}
				morph[-1] = self.featurise_morph('_')
				morph[len(nodes)] = self.featurise_morph('_')
			
			for a, b in itertools.permutations(nodes, 2):
				samples['B-A'].append(b-a)
				
				if not self.ignore_forms:
					samples['forms'].append([forms[a], forms[b]])
				
				if not self.ignore_lemmas:
					samples['lemmas'].append([lemmas[a], lemmas[b]])
				
				samples['pos'].append([
					pos_tags[a-2], pos_tags[a-1], pos_tags[a], pos_tags[a+1], pos_tags[a+2],
					pos_tags[b-2], pos_tags[b-1], pos_tags[b], pos_tags[b+1], pos_tags[b+2]])
				
				if not self.ignore_morph:
					samples['morph A-1'].append(morph[a-1])
					samples['morph A'].append(morph[a])
					samples['morph A+1'].append(morph[a+1])
					
					samples['morph B-1'].append(morph[b-1])
					samples['morph B'].append(morph[b])
					samples['morph B+1'].append(morph[b+1])
				
				if include_targets:
					targets.append((a, b) in edges)
		
		samples_ = {}
		for key, value in samples.items():
			if key in ['forms', 'lemmas']:
				samples_[key] = np.array(value, dtype='uint16')
			else:
				samples_[key] = np.array(value, dtype='uint8')
		del samples
		
		targets = np.array(targets, dtype='uint8')
		
		if include_targets:
			return samples_, targets
		else:
			return samples_
