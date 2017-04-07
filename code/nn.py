import itertools

from keras.layers import Dense, Embedding, Flatten, Input, merge
from keras.models import load_model, Model

import numpy as np



"""
Tuple listing the names of the features that are fed into the network for each
edge of each sentence graph.

Used in the train and calc_probs methods of the NeuralNetwork class below.
"""
EDGE_FEATURES = tuple([
	'pos_A-1', 'pos_A', 'pos_A+1',
	'pos_B-1', 'pos_B', 'pos_B+1',
	'morph_A-1', 'morph_A', 'morph_A+1',
	'morph_B-1', 'morph_B', 'morph_B+1',
	'lemma_A', 'lemma_B', 'B-A'])



class NeuralNetwork:
	
	def __init__(self, model=None, vocab_sizes=None):
		"""
		Constructor. The first keyword argument should be a Keras model. If not
		specified, a new model is created and compiled. In any case, the Keras
		model should be compiled and ready to use right after the init.
		
		The remaining keyword arguments should be set if the first is not (they
		are ignored otherwise).
		"""
		if model is None:
			assert isinstance(vocab_sizes['lemmas'], int)
			assert isinstance(vocab_sizes['morph'], int)
			assert isinstance(vocab_sizes['pos_tags'], int)
			self._init_model(vocab_sizes)
		else:
			self.model = model
	
	
	@classmethod
	def create_from_model_file(cls, model_fp):
		"""
		Returns a new NeuralNetwork instance with its Keras model loaded from
		the specified Keras model file.
		
		Raises an OSError if the file does not exist or cannot be read.
		"""
		keras_model = load_model(model_fp)
		return cls(keras_model)
	
	
	def write_to_model_file(self, model_fp):
		"""
		Writes a hdf5 file containing the Keras model's architecture, weights,
		training configuration, and state of the optimiser. Thus, an identical
		NeuralNetwork can be later restored using the above class method.
		
		If there already is a file at the specified path, it gets overwritten.
		
		Raises an OSError if the file cannot be written.
		"""
		self.model.save(model_fp, overwrite=True)
	
	
	def _init_model(self, vocab_sizes):
		"""
		Inits and compiles the Keras model. This method is only called when
		training; for testing, the Keras model is loaded.
		
		The network takes as input the POS tags and the morphological features
		of two nodes and their immediate neighbours (the context), as well as
		the nodes' lemmas and their relative position to each other, and tries
		to predict the probability of an edge between the two.
		"""
		pos_a = Input(shape=(1,), dtype='int32', name='pos_A')
		pos_a_prev = Input(shape=(1,), dtype='int32', name='pos_A-1')
		pos_a_next = Input(shape=(1,), dtype='int32', name='pos_A+1')
		
		pos_b = Input(shape=(1,), dtype='int32', name='pos_B')
		pos_b_prev = Input(shape=(1,), dtype='int32', name='pos_B-1')
		pos_b_next = Input(shape=(1,), dtype='int32', name='pos_B+1')
		
		pos_embed = Embedding(vocab_sizes['pos_tags'], 32,
						input_length=1, name='pos_embedding')
		pos = merge([
			Flatten()(pos_embed(pos_a_prev)),
			Flatten()(pos_embed(pos_a)),
			Flatten()(pos_embed(pos_a_next)),
			Flatten()(pos_embed(pos_b_prev)),
			Flatten()(pos_embed(pos_b)),
			Flatten()(pos_embed(pos_b_next))], mode='concat')
		
		morph_a = Input(shape=(vocab_sizes['morph'],), name='morph_A')
		morph_a_prev = Input(shape=(vocab_sizes['morph'],), name='morph_A-1')
		morph_a_next = Input(shape=(vocab_sizes['morph'],), name='morph_A+1')
		
		morph_b = Input(shape=(vocab_sizes['morph'],), name='morph_B')
		morph_b_prev = Input(shape=(vocab_sizes['morph'],), name='morph_B-1')
		morph_b_next = Input(shape=(vocab_sizes['morph'],), name='morph_B+1')
		
		morph = merge([
			morph_a_prev, morph_a, morph_a_next,
			morph_b_prev, morph_b, morph_b_next], mode='concat')
		morph = Dense(64, init='uniform', activation='relu')(morph)
		
		lemma_a = Input(shape=(1,), dtype='int32', name='lemma_A')
		lemma_b = Input(shape=(1,), dtype='int32', name='lemma_B')
		lemma_embed = Embedding(vocab_sizes['lemmas'], 256,
						input_length=1, name='lemma_embedding')
		lemmas = merge([
			Flatten()(lemma_embed(lemma_a)),
			Flatten()(lemma_embed(lemma_b))], mode='concat')
		
		rel_pos_raw = Input(shape=(1,), name='B-A')
		rel_pos = Dense(32, init='uniform', activation='relu')(rel_pos_raw)
		
		x = merge([pos, morph, lemmas, rel_pos], mode='concat')
		x = Dense(128, init='he_uniform', activation='relu')(x)
		x = Dense(128, init='he_uniform', activation='relu')(x)
		output = Dense(1, init='uniform', activation='sigmoid')(x)
		
		self.model = Model(input=[
			pos_a_prev, pos_a, pos_a_next,
			pos_b_prev, pos_b, pos_b_next,
			morph_a_prev, morph_a, morph_a_next,
			morph_b_prev, morph_b, morph_b_next,
			lemma_a, lemma_b, rel_pos_raw], output=output)
		
		self.model.compile(optimizer='sgd',
				loss='binary_crossentropy',
				metrics=['accuracy'])
	
	
	def train(self, dataset, extractor, epochs=10):
		"""
		Expects a conllu.Dataset instance to train on and a features.Extractor
		instance to extract the feature vectors with.
		"""
		feats = {key: [] for key in EDGE_FEATURES}
		targets = []
		
		for graph in dataset.gen_graphs():
			edges = graph.edges()
			
			for a, b in itertools.permutations(graph.nodes(), 2):
				d = extractor.featurise_edge(graph, (a, b))
				for key, value in d.items():
					feats[key].append(value)
				
				targets.append((a, b) in edges)
		
		self.model.fit([np.array(feats[key]) for key in EDGE_FEATURES],
			np.array(targets), batch_size=32, nb_epoch=epochs, shuffle=True)
	
	
	def calc_probs(self, graph, extractor):
		"""
		Calculates the probabilities of each edge.
		"""
		feats = {key: [] for key in EDGE_FEATURES}
		
		for a, b in itertools.permutations(graph.nodes(), 2):
			d = extractor.featurise_edge(graph, (a, b))
			for key, value in d.items():
				feats[key].append(value)
		
		feats = [np.array(feats[key]) for key in EDGE_FEATURES]
		probs = self.model.predict(feats, verbose=1)
		
		scores = {}
		
		for index, (a, b) in enumerate(itertools.permutations(graph.nodes(), 2)):
			scores[(a, b)] = probs[index][0]
		
		return scores
