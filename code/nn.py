import itertools

from keras.layers.core import Dense, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model

import networkx as nx
import numpy as np



class NeuralNetwork:
	
	def __init__(self, model=None, vocab_size=None):
		"""
		Constructor. The first keyword argument should be a Keras model. If not
		specified, a new model is created and compiled. In any case, the Keras
		model should be compiled and ready to use right after the init.
		
		The remaining keyword arguments should be set if the first is not (they
		are ignored otherwise).
		"""
		if model is None:
			assert isinstance(vocab_size, int)
			self._init_model(vocab_size)
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
	
	
	def _init_model(self, vocab_size):
		"""
		Inits and compiles the Keras model. This method is only called when
		training; for testing, the Keras model is loaded.
		
		The network consists of two input branches, one handling the edges' POS
		tags and morphological features, and the other handling the lemma
		embeddings. These two inputs are concatenated and then go through a
		standard single-layer perceptron.
		"""
		grammar_branch = Sequential([
			Dense(64, input_dim=244, init='uniform', activation='tanh')
		])
		
		lexicon_branch = Sequential([
			Embedding(vocab_size, 64, input_length=2),
			Flatten()
		])
		
		self.model = Sequential([
			Merge([grammar_branch, lexicon_branch], mode='concat'),
			Dense(128, init='uniform', activation='tanh'),
			Dense(1, init='uniform', activation='sigmoid')
		])
		
		self.model.compile(optimizer='sgd',
				loss='binary_crossentropy',
				metrics=['accuracy'])
	
	
	def train(self, dataset, extractor, epochs=10):
		"""
		Expects a conllu.Dataset instance to train on and a features.Extractor
		instance to extract the feature vectors with.
		
		Currently each sample consists of concatenating the POS tag and
		morphology feature vectors of the parent and the child of each edge.
		"""
		samples_grammar = []
		samples_lexicon = []
		targets = []
		
		for graph in dataset.gen_graphs():
			for edge in graph.edges():
				samples_grammar.append(extractor.featurise_edge(graph, edge))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[edge[0]]['LEMMA']),
					extractor.featurise_lemma(graph.node[edge[1]]['LEMMA'])
				])
				targets.append(1)
			
			for edge in nx.non_edges(graph):
				samples_grammar.append(extractor.featurise_edge(graph, edge))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[edge[0]]['LEMMA']),
					extractor.featurise_lemma(graph.node[edge[1]]['LEMMA'])
				])
				targets.append(0)
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		targets = np.array(targets)
		
		self.model.fit([samples_grammar, samples_lexicon], targets,
				batch_size=32, nb_epoch=epochs, shuffle=True)
	
	
	def calc_probs(self, graph, extractor):
		"""
		Calculates the probabilities of each edge.
		"""
		samples_grammar = []
		samples_lexicon = []
		
		for edge in itertools.permutations(graph.nodes(), 2):
			samples_grammar.append(extractor.featurise_edge(graph, edge))
			samples_lexicon.append([
				extractor.featurise_lemma(graph.node[edge[0]]['LEMMA']),
				extractor.featurise_lemma(graph.node[edge[1]]['LEMMA'])
			])
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		
		probs = self.model.predict_proba([samples_grammar, samples_lexicon])
		
		return probs
