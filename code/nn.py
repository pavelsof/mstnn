import collections
import enum
import itertools

from keras.layers.core import Dense, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

import numpy as np



@enum.unique
class Label(enum.IntEnum):
	NO_EDGES = 0
	
	A_TO_B_AND_C = 1
	A_TO_B = 2
	A_TO_B_TO_C = 3
	A_TO_C = 4
	A_TO_C_TO_B = 5
	
	B_TO_A_AND_C = 6
	B_TO_A = 7
	B_TO_A_TO_C = 8
	B_TO_C = 9
	B_TO_C_TO_A = 10
	
	C_TO_A_AND_B = 11
	C_TO_A = 12
	C_TO_A_TO_B = 13
	C_TO_B = 14
	C_TO_B_TO_A = 15



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
			Dense(64, input_dim=366, init='uniform', activation='relu')
		])
		
		lexicon_branch = Sequential([
			Embedding(vocab_size, 64, input_length=3),
			Flatten()
		])
		
		self.model = Sequential([
			Merge([grammar_branch, lexicon_branch], mode='concat'),
			Dense(128, init='uniform', activation='relu'),
			Dense(128, init='uniform', activation='relu'),
			Dense(len(Label), init='uniform', activation='sigmoid')
		])
		
		self.model.compile(optimizer='sgd',
				loss='categorical_crossentropy',
				metrics=['accuracy'])
	
	
	def train(self, dataset, extractor, epochs=10):
		"""
		Expects a conllu.Dataset instance to train on and a features.Extractor
		instance to extract the feature vectors with.
		"""
		samples_grammar = []
		samples_lexicon = []
		targets = []
		
		for graph in dataset.gen_graphs():
			edges = graph.edges()
			for a, b, c in itertools.combinations(graph.nodes(), 3):
				samples_grammar.append(np.concatenate([
					extractor.featurise_edge(graph, (a, b)),
					extractor.featurise_edge(graph, (b, c)),
					extractor.featurise_edge(graph, (a, c))
				]))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[a]['LEMMA']),
					extractor.featurise_lemma(graph.node[b]['LEMMA']),
					extractor.featurise_lemma(graph.node[c]['LEMMA'])
				])
				
				tri = list(filter(lambda x: x[0] in [a, b, c] and x[1] in [a, b, c], edges))
				label = None
				
				if not tri:
					label = Label.NO_EDGES
				elif len(tri) == 1:
					if (a, b) in tri: label = Label.A_TO_B
					elif (a, c) in tri: label = Label.A_TO_C
					elif (b, a) in tri: label = Label.B_TO_A
					elif (b, c) in tri: label = Label.B_TO_C
					elif (c, a) in tri: label = Label.C_TO_A
					elif (c, b) in tri: label = Label.C_TO_B
				elif len(tri) == 2:
					if (a, b) in tri and (a, c) in tri: label = Label.A_TO_B_AND_C
					elif (a, b) in tri and (b, c) in tri: label = Label.A_TO_B_TO_C
					elif (a, c) in tri and (c, b) in tri: label = Label.A_TO_C_TO_B
					elif (b, a) in tri and (b, c) in tri: label = Label.B_TO_A_AND_C
					elif (b, a) in tri and (a, c) in tri: label = Label.B_TO_A_TO_C
					elif (b, c) in tri and (c, a) in tri: label = Label.B_TO_C_TO_A
					elif (c, a) in tri and (c, b) in tri: label = Label.C_TO_A_AND_B
					elif (c, a) in tri and (a, b) in tri: label = Label.C_TO_A_TO_B
					elif (c, b) in tri and (b, a) in tri: label = Label.C_TO_B_TO_A
				
				if label is None:
					raise ValueError('Something went wrong')
				
				targets.append(label)
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		targets = to_categorical(np.array(targets))
		
		self.model.fit([samples_grammar, samples_lexicon], targets,
				batch_size=32, nb_epoch=epochs, shuffle=True)
	
	
	def calc_probs(self, graph, extractor):
		"""
		Calculates the probabilities of each edge.
		"""
		scores = collections.defaultdict(lambda: 0)
		
		samples_grammar = []
		samples_lexicon = []
		
		for a, b, c in itertools.combinations(graph.nodes(), 3):
			samples_grammar.append(np.concatenate([
				extractor.featurise_edge(graph, (a, b)),
				extractor.featurise_edge(graph, (b, c)),
				extractor.featurise_edge(graph, (a, c))
			]))
			samples_lexicon.append([
				extractor.featurise_lemma(graph.node[a]['LEMMA']),
				extractor.featurise_lemma(graph.node[b]['LEMMA']),
				extractor.featurise_lemma(graph.node[c]['LEMMA'])
			])
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		
		probs = self.model.predict_proba([samples_grammar, samples_lexicon])
		
		for index, (a, b, c) in enumerate(itertools.combinations(graph.nodes(), 3)):
			p = probs[index]
			scores[(a, b)] = p[Label.A_TO_B_AND_C] + p[Label.A_TO_B] + p[Label.A_TO_B_TO_C] + p[Label.C_TO_A_TO_B]
			scores[(b, a)] = p[Label.B_TO_A_AND_C] + p[Label.B_TO_A] + p[Label.B_TO_A_TO_C] + p[Label.C_TO_B_TO_A]
			scores[(a, c)] = p[Label.A_TO_B_AND_C] + p[Label.A_TO_C] + p[Label.A_TO_C_TO_B] + p[Label.B_TO_A_TO_C]
			scores[(c, a)] = p[Label.C_TO_A_AND_B] + p[Label.C_TO_A] + p[Label.C_TO_A_TO_B] + p[Label.B_TO_C_TO_A]
			scores[(b, c)] = p[Label.B_TO_A_AND_C] + p[Label.B_TO_C] + p[Label.B_TO_C_TO_A] + p[Label.A_TO_B_TO_C]
			scores[(c, b)] = p[Label.C_TO_A_AND_B] + p[Label.C_TO_B] + p[Label.C_TO_B_TO_A] + p[Label.A_TO_C_TO_B]
		
		return scores
