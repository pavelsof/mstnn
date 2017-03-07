import enum
import itertools

from keras.layers.core import Dense
from keras.models import Sequential, load_model
from keras.utils.np_utils import to_categorical

import numpy as np



@enum.unique
class Label(enum.IntEnum):
	NO_EDGE = 0
	A_TO_B = 1
	B_TO_A = 2



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
		self.model = Sequential([
			Dense(64, input_dim=244, init='uniform', activation='relu'),
			Dense(64, init='uniform', activation='relu'),
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
		samples = []
		targets = []
		
		for graph in dataset.gen_graphs():
			edges = graph.edges()
			for node_a, node_b in itertools.combinations(graph.nodes(), 2):
				samples.append(
					extractor.featurise_edge(graph, (node_a, node_b)))
				
				if (node_a, node_b) in edges:
					targets.append(Label.A_TO_B)
				elif (node_b, node_a) in edges:
					targets.append(Label.B_TO_A)
				else:
					targets.append(Label.NO_EDGE)
		
		samples = np.array(samples)
		targets = to_categorical(np.array(targets))
		
		self.model.fit(samples, targets,
				batch_size=32, nb_epoch=epochs, shuffle=True)
	
	
	def calc_probs(self, graph, extractor):
		"""
		Calculates the probabilities of each edge.
		"""
		scores = {}
		
		samples = []
		
		for node_a, node_b in itertools.combinations(graph.nodes(), 2):
			samples.append(
				extractor.featurise_edge(graph, (node_a, node_b)))
		
		samples = np.array(samples)
		
		probs = self.model.predict_proba(samples)
		
		for index, (node_a, node_b) in enumerate(itertools.combinations(graph.nodes(), 2)):
			scores[(node_a, node_b)] = probs[index][Label.A_TO_B]
			scores[(node_b, node_a)] = probs[index][Label.B_TO_A]
		
		return scores
