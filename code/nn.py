import enum
import itertools

from keras.layers import Dense, Embedding, Flatten, Input, merge
from keras.models import load_model, Model
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
		
		The network consists of three input branches, one handling the edges'
		POS tags and morphological features, the second handling the lemma
		embeddings, and the third handling the relative positions of the input
		nodes. These branches then are concatenated and go through a standard
		two-layer perceptron.
		"""
		grammar_input = Input(shape=(244,), name='grammar_input')
		grammar = Dense(64, init='uniform', activation='relu')(grammar_input)
		
		lemmas_input = Input(shape=(2,), name='lemmas_input')
		lemmas = Embedding(vocab_size, 64, input_length=2)(lemmas_input)
		lemmas = Flatten()(lemmas)
		
		rel_pos_input = Input(shape=(1,), name='rel_pos_input')
		rel_pos = Dense(1, init='uniform', activation='relu')(rel_pos_input)
		
		x = merge([grammar, lemmas, rel_pos], mode='concat')
		x = Dense(128, init='uniform', activation='relu')(x)
		x = Dense(128, init='uniform', activation='relu')(x)
		output = Dense(len(Label), init='uniform', activation='sigmoid')(x)
		
		self.model = Model(input=[
			grammar_input, lemmas_input, rel_pos_input], output=output)
		
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
		samples_rel_pos = []
		
		targets = []
		
		for graph in dataset.gen_graphs():
			edges = graph.edges()
			for a, b in itertools.combinations(graph.nodes(), 2):
				samples_grammar.append(
					extractor.featurise_edge(graph, (a, b)))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[a]['LEMMA']),
					extractor.featurise_lemma(graph.node[b]['LEMMA'])
				])
				samples_rel_pos.append(b - a)
				
				if (a, b) in edges:
					targets.append(Label.A_TO_B)
				elif (b, a) in edges:
					targets.append(Label.B_TO_A)
				else:
					targets.append(Label.NO_EDGE)
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		samples_rel_pos = np.array(samples_rel_pos)
		targets = to_categorical(np.array(targets))
		
		self.model.fit([samples_grammar, samples_lexicon, samples_rel_pos],
				targets, batch_size=32, nb_epoch=epochs, shuffle=True)
	
	
	def calc_probs(self, graph, extractor):
		"""
		Calculates the probabilities of each edge.
		"""
		scores = {}
		
		samples_grammar = []
		samples_lexicon = []
		samples_rel_pos = []
		
		for a, b in itertools.combinations(graph.nodes(), 2):
			samples_grammar.append(
				extractor.featurise_edge(graph, (a, b)))
			samples_lexicon.append([
				extractor.featurise_lemma(graph.node[a]['LEMMA']),
				extractor.featurise_lemma(graph.node[b]['LEMMA'])
			])
			samples_rel_pos.append(b - a)
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		samples_rel_pos = np.array(samples_rel_pos)
		
		probs = self.model.predict([
			samples_grammar, samples_lexicon, samples_rel_pos], verbose=1)
		
		for index, (a, b) in enumerate(itertools.combinations(graph.nodes(), 2)):
			scores[(a, b)] = probs[index][Label.A_TO_B]
			scores[(b, a)] = probs[index][Label.B_TO_A]
		
		return scores
