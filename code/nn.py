from keras.callbacks import LambdaCallback
from keras.layers import Dense, Embedding, Flatten, Input, merge
from keras.models import load_model, Model



"""
Tuple listing the names of the features that are fed into the network for each
edge of each sentence graph.

Used as the keys of the samples dict returned by the Extractor.extract method.
"""
EDGE_FEATURES = tuple([
	'pos A',
	'pos B',
	'morph A-1', 'morph A', 'morph A+1',
	'morph B-1', 'morph B', 'morph B+1',
	'lemma A', 'lemma B', 'B-A'])



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
		pos_a = Input(shape=(5,), dtype='uint8')
		pos_b = Input(shape=(5,), dtype='uint8')
		
		pos_embed = Embedding(vocab_sizes['pos_tags'], 32, input_length=5)
		pos = merge([
			Flatten()(pos_embed(pos_a)),
			Flatten()(pos_embed(pos_b))], mode='concat')
		
		morph_a = Input(shape=(vocab_sizes['morph'],))
		morph_a_prev = Input(shape=(vocab_sizes['morph'],))
		morph_a_next = Input(shape=(vocab_sizes['morph'],))
		
		morph_b = Input(shape=(vocab_sizes['morph'],))
		morph_b_prev = Input(shape=(vocab_sizes['morph'],))
		morph_b_next = Input(shape=(vocab_sizes['morph'],))
		
		morph = merge([
			morph_a_prev, morph_a, morph_a_next,
			morph_b_prev, morph_b, morph_b_next], mode='concat')
		morph = Dense(64, init='uniform', activation='relu')(morph)
		
		lemma_a = Input(shape=(1,), dtype='uint16')
		lemma_b = Input(shape=(1,), dtype='uint16')
		lemma_embed = Embedding(vocab_sizes['lemmas'], 64, input_length=1)
		lemmas = merge([
			Flatten()(lemma_embed(lemma_a)),
			Flatten()(lemma_embed(lemma_b))], mode='concat')
		
		rel_pos_raw = Input(shape=(1,))
		rel_pos = Dense(32, init='uniform', activation='relu')(rel_pos_raw)
		
		x = merge([pos, morph, lemmas, rel_pos], mode='concat')
		x = Dense(128, init='he_uniform', activation='relu')(x)
		x = Dense(128, init='he_uniform', activation='relu')(x)
		output = Dense(1, init='uniform', activation='sigmoid')(x)
		
		self.model = Model(input=[
			pos_a, pos_b,
			morph_a_prev, morph_a, morph_a_next,
			morph_b_prev, morph_b, morph_b_next,
			lemma_a, lemma_b, rel_pos_raw], output=output)
		
		self.model.compile(optimizer='sgd',
				loss='binary_crossentropy',
				metrics=['accuracy'])
	
	
	def train(self, samples, targets, epochs=10, on_epoch_end=None):
		"""
		Trains the network. The first arg should be a dict where the keys are
		EDGE_FEATURES and the values numpy arrays. The second one should be a
		single numpy array of 0s and 1s.
		"""
		if on_epoch_end:
			callbacks = [LambdaCallback(on_epoch_end=on_epoch_end)]
		else:
			callbacks = []
		
		self.model.fit([samples[key] for key in EDGE_FEATURES], targets,
			batch_size=32, shuffle=True, nb_epoch=epochs, callbacks=callbacks)
	
	
	def calc_probs(self, samples):
		"""
		Calculates the probabilities of each edge. The arg should be a dict
		where the keys are EDGE_FEATURES and the values are the respective
		numpy arrays.
		"""
		return self.model.predict([samples[key] for key in EDGE_FEATURES],
			batch_size=32, verbose=1)
