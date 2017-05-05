from keras.callbacks import LambdaCallback
from keras.layers import Dense, Dropout, Embedding, Flatten, Input, merge
from keras.models import load_model, Model



"""
Tuple listing the names of the features that are fed into the network for each
edge of each sentence graph.

Used as the keys of the samples dict returned by the Extractor.extract method.
"""
EDGE_FEATURES = tuple([
	'pos',
	'morph A-1', 'morph A', 'morph A+1',
	'morph B-1', 'morph B', 'morph B+1',
	'lemma A', 'lemma B', 'form A', 'form B', 'B-A'])



class NeuralNetwork:
	
	def __init__(self, model=None, vocab_sizes=None, forms_weights=None, lemmas_weights=None):
		"""
		Constructor. The first keyword argument should be a Keras model. If not
		specified, a new model is created and compiled. In any case, the Keras
		model should be compiled and ready to use right after the init.
		
		The remaining keyword arguments should be set if the first is not (they
		are ignored otherwise).
		"""
		if model is None:
			assert isinstance(vocab_sizes['forms'], int)
			assert isinstance(vocab_sizes['lemmas'], int)
			assert isinstance(vocab_sizes['pos_tags'], int)
			assert isinstance(vocab_sizes['morph'], int)
			self._init_model(vocab_sizes, forms_weights, lemmas_weights)
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
	
	
	def _init_model(self, vocab_sizes, forms_weights=None, lemmas_weights=None):
		"""
		Inits and compiles the Keras model. This method is only called when
		training; for testing, the Keras model is loaded.
		
		The network takes as input the POS tags and the morphological features
		of two nodes and their immediate neighbours (the context), as well as
		the nodes' lemmas and their relative position to each other, and tries
		to predict the probability of an edge between the two.
		"""
		input_branches = []
		inputs = []
		
		pos_input = Input(shape=(10,), dtype='uint8')
		pos_embed = Embedding(vocab_sizes['pos_tags'], 32, input_length=10)
		pos = Flatten()(pos_embed(pos_input))
		
		inputs.append(pos_input)
		input_branches.append(pos)
		
		if vocab_sizes['morph']:
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
			
			inputs.extend([
				morph_a_prev, morph_a, morph_a_next,
				morph_b_prev, morph_b, morph_b_next])
			input_branches.append(morph)
		
		if vocab_sizes['lemmas']:
			lemma_a = Input(shape=(1,), dtype='uint16')
			lemma_b = Input(shape=(1,), dtype='uint16')
			lemma_embed = Embedding(vocab_sizes['lemmas'], 32, input_length=1,
				weights=None if lemmas_weights is None else [lemmas_weights])
			
			lemmas = merge([
				Flatten()(lemma_embed(lemma_a)),
				Flatten()(lemma_embed(lemma_b))], mode='concat')
			
			inputs.extend([lemma_a, lemma_b])
			input_branches.append(lemmas)
		
		if vocab_sizes['forms']:
			form_a = Input(shape=(1,), dtype='uint16')
			form_b = Input(shape=(1,), dtype='uint16')
			form_embed = Embedding(vocab_sizes['forms'], 32, input_length=1,
				weights=None if forms_weights is None else [forms_weights])
			
			forms = merge([
				Flatten()(form_embed(form_a)),
				Flatten()(form_embed(form_b))], mode='concat')
			
			inputs.extend([form_a, form_b])
			input_branches.append(forms)
		
		rel_pos_raw = Input(shape=(1,))
		rel_pos = Dense(32, init='uniform', activation='relu')(rel_pos_raw)
		inputs.append(rel_pos_raw)
		input_branches.append(rel_pos)
		
		x = merge(input_branches, mode='concat')
		x = Dense(128, init='he_uniform', activation='relu')(x)
		x = Dropout(0.25)(x)
		x = Dense(128, init='he_uniform', activation='relu')(x)
		x = Dropout(0.25)(x)
		output = Dense(1, init='uniform', activation='sigmoid')(x)
		
		self.model = Model(input=inputs, output=output)
		
		self.model.compile(optimizer='adamax',
				loss='binary_crossentropy',
				metrics=['accuracy'])
	
	
	def train(self, samples, targets, epochs=10, batch_size=32, on_epoch_end=None):
		"""
		Trains the network. The first arg should be a dict where the keys are
		EDGE_FEATURES and the values numpy arrays. The second one should be a
		single numpy array of 0s and 1s.
		
		The batch size and the number of training epochs are directly passed
		onto the keras model's fit function.
		
		The last keyword arg is expected to be a function; this will be invoked
		at the end of each training epoch with the epoch number as first arg.
		"""
		samples = [samples[key] for key in EDGE_FEATURES if key in samples]
		
		if on_epoch_end:
			callbacks = [LambdaCallback(on_epoch_end=on_epoch_end)]
		else:
			callbacks = []
		
		self.model.fit(samples, targets,
				batch_size=batch_size, shuffle=True,
				nb_epoch=epochs, callbacks=callbacks)
	
	
	def calc_probs(self, samples):
		"""
		Calculates the probabilities of each edge. The arg should be a dict
		where the keys are EDGE_FEATURES and the values are the respective
		numpy arrays.
		"""
		samples = [samples[key] for key in EDGE_FEATURES if key in samples]
		
		return self.model.predict(samples, batch_size=32, verbose=1)
