import os
import tempfile

from gensim.models.keyedvectors import KeyedVectors

import numpy as np

from code.conllu import Dataset
from code.features import Extractor
from code.model import Model
from code.nn import NeuralNetwork
from code.score import Scorer



class Trainer:
	
	def __init__(self, model_fp):
		"""
		Constructor. Expects a path determining where the trained model(s) are
		to be saved.
		"""
		self.model_dir, self.model_name = os.path.split(model_fp)
		self.checkpoints = []
	
	
	def save_checkpoint(self, extractor, neural_net, epoch):
		"""
		Saves a checkpoints, i.e. an mstnn model storing the weights at the end
		of a training epoch.
		"""
		name = '{}-e{:02}'.format(self.model_name, epoch)
		path = os.path.join(self.model_dir, name)
		
		Model(extractor, neural_net).save(path)
		self.checkpoints.append(path)
	
	
	def train_on(self, dataset, epochs=10, save_checkpoints=False, forms_indices=None, forms_weights=None):
		"""
		Trains an mstnn model for as many epochs on the given conllu.Dataset
		instance. If the boolean flag is set, a model is saved at the end of
		each training epoch.
		
		This method could be altered so that it can be called multiple times by
		initing the extractor and the neural network in the constructor and
		altering the former's read method.
		"""
		extractor = Extractor(forms_indices=forms_indices)
		extractor.read(dataset)
		
		samples, targets = extractor.extract(dataset, include_targets=True)
		
		ann = NeuralNetwork(vocab_sizes=extractor.get_vocab_sizes(), forms_weights=forms_weights)
		
		if save_checkpoints:
			func = lambda epoch, _: self.save_checkpoint(extractor, ann, epoch)
		else:
			func = None
		
		ann.train(samples, targets, epochs=epochs, on_epoch_end=func)
	
	
	def pick_best(self, dataset, num_best=1):
		"""
		Assuming that checkpoints have been saved during training, deletes all
		but those num_best that are performing best against the given Dataset.
		"""
		if num_best >= len(self.checkpoints):
			return
		
		scorer = Scorer(dataset)
		scores = {}
		
		with tempfile.TemporaryDirectory() as temp_dir:
			for path in self.checkpoints:
				parsed = Model.load(path).parse(dataset)
				
				output_fp = os.path.join(temp_dir, os.path.basename(path))
				Dataset(output_fp).write_graphs(parsed)
				
				scores[path] = scorer.score(Dataset(output_fp))
				print('{}: {:.2f}'.format(path, scores[path]))
		
		best = [(uas, path) for path, uas in scores.items()]
		best = sorted(best, reverse=True)[:num_best]
		best = [item[1] for item in best]
		
		for path in self.checkpoints:
			if path not in best:
				os.remove(path)
		
		print('kept: {}'.format(', '.join(best)))



def train(model_fp, train_fp, forms_fp=None, dev_fp=None, ud_version=2, num_best=1, epochs=10):
	"""
	Trains an mstnn model. Expects a path where the models will be written to,
	and a path to a conllu dataset that will be used for training.
	
	The optional path should specify a development dataset to check the trained
	model against. The UD version would apply to both datasets. The last
	keyword argument specifies the number of best performing checkpoints to
	keep when there is a development dataset to check against.
	
	This can be seen as the main function of the cli's train command.
	"""
	if forms_fp:
		keyed = KeyedVectors.load_word2vec_format(forms_fp)
		
		forms_indices = {word: index
			for index, word in enumerate(keyed.index2word, 1)}
		forms_indices['_'] = 0
		forms_indices['__root__'] = forms_indices.pop('</s>')
		
		forms_weights = np.concatenate([
			np.zeros(shape=(1, 50,), dtype=np.float32),
			keyed.syn0])
	else:
		forms_indices = None
		forms_weights = None
	
	trainer = Trainer(model_fp)
	trainer.train_on(Dataset(train_fp, ud_version), epochs,
			save_checkpoints=dev_fp is not None,
			forms_indices=forms_indices, forms_weights=forms_weights)
	
	if dev_fp is not None:
		trainer.pick_best(Dataset(dev_fp, ud_version), num_best=num_best)
