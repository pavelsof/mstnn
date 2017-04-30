import os
import tempfile

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
	
	
	def train_on(self, dataset, epochs=10, save_checkpoints=False):
		"""
		Trains an mstnn model for as many epochs on the given conllu.Dataset
		instance. If the boolean flag is set, a model is saved at the end of
		each training epoch.
		
		This method could be altered so that it can be called multiple times by
		initing the extractor and the neural network in the constructor and
		altering the former's read method.
		"""
		extractor = Extractor()
		extractor.read(dataset)
		
		samples, targets = extractor.extract(dataset, include_targets=True)
		
		ann = NeuralNetwork(vocab_sizes=extractor.get_vocab_sizes())
		
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



def train(model_fp, train_fp, dev_fp=None, ud_version=2, epochs=10):
	"""
	Trains an mstnn model. Expects a path where the models will be written to,
	and a path to a conllu dataset that will be used for training.
	
	The optional path should specify a development dataset to check the trained
	model against. The UD version would apply to both datasets.
	
	This can be seen as the main function of the cli's train command.
	"""
	trainer = Trainer(model_fp)
	trainer.train_on(Dataset(train_fp, ud_version), epochs,
			save_checkpoints=dev_fp is not None)
	
	if dev_fp is not None:
		trainer.pick_best(Dataset(dev_fp, ud_version))
