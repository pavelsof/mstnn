import os
import tempfile

from gensim.models.keyedvectors import KeyedVectors

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
	
	
	def train_on(self, dataset, ignore_forms=False, ignore_lemmas=False,
			ignore_morph=False, epochs=10, batch_size=32, save_checkpoints=False,
			forms_vecs=None, lemmas_vecs=None):
		"""
		Trains one or more mstnn models on the given Dataset. The training's
		batch size and number of epochs can be specified. Word forms, lemmas,
		and/or morphology can be ignored as features. Pre-trained form and/or
		lemma embeddings can be supplied via KeyedVectors instances.
		
		If the save_checkpoints flag is set, a model is saved at the end of
		each training epoch; otherwise the last epoch's weights are used.
		
		This method could be altered so that it can be called multiple times by
		initing the extractor and the neural network in the constructor and
		altering the former's read method.
		"""
		extractor = Extractor(ignore_forms, ignore_lemmas, ignore_morph)
		extractor.read(dataset)
		
		if forms_vecs:
			assert '_' in forms_vecs
			assert '\xa0' in forms_vecs
			extractor.forms = {form: index
						for index, form in enumerate(forms_vecs.index2word)}
		if lemmas_vecs:
			assert '_' in lemmas_vecs
			assert '\xa0' in lemmas_vecs
			extractor.lemmas = {lemma: index
						for index, lemma in enumerate(lemmas_vecs.index2word)}
		
		samples, targets = extractor.extract(dataset, include_targets=True)
		
		ann = NeuralNetwork(vocab_sizes=extractor.get_vocab_sizes(),
				forms_weights=None if forms_vecs is None else forms_vecs.syn0,
				lemmas_weights=None if lemmas_vecs is None else lemmas_vecs.syn0)
		
		if save_checkpoints:
			func = lambda epoch, _: self.save_checkpoint(extractor, ann, epoch)
		else:
			func = None
		
		ann.train(samples, targets, epochs, batch_size, func)
	
	
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



def train(model_fp, train_fp, ud_version=2, ignore_forms=False, ignore_lemmas=False,
			ignore_morph=False, epochs=10, batch_size=32, dev_fp=None, num_best=1,
			forms_word2vec=None, lemmas_word2vec=None):
	"""
	Trains an mstnn model. Expects a path where the models will be written to,
	and a path to a conllu dataset that will be used for training. The epochs,
	batch_size, and ignore_* args are passed on to the train_on method.
	
	The dev_fp optional path should specify a development dataset to check the
	trained model against. The UD version would apply to both datasets. The
	num_best keyword arg specifies the number of best performing checkpoints to
	keep when there is a development dataset to check against.
	
	Paths to pre-trained form and/or lemma embeddings can be specified. These
	are expected to be in binary word2vec format.
	
	This can be seen as the main function of the cli's train command.
	"""
	forms_vecs = None if forms_word2vec is None else \
			KeyedVectors.load_word2vec_format(forms_word2vec, binary=True)
	lemmas_vecs = None if lemmas_word2vec is None else \
			KeyedVectors.load_word2vec_format(lemmas_word2vec, binary=True)
	
	trainer = Trainer(model_fp)
	trainer.train_on(Dataset(train_fp, ud_version),
				ignore_forms, ignore_lemmas, ignore_morph,
				epochs, batch_size, save_checkpoints=True,
				forms_vecs=forms_vecs, lemmas_vecs=lemmas_vecs)
	
	if dev_fp is not None:
		trainer.pick_best(Dataset(dev_fp, ud_version), num_best=num_best)
