from code.conllu import Dataset
from code.features import Extractor
from code.nn import NeuralNetwork



def train(model_fp, data_fp, ud_version=2):
	"""
	Trains an mstnn instance. Expects a path where the trained model will be
	written to, and a path to a .conllu dataset that will be used for training.
	"""
	dataset = Dataset(data_fp)
	
	extractor = Extractor(ud_version)
	extractor.read(dataset)
	
	nn = NeuralNetwork()
	nn.train(dataset, extractor)



def test(model_fp, data_fp):
	"""
	Tests a trained mstnn model against a .conllu dataset. Expects the path to
	a previously trained mstnn model file and the path to the dataset file.
	"""
	dataset = Dataset(data_fp)
