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
	
	neural_net = NeuralNetwork()
	neural_net.train(dataset, extractor)
	
	save_model(model_fp, extractor, neural_net)



def test(model_fp, data_fp):
	"""
	Tests a trained mstnn model against a .conllu dataset. Expects the path to
	a previously trained mstnn model file and the path to the dataset file.
	"""
	extractor, neural_net = load_model(model_fp)
	
	dataset = Dataset(data_fp)



"""
mstnn models
"""

def save_model(model_fp, extractor, neural_net):
	"""
	Writes a h5py file to the specified path. The file contains the keras model
	of the given NeuralNetwork and the parameters of the given Extractor.
	
	Raises a ValueError (?) if the path cannot be written.
	
	The file is in fact the keras model file with an added h5py group at the
	root level to hold the extractor data.
	"""
	neural_net.write_to_model_file(model_fp)
	extractor.write_to_model_file(model_fp)



def load_model(model_fp):
	"""
	Expects a file previously written using save_model and returns an Extractor
	and a NeuralNetwork instances.
	
	Raises a ValueError if there is a problem with reading the file or if this
	is not in the expected format.
	"""
	extractor = Extractor.create_from_model_file(model_fp)
	neural_net = NeuralNetwork.create_from_model_file(model_fp)
	
	return extractor, neural_net
