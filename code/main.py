from code.conllu import Dataset



def train(model_fp, data_fp):
	"""
	Trains an mstnn instance. Expects a path where the trained model will be
	written to, and a path to a .conllu dataset that will be used for training.
	"""
	dataset = Dataset(data_fp)



def test(model_fp, data_fp):
	"""
	Tests a trained mstnn model against a .conllu dataset. Expects the path to
	a previously trained mstnn model file and the path to the dataset file.
	"""
	dataset = Dataset(data_fp)
