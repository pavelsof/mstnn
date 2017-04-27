from itertools import permutations

import networkx as nx

from code.conllu import Dataset
from code.features import Extractor
from code.mst import find_mst, Graph
from code.nn import NeuralNetwork
from code.score import score



def train(model_fp, train_fp, dev_fp=None, ud_version=2, epochs=10):
	"""
	Trains an mstnn model. Expects a path where the models will be written to,
	and a path to a conllu dataset that will be used for training.
	"""
	dataset = Dataset(train_fp, ud_version)
	
	extractor = Extractor()
	extractor.read(dataset)
	samples, targets = extractor.extract(dataset, include_targets=True)
	
	neural_net = NeuralNetwork(vocab_sizes=extractor.get_vocab_sizes())
	on_epoch_end = lambda epoch, _: save_model(model_fp, extractor, neural_net, epoch)
	neural_net.train(samples, targets, epochs=epochs, on_epoch_end=on_epoch_end)
	
	if dev_fp is not None:
		for epoch in range(epochs):
			output_fp = model_fp + '-p{:02}'.format(epoch)
			parse(model_fp + '-e{:02}'.format(epoch), dev_fp, output_fp)
			print('epoch {}: {:.2f}'.format(epoch, score(output_fp, dev_fp)))
	
	# save_model(model_fp, extractor, neural_net)



def parse(model_fp, data_fp, output_fp):
	"""
	Parses a .conllu dataset using a trained mstnn model. Expects the path to a
	previously trained mstnn model file, the path to the dataset file, and the
	path where to write the parsed sentences.
	"""
	extractor, neural_net = load_model(model_fp)
	
	dataset = Dataset(data_fp)
	
	samples = extractor.extract(dataset, include_targets=False)
	probs = neural_net.calc_probs(samples)
	
	parsed = []
	count = 0
	
	for edgeless_graph in dataset.gen_graphs():
		nodes = edgeless_graph.nodes()
		scores = {}
		
		for index, (a, b) in enumerate(permutations(nodes, 2)):
			scores[(a, b)] = probs[count+index][0]
		
		count = count + index + 1
		
		tree = find_mst(Graph(nodes, scores=scores))
		
		new_graph = nx.DiGraph()
		new_graph.add_nodes_from(edgeless_graph.nodes(data=True))
		new_graph.add_edges_from(tree.edges)
		
		parsed.append(new_graph)
	
	Dataset(output_fp).write_graphs(parsed)



"""
mstnn models
"""

def save_model(model_fp, extractor, neural_net, epoch=None):
	"""
	Writes a h5py file to the specified path. The file contains the keras model
	of the given NeuralNetwork and the parameters of the given Extractor. If an
	epoch is specified, it is appended to the file path.
	
	Raises an OSError if the path cannot be written.
	
	The file is in fact the keras model file with an added h5py group at the
	root level to hold the extractor data.
	"""
	if epoch is not None:
		model_fp = model_fp + '-e{:02}'.format(epoch)
	
	neural_net.write_to_model_file(model_fp)
	extractor.write_to_model_file(model_fp)



def load_model(model_fp):
	"""
	Expects a file previously written using save_model and returns an Extractor
	and a NeuralNetwork instances.
	
	Raises an OSError if there is a problem with reading the file or if this is
	not in the expected format.
	"""
	extractor = Extractor.create_from_model_file(model_fp)
	neural_net = NeuralNetwork.create_from_model_file(model_fp)
	
	return extractor, neural_net
