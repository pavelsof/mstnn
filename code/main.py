import random

import networkx as nx

from code.conllu import Dataset
from code.features import Extractor
from code.mst import find_mst, Graph
from code.nn import NeuralNetwork



"""
Probably not the best place to set the random seed.
"""
random.seed(42)



def train(model_fp, data_fp, ud_version=2):
	"""
	Trains an mstnn model. Expects a path where the model will be written to,
	and a path to a .conllu dataset that will be used for training.
	"""
	dataset = Dataset(data_fp)
	
	extractor = Extractor(ud_version)
	extractor.read(dataset)
	
	neural_net = NeuralNetwork(vocab_size=extractor.get_lemma_vocab_size())
	neural_net.train(dataset, extractor)
	
	save_model(model_fp, extractor, neural_net)



def parse(model_fp, data_fp, output_fp):
	"""
	Parses a .conllu dataset using a trained mstnn model. Expects the path to a
	previously trained mstnn model file, the path to the dataset file, and the
	path where to write the parsed sentences.
	"""
	parsed = []
	
	extractor, neural_net = load_model(model_fp)
	
	dataset = Dataset(data_fp)
	
	for graph in dataset.gen_graphs():
		scores = neural_net.calc_probs(graph, extractor)
		
		tree = find_mst(Graph(list(graph.nodes()), scores=scores))
		
		new_graph = nx.DiGraph()
		new_graph.add_nodes_from(graph.nodes(data=True))
		new_graph.add_edges_from(tree.edges)
		
		parsed.append(new_graph)
	
	Dataset(output_fp).write_graphs(parsed)



"""
mstnn models
"""

def save_model(model_fp, extractor, neural_net):
	"""
	Writes a h5py file to the specified path. The file contains the keras model
	of the given NeuralNetwork and the parameters of the given Extractor.
	
	Raises an OSError if the path cannot be written.
	
	The file is in fact the keras model file with an added h5py group at the
	root level to hold the extractor data.
	"""
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
