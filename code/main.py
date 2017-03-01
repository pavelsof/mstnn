import itertools

import networkx as nx

from code.conllu import Dataset, write_graphs
from code.features import Extractor
from code.mst import find_mst, Graph
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
	parsed = []
	
	extractor, neural_net = load_model(model_fp)
	
	dataset = Dataset(data_fp)
	
	for graph in dataset.gen_graphs():
		probs = neural_net.calc_probs(graph, extractor)
		
		nodes = list(graph.nodes())
		scores = {
			edge: probs[index][0]
			for index, edge in enumerate(itertools.permutations(nodes, 2))}
		
		tree = find_mst(Graph(nodes, scores=scores))
		
		new_graph = nx.DiGraph()
		new_graph.add_nodes_from(graph.nodes(data=True))
		new_graph.add_edges_from(tree.edges)
		
		parsed.append(new_graph)
	
	write_graphs('output/test', parsed)



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
