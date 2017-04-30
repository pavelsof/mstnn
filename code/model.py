from itertools import permutations

import networkx as nx

from code.conllu import Dataset
from code.features import Extractor
from code.mst import find_mst, Graph
from code.nn import NeuralNetwork



class Model:
	"""
	Represents a trained mstnn model that can be saved into a file, loaded from
	one, and used to parse a dataset.
	"""
	
	def __init__(self, extractor, neural_net):
		"""
		Constructor. Expects an Extractor and a NeuralNetwork instances; these
		two together define an mstnn model.
		"""
		assert isinstance(extractor, Extractor)
		assert isinstance(neural_net, NeuralNetwork)
		
		self.extractor = extractor
		self.neural_net = neural_net
	
	
	@classmethod
	def load(cls, model_fp):
		"""
		Returns an instance of the class loaded from the specified file. The
		latter is expected to be a h5py file previously written using the save
		method below.
		
		Raises an OSError if there is a problem with reading the file or if it
		is not in the expected format.
		"""
		extractor = Extractor.create_from_model_file(model_fp)
		neural_net = NeuralNetwork.create_from_model_file(model_fp)
		
		return cls(extractor, neural_net)
	
	
	def save(self, model_fp):
		"""
		Writes a h5py file to the specified file path. The file contains the
		keras model of the given NeuralNetwork and the parameters of the given
		Extractor. If an epoch is specified, it is appended to the file path.
		
		Raises an OSError if the path cannot be written.
		
		The file is in fact the keras model file with an added h5py group at
		the root level to hold the extractor data.
		"""
		self.neural_net.write_to_model_file(model_fp)
		self.extractor.write_to_model_file(model_fp)
	
	
	def parse(self, dataset):
		"""
		Parses the given Dataset. Returns the parsed sentences in the form of a
		[] of DiGraph instances.
		"""
		samples = self.extractor.extract(dataset, include_targets=False)
		probs = self.neural_net.calc_probs(samples)
		
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
		
		return parsed



def parse(model_fp, data_fp, output_fp):
	"""
	Parses a conllu dataset using a trained mstnn model. Expects the path to a
	previously trained mstnn model file, the path to the dataset file, and the
	path where to write the parsed sentences.
	
	This can be seen as the main function of the cli's parse command.
	"""
	model = Model.load(model_fp)
	dataset = Dataset(data_fp)
	
	Dataset(output_fp).write_graphs(model.parse(dataset))
