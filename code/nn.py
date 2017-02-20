from keras.layers.core import Dense
from keras.models import Sequential

import networkx as nx
import numpy as np

from code.features import featurise_edge



class NeuralNetwork:
	
	def __init__(self):
		"""
		Constructor. Inits and compiles the Keras model.
		"""
		self.model = Sequential([
			Dense(64, input_dim=306, init='uniform', activation='tanh'),
			Dense(64, init='uniform', activation='tanh'),
			Dense(1, init='uniform', activation='softmax')
		])
	
	
	def load(self, path):
		"""
		"""
		pass
	
	
	def save(self, path):
		"""
		"""
		pass
	
	
	def train(self, dataset, epochs=10):
		"""
		Expects a conllu.Dataset instance to train on.
		
		Currently each sample consists of concatenating the POS tag and
		morphology feature vectors of the parent and the child of each edge.
		"""
		samples = []
		targets = []
		
		for graph in dataset.gen_graphs():
			for edge in graph.edges():
				samples.append(featurise_edge(graph, edge))
				targets.append(1)
			
			for edge in nx.non_edges(graph):
				samples.append(featurise_edge(graph, edge))
				targets.append(0)
		
		samples = np.array(samples)
		targets = np.array(targets)
		
		self.model.fit(samples, targets, nb_epoch=epochs)
	
	
	def test(self, samples, targets):
		"""
		"""
		pass
