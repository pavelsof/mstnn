from keras.layers.core import Dense, Flatten, Merge
from keras.layers.embeddings import Embedding
from keras.models import Sequential

import networkx as nx
import numpy as np



class NeuralNetwork:
	
	def __init__(self):
		"""
		Constructor. Inits and compiles the Keras model.
		"""
		grammar_branch = Sequential([
			Dense(64, input_dim=244, init='uniform', activation='tanh')
		])
		
		lexicon_branch = Sequential([
			Embedding(4222, 64, input_length=2),
			Flatten()
		])
		
		self.model = Sequential([
			Merge([grammar_branch, lexicon_branch], mode='concat'),
			Dense(128, init='uniform', activation='tanh'),
			Dense(1, init='uniform', activation='softmax')
		])
		
		self.model.compile(optimizer='sgd', loss='mse', metrics=['accuracy'])
	
	
	def load(self, path):
		"""
		"""
		pass
	
	
	def save(self, path):
		"""
		"""
		pass
	
	
	def train(self, dataset, extractor, epochs=2):
		"""
		Expects a conllu.Dataset instance to train on and a features.Extractor
		instance to extract the feature vectors with.
		
		Currently each sample consists of concatenating the POS tag and
		morphology feature vectors of the parent and the child of each edge.
		"""
		samples_grammar = []
		samples_lexicon = []
		targets = []
		
		for graph in dataset.gen_graphs():
			for edge in graph.edges():
				samples_grammar.append(extractor.featurise_edge(graph, edge))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[edge[0]]['LEMMA']),
					extractor.featurise_lemma(graph.node[edge[1]]['LEMMA'])
				])
				targets.append(1)
			
			for edge in nx.non_edges(graph):
				samples_grammar.append(extractor.featurise_edge(graph, edge))
				samples_lexicon.append([
					extractor.featurise_lemma(graph.node[edge[0]]['LEMMA']),
					extractor.featurise_lemma(graph.node[edge[1]]['LEMMA'])
				])
				targets.append(0)
		
		samples_grammar = np.array(samples_grammar)
		samples_lexicon = np.array(samples_lexicon)
		targets = np.array(targets)
		
		self.model.fit([samples_grammar, samples_lexicon], targets, nb_epoch=epochs)
	
	
	def test(self, dataset, extractor):
		"""
		"""
		pass
