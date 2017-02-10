from unittest import TestCase

import networkx as nx

from code.conllu import ConlluError, Word, Dataset



class DatasetTestCase(TestCase):
	
	def setUp(self):
		self.dataset = Dataset('data/UD_Basque/eu-ud-dev.conllu')
	
	
	def test_bad_file(self):
		dataset = Dataset('code/dontexist')
		with self.assertRaises(ConlluError):
			[sent for sent in dataset.gen_sentences()]
		
		dataset = Dataset('code/conllu.py')
		with self.assertRaises(ConlluError):
			[sent for sent in dataset.gen_sentences()]
	
	
	def test_gen_sentences(self):
		res = []
		
		for sent in self.dataset.gen_sentences():
			for index, word in enumerate(sent, 1):
				self.assertTrue(isinstance(word, Word))
				self.assertTrue(isinstance(word.ID, int))
				self.assertTrue(isinstance(word.HEAD, int))
				self.assertEqual(word.ID, index)
			
			res.append(sent)
		
		self.assertEqual(len(res[0]), 10)
		self.assertEqual(len(res[-1]), 25)
	
	
	def test_gen_graphs(self):
		res = []
		
		for graph in self.dataset.gen_graphs():
			self.assertTrue(isinstance(graph, nx.DiGraph))
			res.append(graph)
		
		self.assertEqual(res[0].number_of_nodes(), 10+1)
		self.assertEqual(res[-1].number_of_nodes(), 25+1)
