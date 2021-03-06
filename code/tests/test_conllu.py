import filecmp
import os.path
import tempfile

from unittest import TestCase

import networkx as nx

from code.conllu import ConlluError, Word, Dataset



class DatasetTestCase(TestCase):
	
	def setUp(self):
		self.dataset = Dataset('data/UD_Basque/eu-ud-dev.conllu', ud_version=1)
	
	
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
		
		self.assertEqual(res[0][0], Word._make([
			1, 'Atenasen', 'Atenas', 'PROPN', '_',
			{'Case': frozenset(['Ine']), 'Definite': frozenset(['Def']), 'Number': frozenset(['Sing'])},
			8, 'nmod', '_', '_']))
	
	
	def test_gen_graphs(self):
		res = []
		
		for graph in self.dataset.gen_graphs():
			self.assertTrue(isinstance(graph, nx.DiGraph))
			res.append(graph)
		
		self.assertEqual(res[0].number_of_nodes(), 10+1)
		self.assertEqual(res[-1].number_of_nodes(), 25+1)
	
	
	def test_write_sentences(self):
		sents = [sent for sent in self.dataset.gen_sentences()]
		
		with tempfile.TemporaryDirectory() as temp_dir:
			dataset = Dataset(os.path.join(temp_dir, 'test'))
			dataset.write_sentences(sents)
			
			self.assertTrue(filecmp.cmp(self.dataset.file_path,
				dataset.file_path, shallow=False))
	
	
	def test_write_graphs(self):
		graphs = [graph for graph in self.dataset.gen_graphs()]
		
		with tempfile.TemporaryDirectory() as temp_dir:
			dataset = Dataset(os.path.join(temp_dir, 'test'))
			dataset.write_graphs(graphs)
			
			self.assertTrue(filecmp.cmp(self.dataset.file_path,
				dataset.file_path, shallow=False))
