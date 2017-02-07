from unittest import TestCase

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
				self.assertEqual(word.ID, index)
			
			res.append(sent)
		
		self.assertEqual(len(res[0]), 10)
		self.assertEqual(len(res[-1]), 25)
