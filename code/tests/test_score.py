from unittest import TestCase

from code.conllu import Dataset
from code.score import Scorer, score



class ScoreTestCase(TestCase):
	
	def setUp(self):
		self.path = 'data/UD_Basque/eu-ud-dev.conllu'
		self.dataset = Dataset(self.path, ud_version=1)
	
	def test_scorer(self):
		scorer = Scorer(self.dataset)
		self.assertEqual(scorer.score(self.dataset), 100)
	
	def test_score(self):
		self.assertEqual(score(self.path, self.path, ud_version=1), 100)
