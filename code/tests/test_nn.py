from unittest import TestCase

import numpy as np

from code.nn import NeuralNetwork



class NeuralNetworkTestCase(TestCase):
	
	def setUp(self):
		self.net = NeuralNetwork()
	
	
	def test_train(self):
		pass