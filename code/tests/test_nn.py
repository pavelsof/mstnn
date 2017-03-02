import os.path
import tempfile

from unittest import TestCase

from code.nn import NeuralNetwork



class NeuralNetworkTestCase(TestCase):
	
	def setUp(self):
		self.net = NeuralNetwork(vocab_size=42)
	
	
	def test_model_files_error(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'model')
			
			with self.assertRaises(OSError):
				NeuralNetwork.create_from_model_file(path)
			
			with open(path, 'w') as f:
				f.write('hi')
			
			with self.assertRaises(OSError):
				NeuralNetwork.create_from_model_file(path)
		
		assert not os.path.exists(temp_dir)
		
		with self.assertRaises(OSError):
			self.net.write_to_model_file(path)
