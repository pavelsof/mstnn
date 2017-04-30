import os.path
import tempfile

from unittest import TestCase

from hypothesis.strategies import composite, sampled_from, sets, text
from hypothesis import assume, given

import numpy as np

from code.conllu import Dataset
from code.features import Extractor



@composite
def subdicts(draw, source_dict):
	d = {}
	
	keys = draw(sets(sampled_from(source_dict.keys())))
	for key in keys:
		d[key] = tuple(sorted(draw(
			sets(sampled_from(source_dict[key]), min_size=1))))
	
	return d



dataset = Dataset('data/UD_Basque/eu-ud-dev.conllu', ud_version=1)

extractor = Extractor()
extractor.read(dataset)



class FeaturesTestCase(TestCase):
	
	def test_read(self):
		self.assertTrue(isinstance(extractor.pos_tags, tuple))
		self.assertIn('PROPN', extractor.pos_tags)
		self.assertIn('CONJ', extractor.pos_tags)
		
		self.assertTrue(isinstance(extractor.morph, dict))
		self.assertIn('Case', extractor.morph)
		self.assertIn('Definite', extractor.morph)
		self.assertIn('Number', extractor.morph)
		
		self.assertTrue(isinstance(extractor.lemmas, dict))
		self.assertIn('_', extractor.lemmas)
		self.assertIn('__root__', extractor.lemmas)
		self.assertIn('Atenas', extractor.lemmas)
		self.assertIn('ordea', extractor.lemmas)
	
	
	@given(sets(text()))
	def test_featurise_lemma(self, lemmas):
		assume(all([lemma not in ['_', '__root__'] for lemma in lemmas]))
		
		extractor = Extractor()
		for lemma in lemmas:
			extractor.lemmas[lemma]
		
		self.assertEqual(extractor.get_vocab_sizes()['lemmas'], len(lemmas)+2)
		
		self.assertEqual(extractor.featurise_lemma('_'), 0)
		self.assertEqual(extractor.featurise_lemma('__root__'), 1)
		
		res = [extractor.featurise_lemma(lemma) for lemma in lemmas]
		self.assertEqual(len(res), len(lemmas))
		self.assertTrue(all([number not in [0, 1] for number in res]))
	
	
	@given(sampled_from(extractor.pos_tags))
	def test_featurise_pos_tag(self, pos_tag):
		res = extractor.featurise_pos_tag(pos_tag)
		self.assertTrue(isinstance(res, int))
		self.assertTrue(res > 0)
		self.assertTrue(res <= len(extractor.pos_tags))
	
	
	@given(subdicts(extractor.morph))
	def test_featurise_morph(self, subdict):
		res = extractor.featurise_morph(subdict)
		
		self.assertTrue(isinstance(res, np.ndarray))
		self.assertEqual(len(res),
			sum([len(value) for value in extractor.morph.values()]))
		
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
		self.assertEqual(len([i for i in res if i == 1]),
			sum([len(value) for value in subdict.values()]))
	
	
	def test_model_files(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'model')
			extractor.write_to_model_file(path)
			new_ext = Extractor.create_from_model_file(path)
		
		self.assertTrue(isinstance(new_ext, Extractor))
		self.assertEqual(new_ext.lemmas, extractor.lemmas)
		self.assertEqual(new_ext.morph, extractor.morph)
		self.assertEqual(new_ext.pos_tags, extractor.pos_tags)
	
	
	def test_model_files_error(self):
		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'model')
			with self.assertRaises(OSError):
				Extractor.create_from_model_file(path)
			
			with open(path, 'w') as f:
				f.write('hi')
			
			with self.assertRaises(OSError):
				Extractor.create_from_model_file(path)
		
		assert not os.path.exists(temp_dir)
		
		with self.assertRaises(OSError):
			extractor.write_to_model_file(path)
