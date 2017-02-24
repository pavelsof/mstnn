import os.path
import tempfile

from unittest import TestCase

from hypothesis.strategies import composite, dictionaries, integers
from hypothesis.strategies import sampled_from, sets, text
from hypothesis import given

import numpy as np

from code.features import Extractor, FeatureError

from code.ud import POS_TAGS_V2 as POS_TAGS
from code.ud import DEP_RELS_V2 as DEP_RELS
from code.ud import MORPH_FEATURES_V2 as MORPH_FEATURES



@composite
def subdicts(draw, source_dict):
	d = {}
	
	keys = draw(sets(sampled_from(source_dict.keys())))
	for key in keys:
		d[key] = tuple(sorted(draw(
			sets(sampled_from(source_dict[key]), min_size=1))))
	
	return d



class FeaturesTestCase(TestCase):
	
	def setUp(self):
		self.ext = Extractor(ud_version=2)
	
	
	@given(sampled_from(POS_TAGS))
	def test_featurise_pos_tag(self, pos_tag):
		res = self.ext.featurise_pos_tag(pos_tag)
		self.assertTrue(isinstance(res, np.ndarray))
		self.assertEqual(len(res), len(POS_TAGS))
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
	
	
	@given(sampled_from(DEP_RELS))
	def test_featurise_dep_rel(self, dep_rel):
		res = self.ext.featurise_dep_rel(dep_rel)
		self.assertTrue(isinstance(res, np.ndarray))
		self.assertEqual(len(res), len(DEP_RELS))
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
	
	
	@given(subdicts(MORPH_FEATURES))
	def test_featurise_morph(self, subdict):
		if subdict:
			string = '|'.join([
				'{}={}'.format(key, ','.join(values))
				for key, values in subdict.items()])
		else:
			string = '_'
		
		res = self.ext.featurise_morph(string)
		
		self.assertTrue(isinstance(res, np.ndarray))
		self.assertEqual(len(res),
			sum([len(value) for value in MORPH_FEATURES.values()]))
		
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
		self.assertEqual(len([i for i in res if i == 1]),
			sum([len(value) for value in subdict.values()]))
	
	
	def test_feature_error(self):
		with self.assertRaises(FeatureError):
			self.ext.featurise_pos_tag('_')
		
		with self.assertRaises(FeatureError):
			self.ext.featurise_dep_rel('_')
		
		with self.assertRaises(FeatureError):
			self.ext.featurise_morph('')
	
	
	@given(dictionaries(text(), integers()))
	def test_model_files(self, lemmas):
		for key, value in lemmas.items():
			self.ext.lemmas[key] = value
		
		with tempfile.TemporaryDirectory() as temp_dir:
			path = os.path.join(temp_dir, 'model')
			self.ext.write_to_model_file(path)
			new_ext = Extractor.create_from_model_file(path)
		
		self.assertTrue(isinstance(new_ext, Extractor))
		self.assertEqual(new_ext.ud_version, self.ext.ud_version)
		self.assertEqual(new_ext.lemmas, self.ext.lemmas)
	
	
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
			self.ext.write_to_model_file(path)
