from unittest import TestCase

from hypothesis.strategies import composite, sampled_from, sets
from hypothesis import given

from code.features import POS_TAGS, featurise_pos_tag
from code.features import DEP_RELS, featurise_dep_rel
from code.features import MORPH_FEATURES, featurise_morph



@composite
def subdicts(draw, source_dict):
	d = {}
	
	keys = draw(sets(sampled_from(source_dict.keys())))
	for key in keys:
		d[key] = tuple(sorted(draw(
			sets(sampled_from(source_dict[key]), min_size=1))))
	
	return d



class FeaturesTestCase(TestCase):
	
	@given(sampled_from(POS_TAGS))
	def test_featurise_pos_tag(self, pos_tag):
		res = featurise_pos_tag(pos_tag)
		self.assertTrue(isinstance(res, list))
		self.assertEqual(len(res), len(POS_TAGS))
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
	
	
	@given(sampled_from(DEP_RELS))
	def test_featurise_dep_rel(self, dep_rel):
		res = featurise_dep_rel(dep_rel)
		self.assertTrue(isinstance(res, list))
		self.assertEqual(len(res), len(DEP_RELS))
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
	
	
	@given(subdicts(MORPH_FEATURES))
	def test_featurise_morph(self, subdict):
		string = '|'.join([
			'{}={}'.format(key, ','.join(values))
			for key, values in subdict.items()])
		
		res = featurise_morph(string)
		
		self.assertTrue(isinstance(res, list))
		self.assertEqual(len(res),
			sum([len(value) for value in MORPH_FEATURES.values()]))
		
		self.assertTrue(all([i == 0 or i == 1 for i in res]))
		self.assertEqual(len([i for i in res if i == 1]),
			sum([len(value) for value in subdict.values()]))
