from unittest import TestCase

from hypothesis.strategies import integers, lists
from hypothesis.strategies import composite
from hypothesis import given

from code.chu_liu import Graph



@composite
def square_matrices(draw, size):
	return draw(lists(lists(integers(min_value=0, max_value=100),
				min_size=size, max_size=size),
				min_size=size, max_size=size))



class GraphTestCase(TestCase):
	
	@given(integers(min_value=0, max_value=10).flatmap(
			lambda x: square_matrices(x)))
	def test_create_greedy_edges(self, matrix):
		size = len(matrix)
		scores = {
			(i, j): matrix[i][j]
			for i in range(size) for j in range(size)}
		
		graph = Graph(size)
		graph.create_greedy_edges(scores)
		
		for parent, child in graph.edges:
			self.assertEqual(
				matrix[parent][child],
				max([matrix[i][child] for i in range(size) if i != child]))
	
	
	def test_find_cycle(self):
		graph = Graph(3)
		graph.edges = set([(0, 1), (1, 2), (2, 0)])
		self.assertEqual(graph.find_cycle(), set([0, 1, 2]))
