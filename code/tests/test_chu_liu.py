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
	
	@given(integers(min_value=1, max_value=10).flatmap(
			lambda x: square_matrices(x)))
	def test_create_greedy_edges(self, matrix):
		size = len(matrix)
		scores = {
			(i, j): matrix[i][j]
			for i in range(size) for j in range(size)}
		
		graph = Graph(range(size))
		graph.create_greedy_edges(scores)
		
		for parent, child in graph.edges:
			self.assertEqual(
				matrix[parent][child],
				max([matrix[i][child] for i in range(size) if i != child]))
	
	
	def test_find_cycle(self):
		graph = Graph(range(3))
		
		graph.edges = set([(0, 1), (1, 2), (2, 0)])
		self.assertEqual(graph.find_cycle(), set([0, 1, 2]))
		
		graph.edges = set([(0, 1), (0, 2)])
		with self.assertRaises(ValueError):
			graph.find_cycle()
	
	
	def test_contract(self):
		graph = Graph(range(4), [(0, 1), (1, 2), (2, 1), (2, 3)])
		new_graph, new_scores = graph.contract('c', {
			(0, 1): 1, (1, 2): 2, (2, 1): 3, (2, 3): 4})
		
		self.assertEqual(new_graph.nodes, set([0, 'c', 3]))
		self.assertEqual(new_graph.edges, set([(0, 'c'), ('c', 3)]))
		
		self.assertEqual(new_scores, {(0, 'c'): 1, ('c', 3): 4})
