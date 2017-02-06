from unittest import TestCase

from hypothesis.strategies import integers, lists, sets
from hypothesis.strategies import composite
from hypothesis import given

from code.mst import Graph, find_mst, generate_node_name



@composite
def square_matrices(draw, size):
	return draw(lists(lists(integers(min_value=0, max_value=100),
				min_size=size, max_size=size),
				min_size=size, max_size=size))



class ChuLiuTestCase(TestCase):
	
	@given(integers(min_value=1, max_value=10).flatmap(
			lambda x: square_matrices(x)))
	def test_create_greedy_edges(self, matrix):
		size = len(matrix)
		scores = {
			(i, j): matrix[i][j]
			for i in range(size) for j in range(size)}
		
		graph = Graph(range(size), scores=scores)
		graph.create_greedy_edges()
		
		for parent, child in graph.edges:
			self.assertEqual(
				matrix[parent][child],
				max([matrix[i][child] for i in range(size) if i != child]))
		
		# ensure each node apart from the root has a parent
		for node in range(1, size):
			self.assertTrue(any([edge[1] == node for edge in graph.edges]))
	
	
	def test_find_cycle(self):
		graph = Graph(range(3))
		
		graph.edges = set([(0, 1), (1, 2), (2, 0)])
		self.assertEqual(graph.find_cycle(), set([0, 1, 2]))
		
		graph.edges = set([(0, 1), (0, 2)])
		with self.assertRaises(ValueError):
			graph.find_cycle()
	
	
	@given(square_matrices(4))
	def test_contract_and_expand(self, matrix):
		scores = {(i, j): matrix[i][j] for i in range(4) for j in range(4)}
		
		graph1 = Graph(range(4), [(0, 1), (1, 2), (2, 1), (2, 3)], scores)
		
		cycle, graph2 = graph1.contract('c')
		graph2.create_greedy_edges()
		
		self.assertEqual(cycle, set([1, 2]))
		self.assertEqual(graph2.nodes, set([0, 'c', 3]))
		self.assertFalse(any([
			edge[0] in cycle or edge[1] in cycle
			for edge in graph2.scores.keys()]))
		
		graph3 = graph2.expand(graph1, cycle, 'c')
		self.assertEqual(graph3.nodes, set(range(4)))
		self.assertEqual(graph3.scores, scores)
	
	
	@given(square_matrices(5))
	def test_contract_and_expand2(self, matrix):
		scores = {(i, j): matrix[i][j] for i in range(5) for j in range(5)}
		
		graph1 = Graph(range(5),
			[(0, 1), (1, 2), (2, 3), (3, 1), (3, 4)], scores)
		
		cycle, graph2 = graph1.contract('c')
		graph2.create_greedy_edges()
		
		self.assertEqual(cycle, set([1, 2, 3]))
		self.assertEqual(graph2.nodes, set([0, 'c', 4]))
		self.assertFalse(any([
			edge[0] in cycle or edge[1] in cycle
			for edge in graph2.scores.keys()]))
		
		graph3 = graph2.expand(graph1, cycle, 'c')
		self.assertEqual(graph3.nodes, set(range(5)))
		self.assertEqual(graph3.scores, scores)
	
	
	@given(sets(integers(), min_size=1))
	def test_generate_node_name(self, nodes):
		new_node = generate_node_name(nodes)
		self.assertEqual(len(nodes | set([new_node])), len(nodes)+1)
	
	
	@given(integers(min_value=1, max_value=10).flatmap(
			lambda x: square_matrices(x)))
	def test_find_mst_does_not_break(self, matrix):
		size = len(matrix)
		scores = {
			(i, j): matrix[i][j]
			for i in range(size) for j in range(size)}
		
		graph = Graph(range(size), scores=scores)
		
		res = find_mst(graph)
		
		self.assertTrue(isinstance(res, Graph))
		self.assertEqual(res.nodes, set(range(size)))
		self.assertEqual(res.scores, scores)
		
		# ensure each node apart from the root has a parent
		for node in range(1, size):
			self.assertTrue(any([edge[1] == node for edge in res.edges]))
