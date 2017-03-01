"""
Impements the Chu-Liu/Edmonds' algorithm [0] for finding a spanning tree of
minimum weight. This module does not know anything about sentence graphs, UD
grammars, etc; its usage involves providing a set of nodes and the scores for
each pair of those nodes and getting back the minimum spanning tree:
	
	tree = mst.find(mst.Graph(nodes, scores=scores))

The result is also an mst.Graph instance, but with its edges (tree.edges) being
set by the algorithm.

[0] https://en.wikipedia.org/wiki/Edmonds'_algorithm
"""

import networkx as nx



class Graph:
	
	def __init__(self, nodes, edges=[], scores={}):
		"""
		Constructor. Expects a sequence of the node names that the graph should
		contain. The node 0 is considered to be the root node, so there must be
		such a node in the sequence (otherwise a ValueError is raised).
		
		The second argument, if provided, should be a sequence of (i, j) tuples
		where each tuple represents a dependency between nodes i (the head) and
		j (the dependant).
		
		The third argument, if provided, should be a {} with edges as keys and
		scores as values. Note that all possible edges should be available, not
		only the ones that graph is inited with.
		"""
		if 0 not in nodes:
			raise ValueError('Could not find node 0')
		
		self.nodes = set(nodes)
		self.edges = set(edges)
		
		self.scores = scores
	
	
	def __str__(self):
		"""
		Returns a string with the graph's nodes, edges and scores. Useful for
		debugging.
		"""
		def set_to_str(s):
			return ', '.join(map(str, sorted(s)))
		
		return '\n'.join([
			'nodes: {}'.format(set_to_str(self.nodes)),
			'edges: {}'.format(set_to_str(self.edges)),
			'scores: {}'.format(', '.join(['{}: {}'.format(key, value)
				for key, value in sorted(self.scores.items())]))])
	
	
	def create_greedy_edges(self):
		"""
		For each node other than the root, it adds the maximum-score
		child-parent link. Raises a ValueError if no incoming edge can be added
		for any of the non-root nodes.
		
		This method mutates the instance. If the graph already has some edges,
		these are discarded.
		"""
		self.edges = set()
		
		for j in filter(lambda x: x != 0, self.nodes):
			max_score = -1
			parent = None
			
			for i in filter(lambda x: x != j, self.nodes):
				if (i, j) not in self.scores:
					continue
				
				if self.scores[(i, j)] > max_score:
					max_score = self.scores[(i, j)]
					parent = i
			
			if parent is None:
				raise ValueError('No incoming edge for node {}'.format(j))
			
			self.edges.add((parent, j))
	
	
	def find_cycle(self):
		"""
		Returns a set of nodes that form a cycle. If there is more than one
		cycle in the graph, one is chosen arbitrarily. If there is no cycle,
		raises a ValueError.
		
		This method does not mutate the instance.
		"""
		nx_graph = nx.DiGraph()
		nx_graph.add_nodes_from(self.nodes)
		nx_graph.add_edges_from(self.edges)
		
		try:
			cycle = nx.find_cycle(nx_graph)
		except nx.NetworkXNoCycle:
			raise ValueError('Could not find a cycle')
		
		return set([edge[0] for edge in cycle] + [edge[1] for edge in cycle])
	
	
	def contract(self, c_node):
		"""
		Returns a new Graph instance comprising a contracted minor of the this
		instance. Contracting consists of finding a cycle and replacing the
		nodes that form that with a new node. If the graph does not have any
		cycles, a ValueError is raised.
		
		This method does not mutate the instance.
		"""
		cycle = self.find_cycle()
		
		new_nodes = (self.nodes - cycle) | set([c_node])
		new_scores = {}
		
		for edge, score in self.scores.items():
			if edge[0] in cycle and edge[1] in cycle:
				continue
			
			elif edge[0] in cycle:
				new_scores[(c_node, edge[1])] = max([
					score for edge_, score in self.scores.items()
					if edge_[0] in cycle and edge_[1] == edge[1]])
			
			elif edge[1] in cycle:
				new_scores[(edge[0], c_node)] = max([
					score for edge_, score in self.scores.items()
					if edge_[0] == edge[0] and edge_[1] in cycle])
			
			else:
				new_scores[edge] = score
		
		assert all([edge[0] in new_nodes and edge[1] in new_nodes
			for edge in new_scores.keys()])
		
		return cycle, Graph(new_nodes, set(), new_scores)
	
	
	def expand(self, orig_graph, cycle, c_node):
		"""
		Returns a new Graph instance with the specified node expanded back into
		the nodes that had been contracted in order to create it.
		
		This method does not mutate the instance.
		"""
		new_edges = set()
		for edge in self.edges:
			if edge[0] == c_node:
				new_edges.add(next((
					arc for arc, score in orig_graph.scores.items()
					if arc[1] == edge[1] and arc[0] in cycle
					and score == self.scores[edge])))
			elif edge[1] == c_node:
				new_edges.add(next((
					arc for arc, score in orig_graph.scores.items()
					if arc[0] == edge[0] and arc[1] in cycle
					and score == self.scores[edge])))
			else:
				new_edges.add(edge)
		
		cycle_parent = next((
			edge[0] for edge in self.edges if edge[1] == c_node))
		edge_to_del = next((
			edge for edge in orig_graph.edges
			if edge[0] in cycle and edge[1] in cycle
			and (cycle_parent, edge[1]) in new_edges))
		
		for edge in orig_graph.edges:
			if edge[0] in cycle and edge[1] in cycle and edge != edge_to_del:
				new_edges.add(edge)
		
		assert all([
			edge[0] in orig_graph.nodes and edge[1] in orig_graph.nodes
			for edge in new_edges])
		
		return Graph(orig_graph.nodes, new_edges, orig_graph.scores)



def find_mst(graph):
	"""
	The main function of Chu-Liu/Edmond's algorithm. Expects a Graph instance
	and a dictionary with the scores. The latter should have an entry for each
	possible edge in the graph. The graph does not need to have any edges.
	
	Recursively returns a Graph instance that comprises the maximum spanning
	arborescence.
	"""
	graph.create_greedy_edges()
	
	c_node = generate_node_name(graph.nodes)
	
	try:
		cycle, new_graph = graph.contract(c_node)
	except ValueError:
		return graph
	
	new_graph = find_mst(new_graph)
	
	return new_graph.expand(graph, cycle, c_node)



def generate_node_name(nodes):
	"""
	Returns a fresh node name not present in the given set. Assumes that node
	names are integers.
	"""
	return max(nodes) + 1
