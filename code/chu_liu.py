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
	
	
	def create_greedy_edges(self):
		"""
		For each node other than the root, it adds the maximum-score
		child-parent link. Errors from self.scores not containing an edge are
		left propagate.
		
		This method mutates the instance. If the graph already has some edges,
		these are discarded.
		"""
		self.edges = set()
		
		for j in filter(lambda x: x != 0, self.nodes):
			max_score = 0
			parent = None
			
			for i in filter(lambda x: x != j, self.nodes):
				if self.scores[(i, j)] > max_score:
					max_score = self.scores[(i, j)]
					parent = i
			
			if parent is not None:
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
		new_edges = set()
		new_scores = {
			edge: score for edge, score in self.scores.items()
			if edge[0] not in cycle and edge[1] not in cycle}
		
		for edge in self.edges:
			if edge[0] in cycle and edge[1] in cycle:
				pass
			
			elif edge[0] in cycle:
				new_scores[(c_node, edge[1])] = max([
					score for edge_, score in self.scores.items()
					if edge_[0] in cycle and edge_[1] == edge[1]])
				new_edges.add((c_node, edge[1]))
			
			elif edge[1] in cycle:
				new_scores[(edge[0], c_node)] = max([
					score for edge_, score in self.scores.items()
					if edge_[0] == edge[0] and edge_[1] in cycle])
				new_edges.add((edge[0], c_node))
			
			else:
				new_edges.add(edge)
		
		return cycle, Graph(new_nodes, new_edges, new_scores)
	
	
	def expand(self, original_graph, cycle, c_node):
		"""
		Returns a new Graph instance with the specified node expanded back into
		the nodes that had been contracted in order to create it.
		
		This method does not mutate the instance.
		"""
		parent = filter(lambda edge: edge[1] == c_node, self.edges)[0]
		
		new_nodes = (self.nodes - set([c_node])) | cycle
		new_edges = set(self.edges)



def find(graph):
	"""
	The main function of Chu-Liu/Edmond's algorithm. Expects a Graph instance
	and a dictionary with the scores. The latter should have an entry for each
	possible edge in the graph. The graph does not need to have any edges.
	
	Recursively returns a Graph instance that comprises the maximum spanning
	arborescence.
	"""
	graph.create_greedy_edges()
	
	try:
		new_graph, new_scores = graph.contract()
	except ValueError:
		return graph
	
	new_graph = find(new_graph, new_scores)
