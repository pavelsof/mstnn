import networkx as nx



class Graph:
	
	def __init__(self, num_nodes):
		"""
		Constructor. Expects the number of nodes that the graph should contain.
		The node 0 is considered to be the root node.
		
		self.edges is a set of (i, j) tuples where each tuple represents a
		dependency between nodes i (the head) and j (the dependant)
		"""
		self.num_nodes = num_nodes
		self.edges = set()
	
	
	def create_greedy_edges(self, scores):
		"""
		For each node other than the root, it adds the maximum-score
		child-parent link. Expects a {(i, j): score} dict.
		
		This method mutates the instance.
		"""
		for j in range(1, self.num_nodes):
			max_score = 0
			parent = None
			
			for i in filter(lambda x: x != j, range(0, self.num_nodes)):
				if scores[(i, j)] > max_score:
					max_score = scores[(i, j)]
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
		nx_graph.add_nodes_from(list(range(self.num_nodes)))
		nx_graph.add_edges_from(self.edges)
		
		try:
			cycle = nx.find_cycle(nx_graph)
		except nx.NetworkXNoCycle:
			raise ValueError('Could not find a cycle')
		
		return set([edge[0] for edge in cycle] + [edge[1] for edge in cycle])
	
	
	def contract(self):
		"""
		Returns a new Graph instance comprising a contracted minor of the this
		instance. Contracting consists of finding a cycle and replacing the
		nodes that form that with a new node. If the graph does not have any
		cycles, a ValueError is raised.
		
		This method does not mutate the instance.
		"""
		pass



def find():
	"""
	The main function of Chu-Liu/Edmond's algorithm.
	"""
	pass
