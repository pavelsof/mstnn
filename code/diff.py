from code.conllu import Dataset



class DiffError(ValueError):
	"""
	Raised when there is a problem with producing the diff. Should be inited
	with a human-friendly error message.
	"""
	pass



def diff(fp1, fp2):
	"""
	Prints the parses that are not the same in both datasets. Expects the paths
	to two conllu files comprising parses of the same sentences.
	
	Could raise a ConlluError or a DiffError.
	"""
	dataset1 = Dataset(fp1)
	dataset2 = Dataset(fp2)
	
	for graph1, graph2 in zip(dataset1.gen_graphs(), dataset2.gen_graphs()):
		if len(graph1) != len(graph2):
			raise DiffError('Sentence mismatch')
		
		output = []
		is_same = True
		
		loop = zip(graph1.nodes(data=True), graph2.nodes(data=True))
		for (node1, data1), (node2, data2) in loop:
			if node1 == 0: data1['FORM'] = 'ROOT'
			if node2 == 0: data2['FORM'] = 'ROOT'
			
			try:
				assert node1 == node2
				assert data1['FORM'] == data2['FORM']
			except AssertionError:
				raise DiffError('Sentence mismatch')
			
			edges1 = ','.join(map(str, graph1.edge[node1]))
			edges2 = ','.join(map(str, graph2.edge[node2]))
			
			if edges1 != edges2:
				is_same = False
			
			output.append('{!s}\t{}\t\t{}\t{}'.format(
				node1, data1['FORM'], edges1, edges2))
		
		if not is_same:
			print('\n'.join(output) +'\n')
	
	return
