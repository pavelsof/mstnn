"""
Handles reading datasets in the CoNLL-U format [0]:
	
	dataset = Dataset(path)
	for graph in dataset.gen_graphs():
		...

The resulting graphs are nx.DiGraph instances, one for each sentence.

[0] http://universaldependencies.org/format.html
"""

from collections import namedtuple

import networkx as nx



class ConlluError(ValueError):
	"""
	Raised when there is a problem with reading a file in CoNLL-U format.
	"""
	pass



"""
Represents a word, i.e. a row with an integer ID (excluding multiword tokens
and empty nodes). With the exception of ID and HEAD which are integers, the
other fields are non-empty strings.
"""
Word = namedtuple('Word', [
	'ID', 'FORM', 'LEMMA',
	'UPOSTAG', 'XPOSTAG', 'FEATS',
	'HEAD', 'DEPREL', 'DEPS', 'MISC'])



class Dataset:
	
	def __init__(self, file_path):
		"""
		Constructor. Expects the path to the .conllu dataset file. The latter
		is not opened until one of the gen_* methods is invoked.
		"""
		self.file_path = file_path
	
	
	def gen_sentences(self):
		"""
		Generator that yields a tuple of Word named tuples for each sentence in
		the dataset.
		
		Raises ConlluError if the file does not conform to the CoNLL-U format.
		"""
		sent = []
		
		try:
			with open(self.file_path, encoding='utf-8') as f:
				for line_num, line in enumerate(map(lambda x: x.strip(), f)):
					if line.startswith('#'):  # comments
						continue
					
					elif line:
						line = line.split('\t')
						assert len(line) == 10 and all([item for item in line])
						
						try:
							line[0] = int(line[0])
							line[6] = int(line[6])
						except ValueError:
							assert all([c.isdigit() or c in ('-', '.') for c in line[0]])
							continue
						
						sent.append(Word._make(line))
					
					else:  # empty lines mark sentence boundaries
						yield tuple(sent)
						sent = []
		
		except IOError:
			raise ConlluError('Could not open {}'.format(self.file_path))
		
		except AssertionError:
			raise ConlluError('Could not read {}:{}'.format(self.file_path, line_num))
	
	
	def gen_graphs(self):
		"""
		Generator that yields a nx.DiGraph instance for each sentence in the
		dataset.
		
		The graph nodes are the indices (0 being the root) of the respective
		words; FORM, LEMMA, UPOSTAG, and FEATS are stored as node attributes.
		The graph edges are directed from parent to child; DEPREL is stored as
		an edge attribute.
		
		Raises ConlluError if the file does not conform to the ConLL-U format.
		"""
		for sent in self.gen_sentences():
			graph = nx.DiGraph()
			graph.add_node(0, UPOSTAG='ROOT', FEATS='_', LEMMA='_')
			
			for word in sent:
				graph.add_node(word.ID,
					FORM=word.FORM, LEMMA=word.LEMMA,
					UPOSTAG=word.UPOSTAG, FEATS=word.FEATS)
				graph.add_edge(word.HEAD, word.ID, DEPREL=word.DEPREL)
			
			yield graph
