"""
Handles reading and writing datasets in the CoNLL-U format (version 2) [0]:
	
	dataset = conllu.Dataset(path)
	graphs = [graph for graph in dataset.gen_graphs()]
	
	another = conllu.Dataset(path)
	another.write_graphs(graphs)

The graphs in question are nx.DiGraph instances, one for each sentence.

[0] http://universaldependencies.org/format.html
"""

from collections import namedtuple

import networkx as nx



"""
Represents a word, i.e. a row with an integer ID (excluding multiword tokens
and empty nodes). With the exception of ID and HEAD which are integers, the
other fields are non-empty strings.
"""
Word = namedtuple('Word', [
	'ID', 'FORM', 'LEMMA',
	'UPOSTAG', 'XPOSTAG', 'FEATS',
	'HEAD', 'DEPREL', 'DEPS', 'MISC'])



class ConlluError(ValueError):
	"""
	Raised when there is a problem with reading or writing a file in CoNLL-U
	format. Should be inited with a human-friendly error message.
	"""
	pass



class Dataset:
	
	def __init__(self, file_path):
		"""
		Constructor. Expects the path to the .conllu dataset file. The latter
		is not opened until one of the gen_* methods is invoked.
		"""
		self.file_path = file_path
	
	
	"""
	Reading ConLL-U
	"""
	
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
		
		Unlike self.gen_sentences, the output here has an explicit root node
		with its own POS tag (ROOT), lemma and form (__root__) which are not
		part of the UD standard, but still needed for the purposes of mstnn.
		
		Raises ConlluError if the file does not conform to the ConLL-U format.
		"""
		for sent in self.gen_sentences():
			graph = nx.DiGraph()
			
			graph.add_node(0,
				FORM='__root__', LEMMA='__root__',
				UPOSTAG='ROOT', FEATS='_')
			
			for word in sent:
				graph.add_node(word.ID,
					FORM=word.FORM, LEMMA=word.LEMMA,
					UPOSTAG=word.UPOSTAG, FEATS=word.FEATS)
				graph.add_edge(word.HEAD, word.ID, DEPREL=word.DEPREL)
			
			yield graph
	
	
	"""
	Writing ConLL-U
	"""
	
	@staticmethod
	def format_sentence(sentence):
		"""
		Returns a multi-line string representing a parsed sentence in ConLL-U
		format (including the empty line at the end). Expects a sequence of
		well-formated (i.e. no empty-string fields) Word named tuples.
		
		This method is static and is used as a helper for self.write_sentences
		(and, hence, for self.write_graphs as well).
		"""
		lines = ['\t'.join(map(str, list(word))) for word in sentence]
		return '\n'.join(lines + ['', ''])
	
	
	def write_sentences(self, sentences):
		"""
		Writes the sentences to the dataset's file path. The sentences are
		expected to be a sequence of sequences of Word named tuples.
		
		Raises ConlluError if the file cannot be written or if the sentences do
		not conform to the spec.
		"""
		try:
			with open(self.file_path, 'w', encoding='utf-8') as f:
				for sentence in sentences:
					assert all([isinstance(word, Word) for word in sentence])
					f.write(self.format_sentence(sentence))
		
		except IOError:
			raise ConlluError('Could not write {}'.format(self.file_path))
		
		except AssertionError:
			raise ConlluError('Sentences must be sequences of Word tuples')
	
	
	def write_graphs(self, graphs):
		"""
		Writes the given sentence graphs to the dataset's file path. The graphs
		should be a sequence of nx.DiGraph instances, each conforming to the
		spec outlined in self.gen_graphs.
		
		Raises ConlluError if the file cannot be written or if the graphs do not
		conform to the spec.
		"""
		sentences = []
		
		for graph in graphs:
			sentence = []
			
			for key, data in graph.nodes(data=True):
				if key == 0:
					continue  # skip the ROOT
				
				try:
					edge = graph.in_edges(key, data=True)[0]
					assert 'FORM' in data
					assert 'LEMMA' in data
					assert 'UPOSTAG' in data
				except (StopIteration, AssertionError):
					raise ConlluError('Graphs do not conform to the spec')
				
				sentence.append(Word(key,
					data['FORM'],
					data['LEMMA'],
					data['UPOSTAG'],
					data['XPOSTAG'] if 'XPOSTAG' in data else '_',
					data['FEATS'] if 'FEATS' in data else '_',
					edge[0],
					edge[2]['DEPREL'] if 'DEPREL' in edge[2] else '_',
					data['DEPS'] if 'DEPS' in data else '_',
					data['MISC'] if 'MISC' in data else '_'))
			
			sentences.append(sentence)
		
		self.write_sentences(sentences)
