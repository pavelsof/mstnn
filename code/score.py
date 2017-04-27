from code.conllu import Dataset



class ScoreError(ValueError):
	"""
	Raised when there is a problem with calculating the UAS score. Should be
	inited with a human-friendly message.
	"""
	pass



def score(parsed, standard):
	"""
	Calculates and returns the UAS score of a dataset against another dataset,
	usually parser output against gold-standard data. Expects the paths to two
	conllu datasets.
	
	Could raise a ConlluError or a ScoreError.
	"""
	if type(parsed) is str:
		parsed = [sent for sent in Dataset(parsed).gen_sentences()]
	
	if type(standard) is str:
		standard = [sent for sent in Dataset(standard).gen_sentences()]
	
	if len(parsed) != len(standard):
		raise ScoreError('The number of sentences differ')
	
	arcs_umatch = 0
	arcs_total = 0
	
	for i in range(len(parsed)):
		sent_par = parsed[i]
		sent_std = standard[i]
		
		if len(sent_par) != len(sent_std):
			raise ScoreError('The number of words differ in sentence {}'.format(i))
		
		correct = 0
		
		for j in range(len(sent_par)):
			if sent_par[j].HEAD == sent_std[j].HEAD:
				correct += 1
		
		arcs_umatch += correct
		arcs_total += len(sent_par)
	
	return 100 * arcs_umatch / arcs_total
