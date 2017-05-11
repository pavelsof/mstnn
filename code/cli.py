"""
Handles the command-line interface, including the help strings. Only the UI is
done here: as soon as it is clear what the user wants, the respective function
from the rest of the codebase is invoked.
"""

import argparse



class Cli:
	"""
	Singleton that handles the user input, inits the whole machinery, and takes
	care of exiting the programme.
	"""
	
	def __init__(self):
		"""
		Constructor. Inits the argparse parser and then all the subparsers
		through the _init_* methods.
		
		Each of the latter defines a function which takes the argparse args as
		arguments and which will be called if the respective command is called.
		"""
		self.parser = argparse.ArgumentParser(description=(
			'the parse is darc and full of errors'))
		
		self.subparsers = self.parser.add_subparsers(dest='command',
			title='subcommands')
		
		self._init_train()
		self._init_parse()
		self._init_diff()
		self._init_score()
		self._init_unittest()
	
	
	def _init_train(self):
		"""
		Inits the subparser that handles the train command. The latter is used
		to train one or more mstnn models on a given training conllu dataset.
		"""
		def _train(args):
			from code.train import train
			train(args.model_file, args.train_file, args.ud_version,
					args.ignore_forms, args.ignore_lemmas, args.ignore_morph,
					args.epochs, args.batch_size,
					args.dev_file, args.keep,
					args.forms_word2vec, args.lemmas_word2vec)
		
		description = 'train an mstnn model from conllu data'
		
		subp = self.subparsers.add_parser('train',
			description=description, help=description)
		
		subp.add_argument('model_file', help=(
			'path to a file to store the trained model in; '
			'if it exists, it will be overwritten'))
		subp.add_argument('train_file', help=(
			'path to the dataset to train on; '
			'assumed to be a unicode conllu file'))
		
		subp.add_argument('-e', '--epochs', type=int, default=10, help=(
			'number of training epochs; defaults to 10'))
		subp.add_argument('-b', '--batch-size', type=int, default=32, help=(
			'training batch size; defaults to 32'))
		subp.add_argument('-d', '--dev-file', help=(
			'path to a development dataset to fine-tune against; '
			'assumed to be a unicode conllu file'))
		subp.add_argument('-k', '--keep', type=int, default=1, help=(
			'number of best performing epoch checkpoints to keep; '
			'only relevant if a development dataset is set'))
		subp.add_argument('-u', '--ud-version', type=int, default=2, help=(
			'the UD version to use; either 1 or 2 (the default); '
			'also applied to the development dataset, if such'))
		
		subp.add_argument('--ignore-forms', action='store_true', help=(
			'do not use word forms in the model'))
		subp.add_argument('--ignore-lemmas', action='store_true', help=(
			'do not use lemmas in the model'))
		subp.add_argument('--ignore-morph', action='store_true', help=(
			'do not use morphological features in the model'))
		
		subp.add_argument('-f', '--forms-word2vec', help=(
			'path to a word2vec binary file that contains '
			'pre-trained word form embeddings'))
		subp.add_argument('-l', '--lemmas-word2vec', help=(
			'path to a word2vec binary file that contains '
			'pre-trained lemmas embeddings'))
		
		subp.set_defaults(func=_train)
	
	
	def _init_parse(self):
		"""
		Inits the subparser that handles the parse command. The latter is used
		to parse a conllu dataset using a previously trained mstnn model.
		"""
		def _parse(args):
			from code.model import parse
			parse(args.model_file, args.conllu_file, args.output_file)
		
		description = 'parse conllu data using an mstnn model'
		
		subp = self.subparsers.add_parser('parse',
			description=description, help=description)
		
		subp.add_argument('model_file', help=(
			'path to an mstnn model'))
		subp.add_argument('conllu_file', help=(
			'path to the data to parse; '
			'assumed to be a unicode conllu file'))
		subp.add_argument('output_file', help=(
			'path where to write the parsed sentences (in conllu format)'))
		
		subp.set_defaults(func=_parse)
	
	
	def _init_unittest(self):
		"""
		Inits the subparser that handles the unittest command. The latter can
		run all the available unit tests or a specific test suite.
		"""
		def unit_test(args):
			from unittest import TestLoader, TextTestRunner
			
			loader = TestLoader()
			
			if args.module:
				suite = loader.loadTestsFromName(args.module)
			else:
				suite = loader.discover('code/tests')
			
			runner = TextTestRunner()
			runner.run(suite)
		
		
		usage = 'manage.py unittest [module]'
		description = 'run unit tests'
		
		subp = self.subparsers.add_parser('unittest', usage=usage,
			description=description, help=description)
		subp.add_argument('module', nargs='?', help=(
			'dotted name of the module to test; '
			'if omitted, run all tests'))
		subp.set_defaults(func=unit_test)
	
	
	def _init_diff(self):
		"""
		Inits the subparser that handles the diff command. The latter expects
		two conllu datasets and prints a (hopefully) useful report about the
		differences in the parses that have such (i.e. parses that are the same
		are skipped).
		"""
		def _diff(args):
			from code.diff import diff
			out = diff(args.file1, args.file2, ud_version=args.ud_version)
			if out: print(out)
		
		description = 'print the differences between two conllu files'
		
		subp = self.subparsers.add_parser('diff',
			description=description, help=description)
		
		subp.add_argument('file1', help=('path to a conllu file'))
		subp.add_argument('file2', help=('path to another conllu file'))
		
		subp.add_argument('-u', '--ud-version', type=int, default=2, help=(
			'the UD version to use; either 1 or 2 (the default)'))
		
		subp.set_defaults(func=_diff)
	
	
	def _init_score(self):
		"""
		Inits the subparser that handles the score command. The latter expects
		two conllu datasets and prints the unlabelled attachment score of the
		first against the second.
		"""
		def _score(args):
			from code.score import score
			uas = score(args.parser_output, args.gold_standard, args.ud_version)
			print('{:.2f}'.format(uas))
		
		description = 'calculate UAS'
		
		subp = self.subparsers.add_parser('score',
			description=description, help=description)
		
		subp.add_argument('parser_output', help=('path to a conllu file'))
		subp.add_argument('gold_standard', help=('path to another conllu file'))
		
		subp.add_argument('-u', '--ud-version', type=int, default=2, help=(
			'the UD version to use; either 1 or 2 (the default)'))
		
		subp.set_defaults(func=_score)
	
	
	def run(self, raw_args=None):
		"""
		Parses the given arguments (if these are None, then argparse's parser
		defaults to parsing sys.argv), calls the respective subcommand function
		with the parsed arguments, and then exits.
		"""
		args = self.parser.parse_args(raw_args)
		
		if args.command is None:
			return self.parser.format_help()
		
		try:
			args.func(args)
		except Exception as err:
			print(str(err))
		
		self.parser.exit()
