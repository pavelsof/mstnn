"""
Handles the command-line interface, including the help strings. Only the UI is
done here: as soon as it is clear what the user wants, the respective function
from code.main is invoked.
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
			'what do you want to achieve, stranger?'))
		
		self.subparsers = self.parser.add_subparsers(dest='command',
			title='subcommands')
		
		self._init_train()
		self._init_parse()
		self._init_diff()
		self._init_unittest()
	
	
	def _init_train(self):
		"""
		Inits the subparser that handles the train command. The latter expects
		a conllu data file to train on and a model file to save the output of
		the training into.
		"""
		def _train(args):
			from code.main import train
			train(args.model_file, args.conllu_file, ud_version=args.ud_version)
		
		description = 'train an mstnn model from conllu data'
		
		subp = self.subparsers.add_parser('train',
			description=description, help=description)
		
		subp.add_argument('model_file', help=(
			'path to a file to store the trained model in; '
			'if it exists, it will be overwritten'))
		subp.add_argument('conllu_file', help=(
			'path to the data to train on; '
			'assumed to be a unicode conllu file'))
		
		subp.add_argument('-u', '--ud-version', type=int, default=2, help=(
			'the UD version to use; either 1 or 2 (the default)'))
		
		subp.set_defaults(func=_train)
	
	
	def _init_parse(self):
		"""
		Inits the subparser that handles the parse command. The latter expects
		a previously trained model, a conllu dataset which to parse, and a path
		to write the output to.
		"""
		def _parse(args):
			from code.main import parse
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
		two conllu files and prints a (hopefully) useful report about the
		differences in the parses that have such (i.e. parses that are the same
		are skipped).
		"""
		def _diff(args):
			from code.diff import diff
			diff(args.file1, args.file2)
		
		description = 'print the differences between two conllu files'
		
		subp = self.subparsers.add_parser('diff',
			description=description, help=description)
		
		subp.add_argument('file1', help=('path to a conllu file'))
		subp.add_argument('file2', help=('path to another conllu file'))
		
		subp.set_defaults(func=_diff)
	
	
	def run(self, raw_args=None):
		"""
		Parses the given arguments (if these are None, then argparse's parser
		defaults to parsing sys.argv), calls the respective subcommand function
		with the parsed arguments, and then exits.
		"""
		args = self.parser.parse_args(raw_args)
		
		if args.command is None:
			return self.parser.format_help()
		
		args.func(args)
		
		self.parser.exit()
