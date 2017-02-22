import argparse

from code.main import train, test



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
		self._init_test()
		self._init_unittest()
	
	
	def _init_train(self):
		"""
		Inits the subparser that handles the train command. The latter expects
		a conllu data file to train on and a model file to save the output of
		the training into.
		"""
		def _train(args):
			train(args.model_file, args.conllu_file, ud_version=args.ud_version)
		
		description = 'train an mstnn instance from conllu data'
		
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
	
	
	def _init_test(self):
		"""
		Inits the subparser that handles the test command. The latter expects a
		previously trained model and a conllu data file against which to use
		the model. Writes the performance stats to stdout.
		"""
		def _test(args):
			test(args.model_file, args.conllu_file)
		
		usage = 'manage.py test model_file conllu_file'
		description = 'test an mstnn model against conllu data'
		
		subp = self.subparsers.add_parser('test', usage=usage,
			description=description, help=description)
		
		subp.add_argument('model_file', help=(
			'path to an mstnn model'))
		subp.add_argument('conllu_file', help=(
			'path to the test data; '
			'assumed to be a unicode conllu file'))
		
		subp.set_defaults(func=_test)
	
	
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
			self.parser.error(str(err))
		
		self.parser.exit()
