import argparse
import pprint

import data_loader

from models.bert import Bert

def define_argparser():
	p = argparse.ArgumentParser()

	p.add_argument(
		'--batch_size',
		type=int,
		default=16,
		help='Mini batch size for gradient descent. Default=%(default)s',
	)

	p.add_argument(
		'--max_length',
		type=int,
		default=512,
		help='Maximum length of the training sequence. Default=%(default)s',
	)

	p.add_argument(
		'--dropout',
		type=float,
		default=0.0,
		help='Dropout rate. Default=%(default)s',
	)

	p.add_argument(
		'--lr',
		type=float,
		default=0.0001,
		help='Initial learning rate. Default=%(default)s',
	)

	p.add_argument(
		'--n_epochs',
		type=int,
		default=20,
		help='Number of epochs to train. Default%(default)s',
	)

	config = p.parse_args()

	return config

def get_model(config):
	model = Bert(dropout=config.dropout)
	return model

def get_crit():
	crit = nn.CrossEntropyLoss()
	return crit

def get_optimizer(model, config):
	optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=(.9, .98))
	return optimizer
	


def main(config):
	
	# ********** PRINT CONFIG HELP **********
	def print_config(config):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(config))
	print_config(config)


	# ********** LOAD DATA **********
	paths = ['../data/train.negative.csv',
				'../data/train.non-negative.csv',
				'../data/test.negative.csv',
				'../data/test.non-negative.csv']

	train, valid, test = data_loader.data_loader(paths,
												batch_size=config.batch_size,
												max_length=config.max_length)

if __name__ == '__main__':
	config = define_argparser()
	main(config)
