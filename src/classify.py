import argparse
import pprint

import data_loader

from models.bert import Bert
import tester
import model_util as mu

import torch, gc
import torch.nn as nn
from torch import optim

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from transformers import BertTokenizer, TextClassificationPipeline

def define_argparser():
	p = argparse.ArgumentParser()

	p.add_argument(
		'--research_subject',
		required=True,
		help='The name of the research subject. (ex: server1)',
	)

	p.add_argument(
		'--research_num',
		required=True,
		help='The number of current test for a subject experiment. (ex: 01)',
	)

	p.add_argument(
		'--sentence',
		required=True,
		help='Sentence to classify as negative or non-negative.',
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
	optimizer = optim.Adam(model.parameters(), lr=config.lr)
	return optimizer
	


def main(config):
	
	# ********** PRINT CONFIG HELP **********
	def print_config(config):
		pp = pprint.PrettyPrinter(indent=4)
		pp.pprint(vars(config))
	print_config(config)



	# ********** BRING MODEL **********
	subject_title = config.research_subject
	title = subject_title + '_' + config.research_num

	model = mu.getModel(subject_title, title)


	# ********** BRING TOKENIZER **********
	tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")



	# ********** BRING TEXT-CLASSIFIER **********
	classifier = TextClassificationPipeline(
					tokenizer=tokenizer,
					model=model,
					framework='pt',
					return_all_scores=True,
				)


	# ********** CLASSIFY SENTENCE **********
	result = text_classifier(config.sentence)[0]
	print('sentence: ',config.sentence)
	print(result)




if __name__ == '__main__':
	config = define_argparser()
	main(config)
