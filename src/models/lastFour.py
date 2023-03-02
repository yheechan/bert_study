import torch
from torch import nn
from transformers import BertModel

class Classification(nn.Module):
	def __init__(self, dropout=0.0):
		super(Classification, self).__init__()

		# self.fc1 = nn.Linear(768, 2)
		self.fc1 = nn.Linear(3072, 2)
		# self.tanh1 = nn.Tanh()
		# self.dp1 = nn.Dropout(dropout)

		'''
		self.fc2 = nn.Linear(512, 2)
		self.relu2 = nn.ReLU()
		self.dp2 = nn.Dropout(dropout)
		'''

		'''
		self.fc3 = nn.Linear(512, 2)
		self.relu3 = nn.ReLU()
		self.dp3 = nn.Dropout(dropout)
		'''
	
	def forward(self, input_vector):
		
		# y_hat = self.dp1(self.relu1(self.fc1(input_vector)))
		y_hat = self.fc1(input_vector)
		# y_hat = self.dp2(self.relu2(self.fc2(fc1)))
		# y_hat = self.dp3(self.relu3(self.fc3(fc2)))

		return y_hat


class Bert(nn.Module):
	def __init__(self, dropout=0.0):
		super(Bert, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased', return_dict=True,
												output_hidden_states=True)

		self.dp = nn.Dropout(dropout)

		self.lstm = nn.LSTM(
			input_size = 3072,
			hidden_size = 768,
			num_layers = 2,
			dropout = dropout,
			batch_first = True,
			bidirectional = True,
		)

		self.classification = Classification(dropout=dropout)

	
	def forward(self, input_id, mask):
		# _, cls_vector = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		last_hidden_state, pooler_output, hidden_states = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		# |pooler_output| = (batch_size, 768) -> output of [CLS] token

		last_four_encoders = hidden_states[-4:]
		catenated_four = torch.cat(last_four_encoders, dim=2)

		output, (hidden, cell) = self.lstm(catenated_four)

		x = torch.cat([hidden[0], hidden[1], hidden[2], hidden[3]], dim=1)

		dp = self.dp(x)

		y_hat = self.classification(dp)

		return y_hat
