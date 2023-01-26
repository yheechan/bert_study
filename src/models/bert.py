from torch import nn
from transformers import BertModel

class Bert(nn.Module):
	def __init__(self, dropout=0.0):
		super(Bert, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(dropout)
		self.fc1 = nn.Linear(768, 512)
		self.relu1 = nn.ReLU()
		self.fc2 = nn.Linear(512, 2)
		self.relu2 = nn.ReLU()
	
	def forward(self, input_id, mask):
		_, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		dp = self.dropout(pooled_output)
		fc1 = self.relu1(self.fc1(dp))
		y_hat = self.relu2(self.fc2(fc1))

		return y_hat
