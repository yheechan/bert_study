from torch import nn
from transformers import BertModel

class Bert(nn.Module):
	def __init__(self, dropout=0.5):
		self.bert = BertModel.from_pretrained('bert-base-uncased')
		self.dropout = nn.Dropout(dropout)
		self.fc = nn.Linear(768, 2)
		self.relu = nn.ReLU()
	
	def forward(self, input_id, mask):
		_, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		dp = self.dropout(pool_output)
		fc = self.fc(dp)
		y_hat = self.relu(fc)

		return y_hat
