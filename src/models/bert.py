from torch import nn
from transformers import BertModel

class Bert(nn.Module):
	def __init__(self, dropout=0.0):
		super(Bert, self).__init__()

		self.bert = BertModel.from_pretrained('bert-base-uncased')
		# self.dp = nn.Dropout(dropout)

		self.fc1 = nn.Linear(768, 512)
		self.relu1 = nn.ReLU()
		self.dp1 = nn.Dropout(dropout)

		self.fc2 = nn.Linear(512, 512)
		self.relu2 = nn.ReLU()
		self.dp2 = nn.Dropout(dropout)

		self.fc3 = nn.Linear(512, 2)
		self.relu3 = nn.ReLU()
		self.dp3 = nn.Dropout(dropout)

	
	def forward(self, input_id, mask):
		_, pooled_output = self.bert(input_ids=input_id, attention_mask=mask, return_dict=False)
		# |pooled_output| = (batch_size, 728) -> output of [CLS] token
		# dp = self.dp(pooled_output)

		fc1 = self.dp1(self.relu1(self.fc1(pooled_output)))
		fc2 = self.dp2(self.relu2(self.fc2(fc1)))
		y_hat = self.dp3(self.relu3(self.fc3(fc2)))

		return y_hat
