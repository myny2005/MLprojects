import torch
import torch.nn as nn
import torch.nn.functional as F


class ConnectFourNN(nn.Module):
	def __init__(self):
		super(ConnectFourNN, self).__init__()

		# Convolutional layers to extract spatial features from the board
		self.conv1 = nn.Conv2d(2, 64, kernel_size=3, padding=1)  # Input has 2 channels (player 1 & player 2)
		self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
		self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

		# Batch Normalization for stable training and better generalization
		self.bn1 = nn.BatchNorm2d(64)
		self.bn2 = nn.BatchNorm2d(64)
		self.bn3 = nn.BatchNorm2d(64)

		# Policy Head: Determines the best move
		self.policy_conv = nn.Conv2d(64, 2, kernel_size=1)  # 1x1 convolution to refine features
		self.policy_fc = nn.Linear(2 * 6 * 7, 7)  # Fully connected layer (one output per column)
		self.dropout = nn.Dropout(0.2)  # Dropout to reduce overfitting

		# Value Head: Predicts the expected outcome of the game
		self.value_conv = nn.Conv2d(64, 1, kernel_size=1)  # 1x1 convolution to reduce dimensions
		self.value_fc1 = nn.Linear(6 * 7, 32)  # Smaller fully connected layer
		self.value_fc2 = nn.Linear(32, 1)  # Final output: estimated game outcome

	def forward(self, x):
		x = x.view(-1, 2, 6, 7) # (batch size, 2, 6, 7)

		# Pass through three convolutional layers with batch normalization and ReLU activation
		x = F.relu(self.bn1(self.conv1(x)))
		x = F.relu(self.bn2(self.conv2(x)))
		x = F.relu(self.bn3(self.conv3(x)))

		# Policy Head: Predicts the probability distribution over possible moves
		p = F.leaky_relu(self.policy_conv(x), negative_slope=0.01)  # LeakyReLU prevents dead neurons
		p = p.view(-1, 2 * 6 * 7)  # Flatten the output before the fully connected layer
		p = self.dropout(p)  # Apply dropout for regularization
		p = F.softmax(self.policy_fc(p), dim=-1)  # Softmax to get move probabilities

		# Value Head: Predicts how good the current board position is
		v = F.relu(self.value_conv(x))  # Reduce dimensions
		v = v.view(-1, 6 * 7)  # Flatten
		v = F.relu(self.value_fc1(v))  # First fully connected layer
		v = torch.tanh(self.value_fc2(v))  # Output a value between -1 (loss) and 1 (win)

		return p, v

