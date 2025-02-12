import torch
import numpy as np
from nn_model import ConnectFourNN
from connect_four import Board


# Load trained model
model = ConnectFourNN()
model.load_state_dict(torch.load("connect_four_nn.pth"))
model.eval()

# Create an empty board
game = Board()
state = torch.tensor(game.board, dtype=torch.float32).view(1, 6, 7)

# predictions
policy, value = model(state)

print(f"Policy (Move Probabilities): {policy.detach().numpy().flatten()}")
print(f"Value (Position Evaluation): {value.item()}")

