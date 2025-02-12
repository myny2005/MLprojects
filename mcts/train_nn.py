import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from connect_four import Board
from nn_model import ConnectFourNN


def board_to_tensor(board):
	# Converts the board to a tensor with dimensions (1, 2, 6, 7)
	board_tensor = np.zeros((2, 6, 7), dtype=np.float32)
	board_tensor[0] = (board.board == 1)  # Channel for player 1
	board_tensor[1] = (board.board == 2)  # Channel for player 2
	return torch.tensor(board_tensor).unsqueeze(0)  # Add batch dimension

def self_play_games(num_games):
	model = ConnectFourNN()
	model.eval() # Evaluation mode, we are not training in this phase
	training_data = []

	for _ in range(num_games):
		game = Board()
		history = []

		while not game.is_full():
			state = board_to_tensor(game)

			# Predictions from the neural network
			policy, value = model(state)
			policy_probs = torch.softmax(policy, dim=-1).detach().numpy().flatten()

			# Filter only valid moves
			valid_moves = game.get_valid_moves()
			filtered_policy = np.zeros(7)
			for move in valid_moves:
				filtered_policy[move] = policy_probs[move]
			filtered_policy /= np.sum(filtered_policy)  # Normalize to sum to 1

			# Move selection - 80% based on policy network, 20% exploration   (this can change)  20 percent hyperparameter (random but seems reasonable haha)
			if random.random() > 0.2:
				move = np.random.choice(7, p=filtered_policy)
			else:
				move = random.choice(valid_moves)

			game.make_move(move, 1 if np.count_nonzero(game.board) % 2 == 0 else 2)

			# Save the history
			history.append((state, filtered_policy, value, move))

			# Check for a winner
			winner = game.check_winner()
			if winner != 0 or game.is_full():
				reward = 1 if winner == 1 else (-1 if winner == 2 else 0)

				# Assign reward to each position in the game
				for s, p, v, m in history:
					training_data.append((s, torch.tensor(p, dtype=torch.float32), torch.tensor([reward], dtype=torch.float32), torch.tensor([m], dtype=torch.long)))
				break  # End the game after finding a winner

	return training_data

def train_model(training_data, epochs=20, batch_size=32):
	model = ConnectFourNN()
	optimizer = optim.Adam(model.parameters(), lr=0.001)
	scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)  # Reduce LR every 5 epochs
	criterion_value = nn.MSELoss()  # MSE for position value
	criterion_policy = nn.CrossEntropyLoss()  # CrossEntropy for policy

	for epoch in range(epochs):
		total_loss = 0
		random.shuffle(training_data)  # Shuffle data before each epoch
		batches = [training_data[i:i + batch_size] for i in range(0, len(training_data), batch_size)]

		for batch in batches:
			states, policy_targets, value_targets, moves = zip(*batch)

			states = torch.cat(states)  # Concatenate all states into one tensor
			policy_targets = torch.stack(policy_targets)  # Policy for all moves
			value_targets = torch.cat(value_targets)  # Position values
			moves = torch.cat(moves).squeeze()  # Move indices

			optimizer.zero_grad()
			predicted_policy, predicted_value = model(states)

			# Policy loss - CrossEntropy works on indices
			loss_policy = criterion_policy(predicted_policy, moves)
			# Value loss - MSE for value head
			loss_value = criterion_value(predicted_value.squeeze(), value_targets)

			loss = loss_policy + loss_value  # Total loss
			loss.backward()
			optimizer.step()

			total_loss += loss.item()

		scheduler.step()
		print(f"Epoch {epoch + 1}/{epochs} - Loss: {total_loss:.4f}")

	torch.save(model.state_dict(), "connect_four_nn.pth")
	print("Model saved as `connect_four_nn.pth`!")

if __name__ == "__main__":
	print("AI is playing against itself to generate data...")
	training_data = self_play_games(num_games = 4000)

	print(f"Collected {len(training_data)} training examples!")

	print("Starting to train the neural network...")
	train_model(training_data, epochs = 25)

	print("Training finished! AI is ready to play!")

