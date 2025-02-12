import numpy as np
import torch
from connect_four import Board
from mcts_nn import Mcts, Node
from nn_model import ConnectFourNN


def board_to_tensor(board):
	board_tensor = np.zeros((2, 6, 7), dtype=np.float32)
	board_tensor[0] = (board.board == 1)  # Channel for player 1
	board_tensor[1] = (board.board == 2)  # Channel for player 2
	return torch.tensor(board_tensor).unsqueeze(0)  # Add batch dimension

def play_game():
	game = Board()
	mcts = Mcts(exploration_weight = 1.41)

	# Load the model
	model = ConnectFourNN()
	model.load_state_dict(torch.load("connect_four_nn.pth"))
	model.eval()

	print("\nWelcome to Connect Four with AI!")
	while True:
		try:
			player = int(input("Choose your side: 1 - Player 1, 2 - Player 2: "))
			if player in [1, 2]:
				break
			print("Invalid choice! Enter 1 or 2.")
		except ValueError:
			print("Invalid input! Enter 1 or 2.")

	while True:
		game.print_board()

		if game.is_full():
			print("\nDraw! The board is full.")
			break

		current_player = 1 if np.count_nonzero(game.board) % 2 == 0 else 2

		if current_player == player:
			# Player's move
			while True:
				try:
					move = int(input("\nChoose a column (0-6): "))
					if move in game.get_valid_moves():
						break
					print("Invalid move! Column is full or out of range.")
				except ValueError:
					print("Enter a valid number (0-6).")

			game.make_move(move, player)

		else:
			# AI's move
			print("\nAI is thinking...")

			# Convert the board to tensor
			state = board_to_tensor(game)

			# Get the predicted moves from the neural network
			policy, _ = model(state)
			move_probabilities = policy.cpu().detach().numpy().flatten()

			# Filter valid moves only
			valid_moves = game.get_valid_moves()
			filtered_policy = np.zeros(7)
			for move in valid_moves:
				filtered_policy[move] = move_probabilities[move]
			filtered_policy /= np.sum(filtered_policy)  # Normalization

			# AI chooses a move based on the policy network
			move = np.random.choice(7, p=filtered_policy)

			game.make_move(move, current_player)
			print(f"AI chooses column: {move}")

		# Check for a winner
		winner = game.check_winner()
		if winner != 0:
			game.print_board()
			print(f"\nPlayer {winner} wins!")
			break

if __name__ == "__main__":
	play_game()

