import numpy as np
from connect_four import Board
from mcts import Mcts, Node


def play_game():
	game = Board()
	mcts = Mcts(exploration_weight = 1.41)
	
	print("\n Welcome to Connect Four with AI!")
	player = int(input("Choose your side: 1 - Player one, 2 - Player two: "))
	
	while True:
		game.print_board()  # Display the current board state
		if game.is_full():
			print("\n It's a draw! The board is full.")
			break

		current_player = 1 if np.count_nonzero(game.board) % 2 == 0 else 2

		if current_player == player:  # Player's move
			while True:
				try:
					move = int(input("\n Choose a column (0-6): "))
					if game.is_valid_move(move):
						break
					print(" This move is not allowed! Try again.")
				except ValueError:
					print(" Enter a valid number (0-6).")

			game.make_move(move, player)

		else:  # AI's move
			print("\n AI is thinking...")
			root = Node(game)
			best_move = mcts.search(root, simulations = 20000)  # AI is "thinking" haha
			game.make_move(best_move.move, current_player)
			print(f"AI chooses column: {best_move.move}")

		# Check if there is a winner
		winner = game.check_winner()
		if winner != 0:
			game.print_board()
			print(f"\n Player {winner} wins! 🏆")
			break

if __name__ == "__main__":
	play_game()

