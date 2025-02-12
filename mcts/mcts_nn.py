from nn_model import ConnectFourNN
import torch
import numpy as np
import random
import math
from connect_four import Board


class Node:
	def __init__(self, board, parent=None, move=None):
		self.board = board  # copy of the board
		self.parent = parent
		self.move = move  # Move which led to this node
		self.children = []  # List of children
		self.visits = 0  # How many times this node was visited
		self.wins = 0  # How many wins in this node?

	def is_fully_expanded(self):
		return len(self.children) == len(self.get_valid_moves())

	def get_valid_moves(self):
		return self.board.get_valid_moves()

	def best_child(self, exploration_weight = 1.41, policy_probs=None):
		# Selects the best child based on UCT and the policy network
		if not self.children:
			return None

		def uct_value(node, policy_prob):
			if node.visits == 0:
				return float("inf")  # Priority for unvisited moves
			exploitation = node.wins / (node.visits + 1e-6)
			exploration = exploration_weight * math.sqrt(math.log(self.visits + 1) / (node.visits + 1e-6))
			return exploitation + exploration + policy_prob

		# Get policy network or equal division
		if policy_probs is None:
			policy_probs = {child.move: 1 / len(self.children) for child in self.children}
		
		return max(self.children, key=lambda child: uct_value(child, policy_probs.get(child.move, 0)))

class Mcts:
	def __init__(self, exploration_weight = 1.41):
		self.exploration_weight = exploration_weight
		self.model = ConnectFourNN()
		self.model.load_state_dict(torch.load("connect_four_nn.pth"))
		self.model.eval()

	def board_to_tensor(self, board):
		#Converts the board to a 2-channel format (player1, player2).
		board_tensor = np.zeros((2, 6, 7), dtype=np.float32)
		board_tensor[0] = (board.board == 1)  # Channel for player 1
		board_tensor[1] = (board.board == 2)  # Channel for player 2
		return torch.tensor(board_tensor).unsqueeze(0)  # Add batch dimension

	def search(self, root, simulations = 1000):
		for _ in range(simulations):
			node = self.select(root)
			if not node.board.is_full():
				node = self.expand(node)
			winner = self.simulate(node)
			self.backpropagate(node, winner)

		# Get policy network
		board_tensor = self.board_to_tensor(root.board)
		policy, _ = self.model(board_tensor)
		policy_probs = {i: policy[0, i].item() for i in root.get_valid_moves()}

		return root.best_child(exploration_weight = 0.1, policy_probs = policy_probs)

	def select(self, node):
		while not node.board.is_full() and node.is_fully_expanded():
			node = node.best_child(self.exploration_weight)
		return node

	def expand(self, node):
		# Adds all possible children, prioritizing the policy network.
		moves = node.get_valid_moves()
		board_tensor = self.board_to_tensor(node.board)
		policy, _ = self.model(board_tensor)
		move_probabilities = policy.detach().numpy().flatten()
		valid_probs = {move: move_probabilities[move] for move in moves}

		# Create children for all possible moves
		for move in sorted(valid_probs, key=valid_probs.get, reverse=True):  # Priority for better moves
			new_board = Board()
			new_board.board = np.copy(node.board.board)
			new_board.make_move(move, 1 if np.count_nonzero(new_board.board) % 2 == 0 else 2)
			child = Node(new_board, parent = node, move =  move)
			node.children.append(child)
		
		# Select the first child as the expanded node
		return node.children[0]

	def simulate(self, node):
		winner = node.board.check_winner()
		if winner != 0:
			return 1 if winner == 1 else -1  # If there is a real winner, return the result

		board_tensor = self.board_to_tensor(node.board)
		_, value = self.model(board_tensor)
		return value.item()  # The evaluation value of the position from the value network

	def backpropagate(self, node, winner):
		while node is not None:
			node.visits += 1
			node.wins += winner  # Add the result from the value network
			node = node.parent

