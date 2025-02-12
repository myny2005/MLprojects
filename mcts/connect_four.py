import numpy as np


class Board:
	def __init__(self):
		self.board = np.zeros((6, 7), dtype=int) # 0 - empty, 1 - player 1, 2 - player 2
	
	def is_valid_move(self, column):
		_, cols = self.board.shape
		if column < 0 or column >= cols:
			return False
		return self.board[0, column] == 0
		
	def get_valid_moves(self):
		_, cols = self.board.shape 
		return [c for c in range(cols) if self.is_valid_move(c)]
	
	def is_full(self):
		return len(self.get_valid_moves()) == 0

	def make_move(self, column, player):
		if not self.is_valid_move(column):
			return False  # invalid move
    
		row = np.where(self.board[:, column] == 0)[0][-1]  # Find the lowest empty row
		self.board[row, column] = player
		return True # possible move

	def check_winner(self):
		# 1 - player 1 won, 2 - player 2 won, 0 - nobody won
		
		rows, cols = self.board.shape
	
		# horizontal check
		for r in range(rows):
			for c in range(cols - 3):
				if self.board[r, c] != 0 and self.board[r, c] == self.board[r, c+1] == self.board[r, c+2] == self.board[r, c+3]:
					return self.board[r, c]
					
		# vertical check
		for r in range(rows - 3):
			for c in range(cols):
				if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c] == self.board[r+2, c] == self.board[r+3, c]:
					return self.board[r, c]

		# right diagonal check
		for r in range(rows - 3):
			for c in range(cols - 3):
				if self.board[r, c] != 0 and self.board[r, c] == self.board[r+1, c+1] == self.board[r+2, c+2] == self.board[r+3, c+3]:
					return self.board[r, c]

		# left diagonal check
		for r in range(3, rows):
			for c in range(cols - 3):
				if self.board[r, c] != 0 and self.board[r, c] == self.board[r-1, c+1] == self.board[r-2, c+2] == self.board[r-3, c+3]:
					return self.board[r, c]
		
		return 0
	
	def print_board(self):
		print(self.board)

