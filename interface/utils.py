from queue import PriorityQueue

class Node:
	def __init__(self, chars, left = None, right = None):
		self.left = left
		self.right = right
		self.chars = chars
	
	def __lt__(self, other):
		return self.chars < other.chars

def construct_tree(char_info):
	pq = PriorityQueue()
	for char, freq in char_info.items():
		pq.put((freq, Node(char)))

	for _ in range(len(char_info) - 1):
		freq_left, node_left = pq.get()
		freq_right, node_right = pq.get()
		pq.put((
			freq_left + freq_right,
			Node(node_left.chars + node_right.chars, node_left, node_right)
		))
	
	return pq.get()[1]
