import json

from utils import construct_tree

depths = {}
def get_depths(node, depth = 0):
	if len(node.chars) == 1:
		depths[node.chars] = depth
		return

	get_depths(node.left, depth + 1)
	get_depths(node.right, depth + 1)

CHAR_FILE = 'char_info.json'

root_node = None
with open(CHAR_FILE, 'r') as char_file:
	char_info = json.load(char_file)
	temp = {}
	for char, freq in char_info.items():
		if char.isalnum():
			char = char.upper()
			temp[char] = temp.get(char, 0) + freq
	sum_freqs = sum(temp.values())
	char_info = {char: freq / sum_freqs for char, freq in temp.items()}
	print(sorted(char_info.items(), key = lambda x: x[1], reverse = True))
	root_node = construct_tree(char_info)

get_depths(root_node)

average = 0
for char, freq in char_info.items():
	average += depths[char] * freq
print(average)
