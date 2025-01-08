import json
import tkinter as tk

from utils import Node, construct_tree

CHAR_FILE = 'char_info.json'

root = tk.Tk()
root.title("Huffman speller")
root.configure(bg = 'black')
root.geometry('1000x300')

left_label = tk.Label(root, bg = 'black', fg = 'green', font = ('Arial', 20))
left_label.grid(row = 0, column = 0, ipadx = 50, ipady = 50)
right_label = tk.Label(root, bg = 'black', fg = 'red', font = ('Arial', 20))
right_label.grid(row = 0, column = 1, ipadx = 50, ipady = 50)

text_label = tk.Label(root, bg = 'black', fg = 'white', font = ('Arial', 50))
text_label.grid(row = 1, column = 0, columnspan = 2, ipadx = 50, ipady = 50)
text = ''

root.grid_rowconfigure(0, weight = 1)
root.grid_rowconfigure(1, weight = 1)
root.grid_columnconfigure(0, weight = 1)
root.grid_columnconfigure(1, weight = 1)

root_node = None
with open(CHAR_FILE, 'r') as char_file:
	char_info = json.load(char_file)
	root_node = construct_tree(char_info)
curr_node = root_node

def refresh():
	left_label.configure(text = curr_node.left.chars)
	right_label.configure(text = curr_node.right.chars)

def add_char():
	global text, curr_node
	text += curr_node.chars
	text_label.configure(text = text)
	curr_node = root_node

def move_left(event):
	global curr_node
	curr_node = curr_node.left
	if len(curr_node.chars) == 1:
		add_char()
	refresh()

def move_right(event):
	global curr_node
	curr_node = curr_node.right
	if len(curr_node.chars) == 1:
		add_char()
	refresh()

refresh()
root.bind('<Left>', move_left)
root.bind('<Right>', move_right)
root.mainloop()
