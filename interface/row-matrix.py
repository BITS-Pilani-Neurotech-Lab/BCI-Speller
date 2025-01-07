import tkinter as tk
from time import sleep
from random import randrange

root = tk.Tk()
root.title("P300 Row-matrix speller")
# root.geometry('600x600')

matrix = [
	'ABCDEF',
	'GHIJKL',
	'MNOPQR',
	'STUVWX',
	'YZ0123',
	'456789'
]

ROWS = len(matrix)
COLS = len(matrix[0])

curr_row = None
curr_col = None

labels = [[None] * COLS for row in range(ROWS)]
for row in range(ROWS):
	for col in range(COLS):
		labels[row][col] = tk.Label(root,text = matrix[row][col], bg = 'black', fg = 'white', font = ('Arial', 20))
		labels[row][col].grid(row = row, column = col, sticky = tk.NSEW, ipadx = 50, ipady = 50)

def hightlight_row(row):
	global curr_row
	if curr_row is not None:
		for col in range(COLS):
			labels[curr_row][col].configure(fg = 'white')
	if row is not None: 
		for col in range(COLS):
			labels[row][col].configure(fg = 'red')
	curr_row = row

def hightlight_col(col):
	global curr_col
	if curr_col is not None:
		for row in range(ROWS):
			labels[row][curr_col].configure(fg = 'white')
	if col is not None: 
		for row in range(ROWS):
			labels[row][col].configure(fg = 'red')
	curr_col = col

def flip():
	if curr_row == 0:
		hightlight_row(None)
	else:
		hightlight_row(randrange(0, ROWS))
	root.after(1000, flip)

flip()

root.mainloop()
