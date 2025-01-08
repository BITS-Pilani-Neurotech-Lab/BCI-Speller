import tkinter as tk
from time import sleep
from random import randrange

root = tk.Tk()
root.title("P300 Row-matrix speller")
root.configure(bg = 'black')
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

FLASH_INTERVAL = 500

curr_row = None
curr_col = None

selected_row = None
selected_col = None

labels = [[None] * COLS for row in range(ROWS)]
for row in range(ROWS):
	for col in range(COLS):
		labels[row][col] = tk.Label(root, text = matrix[row][col], bg = 'black', fg = 'white', font = ('Arial', 20))
		labels[row][col].grid(row = row, column = col, sticky = tk.NSEW, ipadx = 50, ipady = 50)

text_label = tk.Label(root, bg = 'black', fg = 'white', font = ('Arial', 50))
text_label.grid(row = ROWS, column = 0, columnspan = COLS, ipadx = 50, ipady = 50)
text = ''

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

def event_detected(event):
	global selected_row, selected_col
	if curr_row is not None:
		selected_row = curr_row
	if curr_col is not None:
		selected_col = curr_col

def add_char():
	global text, selected_row, selected_col
	text += matrix[selected_row][selected_col]
	text_label.configure(text = text)
	selected_row = None
	selected_col = None

def select_row():
	if selected_row is not None:
		hightlight_row(None)
		root.after(FLASH_INTERVAL, select_col)
		return
	row = randrange(ROWS)
	hightlight_row(row)
	root.after(FLASH_INTERVAL, select_row)

def select_col():
	if selected_col is not None:
		add_char()
		hightlight_col(None)
		root.after(FLASH_INTERVAL, select_row)
		return
	col = randrange(ROWS)
	hightlight_col(col)
	root.after(FLASH_INTERVAL, select_col)

root.bind('<space>', event_detected)
select_row()
root.mainloop()
