from spline import generate_spline
from tkinter import *
from random import random
import math

# INITIALIZING VARIABLES
colours_list = ["black", "red", "green", "blue", "cyan", "magenta"]
colours_index = math.floor(random() * 6)

# COUNTS THE NUMBER OF POINTS IN CURRENT LINE
click_number = 0

# STORE X AND Y COORDINATES OF CURRENT LINE
x_nodes = []
y_nodes = []


# FUNCTION FOR DRAWING CIRCLE ON TKINTER CANVAS
def draw_point(x_coord, y_coord, size, colour):
    x1, y1 = (x_coord - size), (y_coord - size)
    x2, y2 = (x_coord + size), (y_coord + size)
    canvas.create_oval(x1, y1, x2, y2, width=size, fill=colour, outline=colour)


# WHEN THE USER CLICKS THE MOUSE'S LEFT BUTTON, FUNCTION:
#  - DRAWS CIRCLE AT POINT WHERE THE USER CLICKS \
#  - STORES X (event.x) AND Y (event.y) COORDINATES OF THIS POINT IN RESPECTIVE LISTS  \
#  - INCREMENTS click_number BY 1
def register_point(event):
    global click_number, x_nodes, y_nodes
    x = event.x
    y = event.y
    draw_point(x_coord=x, y_coord=y, size=7, colour=colours_list[colours_index])
    x_nodes.append(x)
    y_nodes.append(y)
    click_number += 1


# WHEN THE USER CLICKS THE MOUSE'S RIGHT BUTTON:
#  IF click_number GREATER OR EQUAL TO 4:
#    - CUBIC SPLINE COORDINATES GENERATED BY generate_spline FUNCTION IN spline.py
#    - THESE POINTS ARE DRAWN WITH draw_point FUNCTION FROM ABOVE
#  ELSE:
#    - WARNING MESSAGE DISPLAYED

def draw_line(event=None):
    global click_number, x_nodes, y_nodes, colours_index, colours_list
    if click_number >= 2:
        for i in range(len(x_nodes)):
            for j in range(i+1, len(x_nodes)):
                if x_nodes[i] > x_nodes[j]:
                    x_nodes[i], x_nodes[j] = x_nodes[j], x_nodes[i]
                    y_nodes[i], y_nodes[j] = y_nodes[j], y_nodes[i]
        points = generate_spline(x_nodes, y_nodes)
        for i in range(len(points[0])):
            draw_point(x_coord=points[0][i], y_coord=points[1][i], size=1, colour=colours_list[colours_index])

        # VARIABLES ARE RESET
        x_nodes, y_nodes = [], []
        click_number = 0
        colours_index = math.floor(random() * 6)
    else:
        canvas.create_text(500, 400, text='YOU PLOTTED LESS THAN 1 POINT SO'
                                          '\nA CUBIC SPLINE COULD NOT BE DRAWN.'
                                          '\nPLEASE PLOT MORE.',
                           font=('times', 16, 'bold'), fill="red")


def display_canvas_instructions():
    canvas.create_text(500, 40, text='INSTRUCTIONS:', font=('times', 18, 'italic'))
    canvas.create_text(500, 100, text='1. Left click to plot a point.', font=('times', 18, 'italic'))
    canvas.create_text(500, 160, text='2. Right click (or click draw) to draw a spline.', font=('times', 18, 'italic'))


# FUNCTION EXECUTED WHEN RESET BUTTON PRESSED TO CLEAR CANVAS AND RESET VARIABLES
def reset():
    global x_nodes, y_nodes
    x_nodes, y_nodes = [], []
    canvas.delete("all")
    display_canvas_instructions()


# SETTING UP TKINTER WINDOW, CANVAS AND BUTTONS
window = Tk()
window.title('Spline tool')
window.config(padx=50, pady=50, bg='white')

canvas = Canvas(width=1000, height=1000, background='#f7f5dd')
canvas.grid(row=1, column=0)
display_canvas_instructions()

reset_button = Button(text='Reset', highlightthickness=0, command=reset, bg='yellow')
reset_button.grid(row=0, column=0)

draw_button = Button(text='Draw', highlightthickness=0, command=draw_line, bg='green', fg='white')
draw_button.grid(row=2, column=0)


# EVENT LISTENERS AND THEIR ASSOCIATED CALLBACK FUNCTIONS
canvas.bind('<Button-1>', register_point)  # EVENT: LEFT MOUSE CLICK
canvas.bind('<Button-3>', draw_line)  # EVENT: RIGHT MOUSE CLICK
window.mainloop()
