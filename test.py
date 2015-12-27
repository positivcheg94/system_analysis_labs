#!/usr/bin/env python

from numpy import arange, sin, pi
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


from matplotlib.figure import Figure

import sys
if sys.version_info[0] < 3:
    import Tkinter as Tk
else:
    import tkinter as Tk

root = Tk.Tk()
root.wm_title("Embedding in TK")


f = Figure()
a = f.add_subplot(4,1,1)
b = f.add_subplot(4,1,2)
c = f.add_subplot(4,1,3)
d = f.add_subplot(4,1,4)
t = arange(0.0, 3.0, 0.01)
s = sin(2*pi*t)

a.plot(t, s,'r')
b.plot(t, s,'g')
c.plot(t, s,'b')
d.plot(t, s,'o')
print(type(a))


# a tk.DrawingArea
canvas = FigureCanvasTkAgg(f, master=root)
canvas.show()
canvas.get_tk_widget().pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)

canvas._tkcanvas.pack(side=Tk.TOP, fill=Tk.BOTH, expand=1)


def on_key_event(event):
    print('you pressed %s' % event.key)

canvas.mpl_connect('key_press_event', on_key_event)


def clear():
    pass

def _quit():
    root.quit()     # stops mainloop
    root.destroy()  # this is necessary on Windows to prevent
                    # Fatal Python Error: PyEval_RestoreThread: NULL tstate

button = Tk.Button(master=root, text='Quit', command=_quit)
button.pack(side=Tk.BOTTOM)


Tk.mainloop()