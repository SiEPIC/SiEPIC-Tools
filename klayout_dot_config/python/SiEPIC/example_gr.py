#example of plotting with gr library
from gr import pygr
from numpy import *
x = list(i for i in range(10))
y = [sin(i) for i in range(10)]
pygr.plot(x,y)