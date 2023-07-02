# módulos numéricos y de funciones científicas
import numpy as np
from scipy import signal as sig
from scipy.signal import TransferFunction
import matplotlib.pyplot as plt
import matplotlib as mpl

# PyTC2: La librería para TC2
from pytc2.sistemas_lineales import bodePlot, pzmap, GroupDelay, analyze_sys, pretty_print_lti

plt.figure(1)
plt.close(1)

fig_sz_x = 13
fig_sz_y = 7
fig_dpi = 80
fig_font_size = 16

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi
plt.rcParams.update({'font.size':fig_font_size})

epsilon = 0.509
erc = np.cbrt(epsilon) # epsilon^(1/3)
norma = 2*np.pi*40000

num = np.array([1,0,0,0]) 
den = np.array([1,2*erc*norma,2*erc*erc*norma*norma,epsilon*norma*norma*norma])

TS = sig.TransferFunction(num,den)

analyze_sys(TS, 'T_HP')