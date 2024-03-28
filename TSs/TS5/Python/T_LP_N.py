# módulos numéricos y de funciones científicas
import numpy as np
from scipy import signal as sig
from scipy.signal import TransferFunction
import matplotlib.pyplot as plt
import matplotlib as mpl

# PyTC2: La librería para TC2
from pytc2.sistemas_lineales import bodePlot, pzmap, GroupDelay, analyze_sys, pretty_print_lti, pretty_print_bicuad_omegayq, pretty_print_SOS

plt.figure(1)
plt.close(1)

fig_sz_x = 13
fig_sz_y = 7
fig_dpi = 80
fig_font_size = 16

mpl.rcParams['figure.figsize'] = (fig_sz_x, fig_sz_y)
mpl.rcParams['figure.dpi'] = fig_dpi
plt.rcParams.update({'font.size':fig_font_size})

#-------------------------------------------------------------------------------------------------------------------#

# Datos de plantilla

alfa_max = 0.5                        # dB
alfa_min = 16                       # dB

f_0_plantilla = 22e3               # Hz
f_s1_plantilla = 17e3               # Hz
f_s2_plantilla = 36e3               # Hz

omega_0_plantilla = 2*np.pi*f_0_plantilla
omega_s1_plantilla = 2*np.pi*f_s1_plantilla
omega_s2_plantilla = 2*np.pi*f_s2_plantilla

norm_w = omega_0_plantilla

omega_0_norm = omega_0_plantilla/norm_w
omega_s1_norm = omega_s1_plantilla/norm_w
omega_s2_norm = omega_s2_plantilla/norm_w
omega_p1_norm = 0.905               # Hz
omega_p2_norm = 1.105               # Hz


print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('-- Datos de plantilla - Filtro pasa banda objetivo --')
print(' ')

print('α_min = {}  (se utiliza la más exigente)'.format(alfa_min))
print('α_max = {}'.format(alfa_max))
print(' ')
print('f0 = {}'.format(f_0_plantilla))
print('fs1 = {}'.format(f_s1_plantilla))
print('fs2 = {}'.format(f_s2_plantilla))
print(' ')
print('ω_0_N = {:3.4f}'.format(omega_0_norm))
print('ω_p1_N = {:3.4f}'.format(omega_p1_norm))
print('ω_p2_N = {:3.4f}'.format(omega_p2_norm))
print('ω_s1_N = {:3.4f}'.format(omega_s1_norm))
print('ω_s2_N = {:3.4f}'.format(omega_s2_norm))


#-------------------------------------------------------------------------------------------------------------------#

# Obtenemos orden y omega_0 del filtro

n, wn = sig.cheb1ord([omega_p1_norm,omega_p2_norm], [omega_s1_norm,omega_s2_norm], alfa_max, alfa_min, analog=True)

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('Orden del filtro: n = {}'.format(n))
print('Frecuencias angulares de paso: ω_p_N_1;2 = {}'.format(wn))

#-------------------------------------------------------------------------------------------------------------------#

# Obtenemos transferencia pasabajos prototipo normalizada

z, p, k = sig.cheb1ap(n, alfa_max)

print(' ')
print(' ')
print('#---------------------------------------------------------------------------------------------#')
print('Función transferencia pasa bajos prototipo normalizada:')
print(' ')

print('-- z: Ceros coincidentes con los obtenidos de forma analítica --')
print(' ')

print('z= {}'.format(z))

print(' ')
print('-- p: Polos coincidentes con los obtenidos de forma analítica --')
print(' ')

print('p= {}'.format(p))

print(' ')
print('-- k: Ganancia coincidentes con la obtenida de forma analítica --')
print(' ')

print('k= {}'.format(k))

# Transformamos las singularidades que nos devuelve cheb1ap() para obtener 
# el numerador y el denominador de la T_LP_N(s) del filtro pasa bajos prototipo normalizado:

num_lp, den_lp = sig.zpk2tf(z,p,k)

# Imprimimos la T_LP_N(s) en formato T(s)= num/den con la funcion pretty_print_lti(num, den):

print(' ')
print(' ')
print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('T_LP_N(s):')
pretty_print_lti(num_lp, den_lp)
print(' ')
print('#---------------------------------------------------------------------------------------------#')
print(' ')

TS5 = sig.TransferFunction(num_lp,den_lp)

analyze_sys(TS5, 'T_LP_N')