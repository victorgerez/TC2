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

#-------------------------------------------------------------------------------------------------------------------#

# Datos de plantilla

alfa_max = 1                        # dB
alfa_min = 30                       # dB

f_s_plantilla = 10e3               # Hz
f_p_plantilla = 40e3               # Hz

omega_p_plantilla = 2*np.pi*f_p_plantilla
omega_s_plantilla = 2*np.pi*f_s_plantilla

norm_w = omega_p_plantilla

omega_p_norm = omega_p_plantilla/norm_w
omega_s_norm = omega_s_plantilla/norm_w

omega_p_pb_prototipo = 1/omega_p_norm
omega_s_pb_prototipo = 1/omega_s_norm

print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('-- Datos de plantilla - Filtro pasa altos objetivo --')
print(' ')

print('α_min = {}'.format(alfa_min))
print('α_max = {}'.format(alfa_max))
print('fs = {}'.format(f_s_plantilla))
print('fp = {}'.format(f_p_plantilla))

print(' ')

print('ωp = {:3.4f}'.format(omega_p_plantilla))
print('ωs = {:3.4f}'.format(omega_s_plantilla))
print('ωp_n = {:3.4f}'.format(omega_p_norm))
print('ωs_n = {:3.4f}'.format(omega_s_norm))

print(' ')
print('-- Dominio transformado - Filtro pasa bajos prototipo --')
print(' ')

print('Ωp = {:3.4f}'.format(omega_p_pb_prototipo))
print('Ωs = {:3.4f}'.format(omega_s_pb_prototipo))

#-------------------------------------------------------------------------------------------------------------------#

# Obtenemos orden y omega 0 del filtro pasa bajos prototipo

n, w0 = sig.buttord(omega_p_pb_prototipo, omega_s_pb_prototipo, alfa_max, alfa_min, analog=True)


print('#---------------------------------------------------------------------------------------------#')
print(' ')
print('Orden del filtro: n = {}'.format(n))
print('Frecuencia angular de corte: w0 = {}'.format(w0))

#-------------------------------------------------------------------------------------------------------------------#

# Obtenemos filtro pasa bajos prototipo de máxima planicidad

z,p,k = sig.butter(n, w0, btype='lowpass', analog=True, output='zpk')

print(' ')
print('#---------------------------------------------------------------------------------------------#')

print(' Filtro pasa bajos prototipo')
print('#---------------------------------------------------------------------------------------------#')

print('-- z: Ceros coincidentes con los obtenidos de forma analítica --')
print(' ')

print('z= {}'.format(z))

print(' ')
print('-- p: Polos coincidentes con los obtenidos de forma analítica --')
print(' ')

print('p= {}'.format(p))

# Transformamos las singularidades que nos devuelve butter() para obtener 
# el numerador y el denominador de la T_LP_N(s) del filtro pasa bajos prototipo normalizado:

num_lp, den_lp = sig.zpk2tf(z,p,k)

# Imprimimos la T_LP_N(s) en formato T(s)= num/den con la funcion pretty_print_lti(num, den):

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print('-- Función transferencia T_LP(s) normalizada:')
print(' ')

pretty_print_lti(num_lp, den_lp)

#-------------------------------------------------------------------------------------------------------------------#

# Transformamos la T_LP_N(s) del filtro pasa bajos prototipo para obtener 
# el numerador y el denominador de la T_HP_N(s) con lp2hp(num, den):
print(' ')
print('#---------------------------------------------------------------------------------------------#')

print(' Filtro pasa altos objetivo')
print('#---------------------------------------------------------------------------------------------#')

num_hp, den_hp = sig.lp2hp(num_lp, den_lp)

print(' ')
print('-- z: Ceros coincidentes con los obtenidos de forma analítica --')
print(' ')

print('z1,2= {}'.format(np.roots(num_hp)))

print(' ')
print('-- p: Polos coincidentes con los obtenidos de forma analítica --')
print(' ')

print('p1,2= {}'.format(np.roots(den_hp)))

print(' ')
print('#---------------------------------------------------------------------------------------------#')

# Imprimimos la T_HP_N(s) en formato T(s)= num/den con la funcion pretty_print_lti(num, den):

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print('-- Función transferencia T_HP(s) normalizada:')
print(' ')

pretty_print_lti(num_hp, den_hp)

#-------------------------------------------------------------------------------------------------------------------#


T_HP_N= sig.TransferFunction(num_hp, den_hp)                   # numerador y denominador con butter()

#-------------------------------------------------------------------------------------------------------------------#

# Representamos módulo, fase, diagrama de polos y ceros y retardo de fase con la analyze_sys() que tiene, entre
# otras cosas, indicadores (w0 y Q) de las singularidades en el diagrama de polos y ceros

print(' ')
print('#---------------------------------------------------------------------------------------------#')
print('-- Módulo, fase, diagrama de polos y ceros y retardo de fase --')
print(' ')

analyze_sys(T_HP_N, sys_name='T_HP_N')

#-------------------------------------------------------------------------------------------------------------------#
