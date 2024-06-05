#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 12 21:31:57 2023

@author: cghiaus

Thermal circuit and state-space representation for a thermal circuit
with capacities in some nodes: cubic building
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import pandas as pd
from dm4bem import read_epw

#sys.path.append(r"C:\Users\ibit\OneDrive\Desktop\Ibi\Uni\Master\SEM 3 Grenoble\Smart_Cities\BE_cghiaus\modeling_cubic_building")
import dm4bem



# Physical analysis
# =================
l = 3               # m length of the cubic room
Sg = l**2           # m² surface of the glass wall
Sc = Si = 5 * Sg    # m² surface of concrete & insulation of the 5 walls

# Thermo-physical properties
air = {'Density': 1.2,                      # kg/m³
       'Specific heat': 1000}               # J/(kg·K)

wall = {'Conductivity': [1.4, 0.027, 1.4],  # W/(m·K)
        'Density': [2300, 55, 2500],        # kg/m³
        'Specific heat': [880, 1210, 750],  # J/(kg·K)
        'Width': [0.2, 0.08, 0.004],
        'Surface': [5 * l**2, 5 * l**2, l**2],  # m²
        'Slices': [1, 1, 1]}                # number of  slices
wall1 = pd.DataFrame(wall, index=['Concrete', 'Insulation', 'Glass'])

concrete = {'Conductivity': 1.400,
            'Density': 2300.0,
            'Specific heat': 880,
            'Width': 0.2,
            'Surface': 5 * l**2}

insulation = {'Conductivity': 0.027,
              'Density': 55.0,
              'Specific heat': 1210,
              'Width': 0.08,
              'Surface': 5 * l**2}

glass = {'Conductivity': 1.4,
         'Density': 2500,
         'Specific heat': 1210,
         'Width': 0.04,
         'Surface': l**2}

wall = pd.DataFrame.from_dict({'Layer_out': concrete,
                               'Layer_in': insulation,
                               'Glass': glass},
                              orient='index')

# radiative properties
ε_wLW = 0.85    # long wave emmisivity: wall surface (concrete)
ε_gLW = 0.90    # long wave emmisivity: glass pyrex
α_wSW = 0.25    # short wave absortivity: white smooth surface
α_gSW = 0.38    # short wave absortivity: reflective blue glass
τ_gSW = 0.30    # short wave transmitance: reflective blue glass

σ = 5.67e-8     # W/(m²⋅K⁴) Stefan-Bolzmann constant

Fwg = 1 / 5     # view factor wall - glass

# Decisive Values for the model

Kp = 1000   #gain for the heater
T_out = -5  # outside temperature
T_set = 20  # set temperature for inside the room
T_set_day = 22 # set temperature for inside the room in the day, intermittent heating
T_set_night = 16 # set temperature for inside the room in the night, intermittent heating
Qa = 1000   # heat source inside the room
Phi_out = 0      # radiation outside (wall), Φo
Phi_in = 0      # radioation inside (through window and in room absorbed), Φi
Phi_abs = 0      # radiation absorbed (glass), Φa
neglect_glass = True # do we neglect the capacity of the glass or not
neglect_air = True    # do we neglect capacity of air inside
X_time = 0.8    # Factor of shrinking time step to stabilize the y_exp

# convection coefficients
h = pd.DataFrame([{'in': 8., 'out': 25}], index=['h'])  # W/(m²⋅K)

# ---------------
def degree_hours (temp_inside_day, temp_inside_night):
    # degree hours calculation
    
    # Inputs
    # ======
    filename = 'FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2004-2018.epw'
    
    θ = temp_inside_day          # °C, indoor temperature all time
    θday = θ                     # °C, indoor temperature during day,, e.g.: 06:00 - 22:00
    θnight = temp_inside_night     # °C, indoor temperature during night 23:00 - 06:00
    
    period_start = '2012-01-01'
    period_end = '2012-12-31'
    
    daytime_start = '06:00:00+01:00'
    daytime_end = '22:00:00+01:00'
    # Computation
    # ===========
    # read Energy Plus Weather data (file .EPW)
    [data, meta] = read_epw(filename, coerce_year=2012)
    
    # select outdoor air temperature; call it θout
    df = data[["temp_air"]]
    del data
    df = df.rename(columns={'temp_air': 'θout'})
    
    # Select the data for a period of the year
    df = df.loc[period_start:period_end]
    # Compute degree-hours for fixed set-point
    # ----------------------------------------
    df['Δθfix'] = θ - df['θout'].where(
        df['θout'] < θ,
        θ)
    # Define start time for day and night
    day_start = pd.to_datetime(daytime_start).time()
    day_end = pd.to_datetime(daytime_end).time()
    
    # Daytime should be between 00:00 and 24:00
    # Daytime including midnight is not allowed, e.g., 22:00 till 06:00
    day = (df.index.time >= day_start) & (df.index.time <= day_end)
    night = ~day
    # Degree-hours for daytime
    df['Δθday'] = θday - df['θout'].where(
        (df['θout'] < θday) & day,
        θday)
    # Degree-hours for nighttime
    df['Δθnight'] = θnight - df['θout'].where(
        (df['θout'] < θnight) & night,
        θnight)
    # Sum of degree-hours for fixed indoor temperature
    DHH_fix = df['Δθfix'].sum()
    
    # Sum of degree-hours for intermittent heating
    DHH_interm = df['Δθday'].sum() + df['Δθnight'].sum()
    
    # Results
    # =======
    print(f"degree-hours fixed set-point: {DHH_fix:.1f} h·K")
    print(f"degree-hours variable set-point: {DHH_interm:.1f} h·K")
    print(f"Estimated savings: {(DHH_fix - DHH_interm) / DHH_fix * 100:.0f} %")



degree_hours(T_set_day, T_set_night)









# Thermal circuit
# ===============
# thermal conductances
# conduction
G_cd = wall['Conductivity'] / wall['Width'] * wall['Surface']

# convection
Gw = h * wall['Surface'][0]     # wall
Gg = h * wall['Surface'][2]     # glass

# long wave radiation
Tm = 20 + 273   # K, mean temp for radiative exchange

GLW1 = 4 * σ * Tm**3 * ε_wLW / (1 - ε_wLW) * wall['Surface']['Layer_in']
GLW12 = 4 * σ * Tm**3 * Fwg * wall['Surface']['Layer_in']
GLW2 = 4 * σ * Tm**3 * ε_gLW / (1 - ε_gLW) * wall['Surface']['Glass']

GLW = 1 / (1 / GLW1 + 1 / GLW12 + 1 / GLW2)

# ventilation & advection
Va = l**3                   # m³, volume of air
ACH = 1                     # air changes per hour
Va_dot = ACH / 3600 * Va    # m³/s, air infiltration

Gv = air['Density'] * air['Specific heat'] * Va_dot

# P-controler gain
#Kp = 1e4            # almost perfect controller Kp -> ∞
#Kp = 1e-3           # no controller Kp -> 0
Kp = Kp

# glass: convection outdoor & conduction
Ggs = float(1 / (1 / Gg['out'] + 1 / (2 * G_cd['Glass'])))

# Thermal capacities
C = wall['Density'] * wall['Specific heat'] * wall['Surface'] * wall['Width']
C['Air'] = air['Density'] * air['Specific heat'] * Va

# System of algebraic-differential equations (DAE)
# ================================================
A = np.zeros([12, 8])       # n° of branches X n° of nodes
A[0, 0] = 1                 # branch 0: -> node 0
A[1, 0], A[1, 1] = -1, 1    # branch 1: node 0 -> node 1
A[2, 1], A[2, 2] = -1, 1    # branch 2: node 1 -> node 2
A[3, 2], A[3, 3] = -1, 1    # branch 3: node 2 -> node 3
A[4, 3], A[4, 4] = -1, 1    # branch 4: node 3 -> node 4
A[5, 4], A[5, 5] = -1, 1    # branch 5: node 4 -> node 5
A[6, 4], A[6, 6] = -1, 1    # branch 6: node 4 -> node 6
A[7, 5], A[7, 6] = -1, 1    # branch 7: node 5 -> node 6
A[8, 7] = 1                 # branch 8: -> node 7
A[9, 5], A[9, 7] = 1, -1    # branch 9: node 5 -> node 7
A[10, 6] = 1                # branch 10: -> node 6
A[11, 6] = 1                # branch 11: -> node 6

G = np.hstack(
    [Gw['out'], 2 * G_cd['Layer_out'], 2 * G_cd['Layer_out'],
     2 * G_cd['Layer_in'], 2 * G_cd['Layer_in'], GLW,
     Gw['in'], Gg['in'], Ggs, 2 * G_cd['Glass'], Gv, Kp])

#G = np.diag(G)

neglect_glass = neglect_glass
neglect_air = neglect_air
                    # Capacities
if neglect_glass & neglect_air:
    C = [0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                0 , 0]
    #C = np.diag(C)
elif neglect_glass:
    C = [0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                C['Air'] , 0]
elif neglect_air:
    C = [0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                0 , C['Glass']]
else:
    C = [0, C['Layer_out'], 0, C['Layer_in'], 0, 0,
                 C['Air'], C['Glass']]
    #C = np.diag(C)

b = np.zeros(12)        # branches
b[[0, 8, 10, 11]] = 1   # branches with temperature sources

f = np.zeros(8)         # nodes
f[[0, 4, 6, 7]] = 1     # nodes with heat-flow sources

y = np.zeros(8)         # nodes
y = np.ones(8)
#y[[6]] = 1              # nodes (temperatures) of interest

# State-space representation
# ==========================
q = ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7", "q8", "q9", "q10", "q11"]
θ = ["θ0", "θ1", "θ2", "θ3", "θ4", "θ5", "θ6", "θ7"]


A = pd.DataFrame(A, index=q, columns=θ)
G = pd.Series(G, index=q)
C = pd.Series(C, index=θ)
b = pd.Series(b, index=q)
f = pd.Series(f, index=θ)
y = pd.Series(y, index=θ)

TC = {"A": A, "G": G, "C": C, "b": b, "f": f, "y": y}

[As, Bs, Cs, Ds, us] = dm4bem.tc2ss(TC)

# Steady-state
# ============
# from system of DAE
b = np.zeros(12)        # temperature sources
b[[0, 8, 10]] = T_out      # outdoor temperature
b[[11]] = T_set          # indoor set-point temperature

f = np.zeros(8)         # flow-rate sources
f[6] = Qa               # no consideration of radiation

G = np.diag(G)
C = np.diag(C)
θ = np.linalg.inv(A.T @ G @ A.to_numpy()) @ (A.T @ G @ b + f) # temperature of the nodes in steady state
                                                            # regarding DAE representation
q = G @ (-A @ θ + b)            # heat flows in steady state regarding DAE representation
print(f'θ = {θ} °C')
print(f"q = {q}, W")

# from state-space representation
bT = np.array([T_out, T_out, T_out, T_set])     # [To, To, To, T_set]
#bT = np.array([2*T_out, 2*T_out, 2*T_out, 20])     # [To, To, To, Tisp], so temp outside and set for inside
fQ = np.array([Phi_out, Phi_in, Qa, Phi_abs])         # [Φo, Φi, Qa, Φa]
u = np.hstack([bT, fQ])
yss = (-Cs.to_numpy() @ np.linalg.inv(As) @ Bs.to_numpy() + Ds) @ u # temp of the nodes in steady state
                                                                # regarding State-Space representation
print(f'yss = {yss} °C')

# Dynamic simulation
# ==================
λ = np.linalg.eig(As)[0]    # eigenvalues of matrix As

print('Time constants: \n', -1 / λ, 's \n')
print('2 x Time constants: \n', -2 / λ, 's \n')
dtmax = 2 * min(-1. / λ)
print(f'Maximum time step: {dtmax:.2f} s = {dtmax / 60:.2f} min')

# time step
dt = np.floor(dtmax / 60) * 60   # s
print(f'dt = {dt} s = {dt / 60:.0f} min')
dt = X_time * dt     # factor in order to reduce unstability of explicit Euler
# settling time
time_const = np.array([int(x) for x in sorted(-1 / λ)])
print('4 * Time constants: \n', 4 * time_const, 's \n')

t_settle = 4 * max(-1 / λ)
print(f'Settling time: \
{t_settle:.0f} s = \
{t_settle / 60:.1f} min = \
{t_settle / (3600):.2f} h = \
{t_settle / (3600 * 24):.2f} days')

# Step response
# -------------
# Find the next multiple of 3600 s that is larger than t_settle
duration = np.ceil(t_settle / 3600) * 3600
n = int(np.floor(duration / dt))    # number of time steps
t = np.arange(0, n * dt, dt)        # time vector for n time steps

print(f'Duration = {duration} s')
print(f'Number of time steps = {n}')

u = np.zeros([8, n])                # u = [To To To Tisp Φo Φi Qa Φa]
u[0:3, :] = T_out * np.ones([3, n])    # T_out = outside for temperature for n time steps
u[3, :] = T_set * np.ones([1, n])      # T_set = set Temperature for n time steps
u[6, :] = Qa * np.ones([1, n])    # Qa = inside heat source value for n time steps


n_s = As.shape[0]                      # number of state variables
θ_exp = np.zeros([n_s, t.shape[0]])    # explicit Euler in time t
θ_imp = np.zeros([n_s, t.shape[0]])    # implicit Euler in time t

I = np.eye(n_s)                        # identity matrix


for k in range(n - 1):
    θ_exp[:, k + 1] = (I + dt * As) @\
        θ_exp[:, k] + dt * Bs @ u[:, k]
    θ_imp[:, k + 1] = np.linalg.inv(I - dt * As) @\
        (θ_imp[:, k] + dt * Bs @ u[:, k])

y_exp = Cs @ θ_exp + Ds @  u    # explicit creating the numerical outputs of temperature nodes for time steps
y_imp = Cs @ θ_imp + Ds @  u    # implicit creating the numerical outputs of temperature nodes for time steps

fig, ax_1 = plt.subplots()      # plot for all y_exp and y_imp[6]
for i in range(y_exp.shape[0]): # all of y_exp except T_in
    if i != 6:
        ax_1.plot(t / 3600, y_exp.T.iloc[:, i]) 

ax_1.plot(t / 3600, y_exp.T.iloc[:,6], label='T_in explizit') # plot of T_in reg. y_exp
ax_1.plot(t / 3600, y_imp.T.iloc[:,6], label='T_in implizit') # plot of T_in reg. y_imp

ax_1.set(xlabel='$Time$ [h]',
        ylabel='$T_i$ [°C]',
        title='Step input: To, step response y_exp')
ax_1.legend()
ax_1.grid()
plt.show()


fig, ax_2 = plt.subplots()              #plot for y_exp[6] and y_imp[6]


ax_2.plot(t/3600, y_exp.T.iloc[:,6], label='T_in explizit')
ax_2.plot(t/3600, y_imp.T.iloc[:,6], label='T_in implizit')
T_in_st_st_ss = yss[6] * np.ones([y_exp.shape[1], 1])                # creating value of steady state temperature
ax_2.plot(t/3600, T_in_st_st_ss, label='T_in Steady State ss')         # putting into same graphic for comparison of
                                                                    # end value of step response and steady state

ax_2.set(xlabel='$Time$ [h]',
        ylabel='$T_i$ [°C]',
        title='Step input: To, step response Ti')
ax_2.legend()
ax_2.grid()

plt.show()

print('Steady-state indoor temperature obtained with:')
print(f'- DAE model: {float(θ[6]):.4f} °C')
print(f'- state-space model: {float(yss[6]):.4f} °C')
#print(y_exp)
print(f'- step response: steady state value after step input: {y_exp.loc["θ6"].tail(1).values[0]:.4f} °C')
#print(f'- steady-state response to step input: {float(y_exp[7, 17]):.4f} °C')


# Simulation with weather data
# ----------------------------
# Input vector
# weather data
start_date = '01-03 10:00:00'
end_date = '02-09 18:00:00'

start_date = '2012-' + start_date
end_date = '2012-' + end_date
print(f'{start_date} \tstart date')
print(f'{end_date} \tend date')


filename = 'FRA_AR_Grenoble.Alpes.Isere.AP.074860_TMYx.2004-2018.epw'
[data, meta] = dm4bem.read_epw(filename, coerce_year=2012)
weather = data[["temp_air", "dir_n_rad", "dif_h_rad"]]
del data
weather.index = weather.index.map(lambda t: t.replace(year=2012))
weather = weather[(
    weather.index >= start_date) & (
    weather.index < end_date)]

# solar radiatiion on the walls
surface_orientation = {'slope': 90,
                        'azimuth': 0,
                        'latitude': 45}
albedo = 0.2
rad_surf = dm4bem.sol_rad_tilt_surf(
    weather, surface_orientation, albedo)
rad_surf['Etot'] = rad_surf.sum(axis=1)

# resample the weather data
data = pd.concat([weather['temp_air'], rad_surf['Etot']], axis=1)
#if dt>3600:
 #   dt = 3600
data = data.resample(str(dt) + 'S').interpolate(method='linear')
# filling of nan values
#data = data.ffill().bfill()
data = data.rename(columns={'temp_air': 'To'})

# other inputs
data['Ti'] = T_set * np.ones(data.shape[0])
data['Q_a'] = Qa * np.ones(data.shape[0])

# input vector
To = data['To']
Ti = data['Ti']
Φo = α_wSW * wall['Surface']['Layer_out'] * data['Etot']
Φi = τ_gSW * α_wSW * wall['Surface']['Glass'] * data['Etot']
Q_a = data['Q_a']
Φa = α_gSW * wall['Surface']['Glass'] * data['Etot']

u_wheather = pd.concat([To, To, To, Ti, Φo, Φi, Q_a, Φa], axis=1)
u_wheather.columns.values[[4, 5, 7]] = ['Φo', 'Φi', 'Φa']
u_wheather = u_wheather.fillna(0)

# Initial conditions
θ_exp_weather = T_set * np.ones([As.shape[0], u_wheather.shape[0]])

print("As shape:", As.shape)
print("Bs shape:", Bs.shape)
print("Cs shape:", Cs.shape)
print("Ds shape:", Ds.shape)
print("u_wheather shape:", u_wheather.shape)
print("theta_exp shape:", θ_exp_weather.shape)

print(u_wheather.iloc[0, :])



# Time integration
for k in range(u_wheather.shape[0] - 1):
    θ_exp_weather[:, k + 1] = (I + dt * As.to_numpy()) @ θ_exp_weather[:, k] + dt * Bs.to_numpy() @ u_wheather.iloc[k, :].to_numpy()


for k in range(u_wheather.shape[0] - 1):
    θ_exp_weather[:, k + 1] = (I + dt * As.to_numpy()) @ θ_exp_weather[:, k]\
        + dt * Bs.to_numpy() @ u_wheather.iloc[k, :]

for k in range(u_wheather.shape[0] - 1):
    θ_exp_weather[:, k + 1] = (I + dt * As) @ θ_exp_weather[:, k]\
        + dt * Bs.to_numpy() @ u_wheather.iloc[k, :]


# outputs
y_exp_weather = Cs.to_numpy() @ θ_exp_weather + Ds.to_numpy() @ u_wheather.to_numpy().T
#y_exp_weather = Cs @ θ_exp_weather + Ds.to_numpy() @ u_wheather.to_numpy().T
q_HVAC = Kp * (data['Ti'] - y_exp_weather[0, :])

# plot
t = dt * np.arange(data.shape[0])   # time vector

fig, axs = plt.subplots(2, 1)
# plot indoor and outdoor temperature
axs[0].plot(t / 3600 / 24, y_exp_weather[6, :], label='$T_{indoor}$')
axs[0].plot(t / 3600 / 24, data['To'], label='$T_{outdoor}$')
axs[0].set(xlabel='Time [days]',
            ylabel='Temperatures [°C]',
            title='Simulation for weather')
axs[0].legend(loc='upper right')

# plot total solar radiation and HVAC heat flow
axs[1].plot(t / 3600 / 24, q_HVAC, label='$q_{HVAC}$')
axs[1].plot(t / 3600 / 24, data['Etot'], label='$Φ_{total}$')
axs[1].set(xlabel='Time [days]',
            ylabel='Heat flows [W]')
axs[1].legend(loc='upper right')

fig.tight_layout()


