"""
Hector Juan Mari (St. number: 5620325)

This module computes the dynamics of an interplanetary low-thrust trajectory, using a thrust profile determined from
a semi-analytical Hodographic shaping method (see Gondelach and Noomen, 2015). This file propagates the dynamics
using a variety of  integrator and propagator settings. For each run, the differences w.r.t. a benchmark propagation are
computed, providing a proxy for setting quality. The benchmark settings are currently defined semi-randomly, and are to be
analyzed/modified.

The semi-analytical trajectory of the vehicle is determined by its departure and arrival time (which define the initial and final states)
as well as the free parameters of the shaping method. The free parameters of the shaping method defined here are the same
as for the 'higher-order solution' in Section V.A of Gondelach and Noomen (2015). The free parameters define the amplitude
of specific types of velocity shaping functions. The low-thrust hodographic trajectory is parameterized by the values of
the variable trajectory_parameters (see below). The low-thrust trajectory computed by the shape-based method starts
at the Earth's center of mass, and terminates at Mars's center of mass.

The semi-analytical model is used to compute the thrust as a function of time (along the ideal semi-analytical trajectory).
This function is then used to define a thrust model in the numerical propagation.

In the propagation, the vehicle starts on the Hodographic low-thrust trajectory, 30 days
(defined by the time_buffer variable) after it 'departs' the Earth's center of mass.

The propagation is terminated as soon as one of the following conditions is met (see
get_propagation_termination_settings() function):

* Distance to Mars < 50000 km
* Propagation time > Time-of-flight of hodographic trajectory

This propagation as provided assumes only point mass gravity by the Sun and thrust acceleration of the vehicle.
Both the translational dynamics and mass of the vehicle are propagated, using a fixed specific impulse.

The entries of the vector 'trajectory_parameters' contains the following:
* Entry 0: Departure time (from Earth's center-of-mass) in Julian days since J2000
* Entry 1: Time-of-flight from Earth's center-of-mass to Mars' center-of-mass, in Julian days
* Entry 2: Number of revolutions around the Sun
* Entry 3,4: Free parameters for radial shaping functions
* Entry 5,6: Free parameters for normal shaping functions
* Entry 7,8: Free parameters for axial shaping functions

"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
from matplotlib import pyplot as plt
import datetime

# Tudatpy imports
from tudatpy.kernel.interface import spice
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array
from tudatpy.kernel.astro import time_conversion
import pickle

# Problem-specific imports
import Utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

initial_time_1 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2028, 5, 1))-constants.JULIAN_DAY_ON_J2000
initial_time_2 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2029, 5, 1))-constants.JULIAN_DAY_ON_J2000


n = 100
initial_time_list_1 = np.linspace(initial_time_1, initial_time_2, n)
time_of_flight_1 = 100
time_of_flight_2 = 500


time_of_flight_list_1 = np.linspace(time_of_flight_1, time_of_flight_2, n)

initial_time_1 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2028, 5, 1))-constants.JULIAN_DAY_ON_J2000
initial_time_2 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2029, 5, 1))-constants.JULIAN_DAY_ON_J2000


n = 100
initial_time_list_2 = np.linspace(initial_time_1, initial_time_2, n)
time_of_flight_1 = 100
time_of_flight_2 = 500


time_of_flight_list_2 = np.linspace(time_of_flight_1, time_of_flight_2, n)

delta_v = np.zeros([n, n, 2])
delta_v_node = np.zeros([n, 2])

compute = True
# Choose whether benchmark is run
use_benchmark = True
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = 'C:/Users/hecto/Desktop/TU Delft/Thesis'

if compute:

    # Define settings for celestial bodies
    bodies_to_create = ['Earth',
                        'Mars',
                        'Sun']
    # Define coordinate system
    global_frame_origin = 'SSB'
    global_frame_orientation = 'ECLIPJ2000'
    # Create body settings
    body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                                global_frame_origin,
                                                                global_frame_orientation)
    # Create bodies
    bodies = environment_setup.create_system_of_bodies(body_settings)

    i = 0
    for initial_time in initial_time_list_1:
        print(i)
        j = 0
        for time_of_flight in time_of_flight_list_1:
            trajectory_parameters = [initial_time,
                                     time_of_flight,
                                     0,
                                     8661.95,
                                     8425.79,
                                     -8632.97,
                                     5666.72,
                                     -3567.68,
                                     -2806.92]

            ###########################################################################
            # DEFINE SIMULATION SETTINGS ##############################################
            ###########################################################################

            # Vehicle settings
            vehicle_mass = 22
            specific_impulse = 2500.0
            # Fixed parameters
            minimum_mars_distance = 5.0E7
            # Time since 'departure from Earth CoM' at which propagation starts (and similar
            # for arrival time)
            time_buffer = 0.0 * constants.JULIAN_DAY
            # Time at which to start propagation
            initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                                        time_buffer)

            ###########################################################################
            # CREATE ENVIRONMENT ######################################################
            ###########################################################################

            # Create vehicle object and add it to the existing system of bodies
            bodies.create_empty_body('Vehicle')
            bodies.get_body('Vehicle').mass = vehicle_mass

            # # Create Lambert arc state model
            # lambert_arc_ephemeris = Util.get_lambert_problem_result(bodies, 'Mars', Util.get_trajectory_initial_time(trajectory_parameters_lambert), Util.get_trajectory_final_time(trajectory_parameters_lambert))
            # # Create propagation settings and propagate dynamics
            # dynamics_simulator = Util.propagate_trajectory(Util.get_trajectory_initial_time(trajectory_parameters_lambert), Util.get_trajectory_final_time(trajectory_parameters_lambert), bodies, lambert_arc_ephemeris,
            #                                           use_perturbations=False)
            # # Extract state history from dynamics simulator
            # state_history = dynamics_simulator.state_history
            # state_history_matrix = result2array(state_history)

            # Adding excess velocity
            vinf = np.zeros([3, 1])
            # vinf = state_history_matrix[0,4:]-bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:]
            # vinf = 4000/np.linalg.norm(bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:])*bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:]

            ###########################################################################
            # WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
            ###########################################################################

            # Create problem without propagating
            hodographic_shaping_object_1 = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                              bodies, vinf)

            delta_v[i, j, 0] = hodographic_shaping_object_1.delta_v_per_leg[0]
            delta_v_node[i] = hodographic_shaping_object_1.delta_v_per_node

            j += 1
        i += 1

    i = 0
    for initial_time in initial_time_list_2:
        print(i)
        j = 0
        for time_of_flight in time_of_flight_list_2:
            trajectory_parameters = [initial_time,
                                     time_of_flight,
                                     0,
                                     8661.95,
                                     8425.79,
                                     -8632.97,
                                     5666.72,
                                     -3567.68,
                                     -2806.92]

            ###########################################################################
            # DEFINE SIMULATION SETTINGS ##############################################
            ###########################################################################

            # Vehicle settings
            vehicle_mass = 22
            specific_impulse = 2500.0
            # Fixed parameters
            minimum_mars_distance = 5.0E7
            # Time since 'departure from Earth CoM' at which propagation starts (and similar
            # for arrival time)
            time_buffer = 0.0 * constants.JULIAN_DAY
            # Time at which to start propagation
            initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                                        time_buffer)

            ###########################################################################
            # CREATE ENVIRONMENT ######################################################
            ###########################################################################

            # Create vehicle object and add it to the existing system of bodies
            bodies.create_empty_body('Vehicle')
            bodies.get_body('Vehicle').mass = vehicle_mass

            # # Create Lambert arc state model
            # lambert_arc_ephemeris = Util.get_lambert_problem_result(bodies, 'Mars', Util.get_trajectory_initial_time(trajectory_parameters_lambert), Util.get_trajectory_final_time(trajectory_parameters_lambert))
            # # Create propagation settings and propagate dynamics
            # dynamics_simulator = Util.propagate_trajectory(Util.get_trajectory_initial_time(trajectory_parameters_lambert), Util.get_trajectory_final_time(trajectory_parameters_lambert), bodies, lambert_arc_ephemeris,
            #                                           use_perturbations=False)
            # # Extract state history from dynamics simulator
            # state_history = dynamics_simulator.state_history
            # state_history_matrix = result2array(state_history)

            # vinf = state_history_matrix[0,4:]-bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:]
            # vinf = 4000/np.linalg.norm(bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:])*bodies.get_body('Earth').state_in_base_frame_from_ephemeris(Util.get_trajectory_initial_time(trajectory_parameters))[3:]
            vinf = [[4000], [0], [0]]
            # aux = np.zeros([3,1])
            # aux[0] = vinf[0]
            # aux[1] = vinf[1]
            # aux[2] = vinf[2]

            # vec1 = vinf
            # vec2 = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(
            #     Util.get_trajectory_initial_time(trajectory_parameters))[3:]
            # plt.figure()
            # ax = plt.axes(projection='3d')
            # ax.plot3D([0,vec2[0]], [0,vec2[1]], [0,vec2[2]])
            # ax.plot3D([0,vec1[0]], [0,vec1[1]], [0,vec1[2]])

            # vinf = aux

            ###########################################################################
            # WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
            ###########################################################################

            # Create problem without propagating
            hodographic_shaping_object_2 = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                              bodies, vinf)

            delta_v[i, j, 1] = hodographic_shaping_object_2.delta_v_per_leg[0]

            j += 1
        i += 1

    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/HodographicAnalysis/delta_v_7.pickle",
                  "wb") as f:
            pickle.dump(delta_v, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
else:
    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/HodographicAnalysis/delta_v_7.pickle",
                  "rb") as f:
            delta_v = pickle.load(f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

delta_v[:, :, 0] = delta_v[:, :, 0].transpose()

departure_date_list = [time_conversion.julian_day_to_calendar_date(epoch + constants.JULIAN_DAY_ON_J2000) for epoch in
                       initial_time_list_1]

ind = np.unravel_index(np.argmin(delta_v[:, :, 0], axis=None), delta_v[:, :, 0].shape)
print(departure_date_list[ind[1]], time_of_flight_list_1[ind[0]])
fig, ax = plt.subplots(figsize=(width, height))
c1 = ax.pcolormesh(departure_date_list, time_of_flight_list_1, delta_v[:, :, 0], cmap='coolwarm', vmax=50000)
c2 = ax.scatter(departure_date_list[ind[1]], time_of_flight_list_1[ind[0]], color='tab:green')
fig.colorbar(c1, ax=ax)
plt.title("Without excess velocity")

delta_v[:, :, 1] = delta_v[:, :, 1].transpose()

departure_date_list = [time_conversion.julian_day_to_calendar_date(epoch + constants.JULIAN_DAY_ON_J2000) for epoch in
                       initial_time_list_2]

ind = np.unravel_index(np.argmin(delta_v[:, :, 1], axis=None), delta_v[:, :, 1].shape)
print(departure_date_list[ind[1]], time_of_flight_list_2[ind[0]])
fig, ax = plt.subplots(figsize=(width, height))
c1 = ax.pcolormesh(departure_date_list, time_of_flight_list_2, delta_v[:, :, 1], cmap='coolwarm', vmax=50000)
c2 = ax.scatter(departure_date_list[ind[1]], time_of_flight_list_2[ind[0]], color='tab:green')
fig.colorbar(c1, ax=ax)
plt.title("With excess velocity")

print('Minimum delta-V without excess velocity is', np.min(delta_v[:, :, 0]), "m/s")

print('Minimum delta-V with excess velocity is', np.min(delta_v[:, :, 1]), "m/s")

plt.show()

exit()

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
else:
    output_path = None

# Retrieves analytical results and write them to a file
states_1 = Util.get_hodographic_trajectory(hodographic_shaping_object_1,
                                           output_path)
states_2 = Util.get_hodographic_trajectory(hodographic_shaping_object_2,
                                           output_path)

states_list_1 = result2array(states_1)
states_list_2 = result2array(states_2)

# Retrieve the Earth trajectory over vehicle propagation epochs from spice
earth_states_from_spice = {
    epoch: bodies.get_body('Earth').state_in_base_frame_from_ephemeris(epoch)
    for epoch in list(states_1.keys())
}
# Convert the dictionary to a multi-dimensional array
earth_array = result2array(earth_states_from_spice)

# Retrieve the Mars trajectory over vehicle propagation epochs from spice
mars_states_from_spice = {
    epoch: bodies.get_body('Mars').state_in_base_frame_from_ephemeris(epoch)
    for epoch in list(states_1.keys())
}
# Convert the dictionary to a multi-dimensional array
mars_array = result2array(mars_states_from_spice)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

plt.figure(figsize=(width, height))
ax = plt.axes(projection='3d')

ax.plot3D(states_list_1[:, 1], states_list_1[:, 2], states_list_1[:, 3], label='Vehicle', linewidth=1.5,
          color='tab:blue')
ax.plot3D(states_list_2[:, 1], states_list_2[:, 2], states_list_2[:, 3], label='Vehicle', linewidth=1.5,
          color='tab:red')
ax.plot3D(earth_array[:, 1], earth_array[:, 2], earth_array[:, 3], label='Earth', linewidth=0.8, color='tab:green')
ax.plot3D(mars_array[:, 1], mars_array[:, 2], mars_array[:, 3], label='Mars', linewidth=0.8, color='tab:orange')

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20

ax.legend(fontsize='small')
# plt.grid(True)
# plt.tight_layout()
plt.savefig("hodographic-transfer.png")

vec1 = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(
    Util.get_trajectory_initial_time(trajectory_parameters))[3:]
vec2 = states_list_2[0, 4:7] - vec1
plt.figure()
ax = plt.axes(projection='3d')
ax.plot3D([0, vec1[0]], [0, vec1[1]], [0, vec1[2]], label='Earth')
ax.plot3D([0, vec2[0]], [0, vec2[1]], [0, vec2[2]], label='Vehicle')
plt.legend()

C3 = (np.linalg.norm(states_list_2[0, 4:7] - earth_array[0, 4:7]) / 1000) ** 2

print("The initial C3 is", C3, "km^2/s^2")

final_dist = np.linalg.norm(states_list_2[-1, 1:4] - mars_array[-1, 1:4])

print("The final distance to Mars is", final_dist, "m")

delta_v = hodographic_shaping_object_1.delta_v_per_leg

print("The delta-V for the transfer is", delta_v, "m/s")

delta_v = hodographic_shaping_object_2.delta_v_per_leg

print("The delta-V for the transfer with excess velocity is", delta_v, "m/s")

plt.show()
