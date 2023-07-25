"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

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
This function is then used to define a thrust model in the numerical propagation

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

Details on the outputs written by this file can be found:
* Benchmark data: comments for 'generate_benchmarks()' and 'compare_benchmarks()' function
* Results for integrator/propagator variations: comments under "RUN SIMULATION FOR VARIOUS SETTINGS"
* Trajectory for semi-analytical hodographic shape-based solution: comments with, and call to
    get_hodographic_trajectory() function

Frequent warnings and/or errors that might pop up:

* One frequent warning could be the following (mock values):
    "Warning in interpolator, requesting data point outside of boundaries, requested data at 7008 but limit values are
    0 and 7002, applying extrapolation instead."

    It can happen that the benchmark ends earlier than the regular simulation, due to the smaller step size. Therefore,
    the code will be forced to extrapolate the benchmark states (or dependent variables) to compare them to the
    simulation output, producing a warning. This warning can be deactivated by forcing the interpolator to use the boundary
    value instead of extrapolating (extrapolation is the default behavior). This can be done by setting:

    interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation = interpolators.extrapolate_at_boundary)

* One frequent error could be the following:
    "Error, propagation terminated at t=4454.723896, returning propagation data up to current time."
    This means that an error occurred with the given settings. Typically, this implies that the integrator/propagator
    combination is not feasible. It is part of the assignment to figure out why this happens.

* One frequent error can be one of:
    "Error in RKF integrator, step size is NaN"
    "Error in ABM integrator, step size is NaN"
    "Error in BS integrator, step size is NaN"

This means that a variable time-step integrator wanting to take a NaN time step. In such cases, the selected
integrator settings are unsuitable for the problem you are considering.

NOTE: When any of the above errors occur, the propagation results up to the point of the crash can still be extracted
as normal. It can be checked whether any issues have occured by using the function

dynamics_simulator.integration_completed_successfully

which returns a boolean (false if any issues have occured)

* A frequent issue can be that a simulation with certain settings runs for too long (for instance if the time steo
becomes excessively small). To prevent this, you can add an additional termination setting (on top of the existing ones!)

    cpu_time_termination_settings = propagation_setup.propagator.cpu_time_termination(
        maximum_cpu_time )

where maximum_cpu_time is a varaiable (float) denoting the maximum time in seconds that your simulation is allowed to
run. If the simulation runs longer, it will terminate, and return the propagation results up to that point.

* Finally, if the following error occurs, you can NOT extract the results up to the point of the crash. Instead,
the program will immediately terminate

    SPICE(DAFNEGADDR) --

    Negative value for BEGIN address: -214731446

This means that a state is extracted from Spice at a time equal to NaN. Typically, this is indicative of a
variable time-step integrator wanting to take a NaN time step, and the issue not being caught by Tudat.
In such cases, the selected integrator settings are unsuitable for the problem you are considering.
"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################


# General imports
import numpy as np
from matplotlib import pyplot as plt
import os
import datetime


# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.astro import time_conversion
from tudatpy.util import result2array


# Problem-specific imports
import Utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

initial_time = time_conversion.calendar_date_to_julian_day(datetime.datetime(2028, 8, 6))-constants.JULIAN_DAY_ON_J2000
time_of_flight = 445

trajectory_parameters = [7454.4212962963,
                         320.5648148148,
                         0,
                         8661.95,
                         8425.79,
                         -8632.97,
                         5666.72,
                         -3567.68,
                         -2806.92]

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = False

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = 'C:/Users/hecto/Desktop/TU Delft/Thesis'

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 22
specific_impulse = 2500
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY
# Time at which to start propagation
initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                            time_buffer)

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Define settings for celestial bodies
bodies_to_create = ['Sun',
                    'Mercury',
                    'Venus',
                    'Earth',
                    'Moon',
                    'Mars',
                    'Jupiter',
                    'Saturn',
                    'Uranus',
                    'Neptune']
# Define coordinate system
global_frame_origin = 'SSB'
global_frame_orientation = 'ECLIPJ2000'
# Create body settings
body_settings = environment_setup.get_default_body_settings(bodies_to_create,
                                                            global_frame_origin,
                                                            global_frame_orientation)
# Create bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

# Create vehicle object and add it to the existing system of bodies
bodies.create_empty_body('Vehicle')
bodies.get_body('Vehicle').mass = vehicle_mass
thrust_magnitude_settings = (
    propagation_setup.thrust.custom_thrust_magnitude_fixed_isp(lambda time: 0.0, specific_impulse))
environment_setup.add_engine_model(
    'Vehicle', 'LowThrustEngine', thrust_magnitude_settings, bodies)
environment_setup.add_rotation_model(
    bodies, 'Vehicle', environment_setup.rotation_model.custom_inertial_direction_based(
        lambda time: np.array([1, 0, 0]), global_frame_orientation, 'VehicleFixed'))

# Create radiation pressure interface
reference_area_radiation = 0.6
radiation_pressure_coefficient = 1.2
occulting_bodies = []
radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
    "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
)

environment_setup.add_radiation_pressure_interface(
    bodies, "Vehicle", radiation_pressure_settings
)

###########################################################################
# CREATE PROPAGATOR SETTINGS ##############################################
###########################################################################

# Retrieve termination settings
termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                     minimum_mars_distance,
                                                     time_buffer)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

propagator_settings = Util.get_propagator_settings(
    trajectory_parameters,
    bodies,
    initial_propagation_time,
    vehicle_mass,
    termination_settings,
    dependent_variables_to_save,
    current_propagator=propagation_setup.propagator.cowell,
    model_choice=17,
    vinf=[[4000], [0], [0]])

propagator_settings.integrator_settings = Util.get_integrator_settings(
    0, 7, 1, initial_propagation_time)
# Propagate dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings)

benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

### OUTPUT OF THE SIMULATION ###
# Retrieve propagated state and dependent variables
# NOTE TO STUDENTS, the following retrieve the propagated states, converted to Cartesian states
state_history = dynamics_simulator.state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history

# Save results to a dictionary
simulation_results = [state_history, dependent_variable_history]

###########################################################################
# # WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
# ###########################################################################

# Create problem without propagating
hodographic_shaping_object = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                bodies, vinf=[[4000], [0], [0]])

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
else:
    output_path = None
# Retrieves analytical results and write them to a file
hodographic_state_history = Util.get_hodographic_trajectory(hodographic_shaping_object,
                                output_path)

hodographic_state_history_list = result2array(hodographic_state_history)
propagation_state_history_list = result2array(state_history)
dependent_var = result2array(dependent_variable_history)
time = dependent_var[:,0]
time_days = (time-time[0])/constants.JULIAN_DAY

plt.rc('font', size=9)  # controls default text size
plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

plt.figure(figsize=(width, height))
ax = plt.axes(projection='3d')

ax.plot3D(hodographic_state_history_list[:,1], hodographic_state_history_list[:,2], hodographic_state_history_list[:,3], label='Vehicle', linewidth=1.5, color='tab:red')
ax.plot3D(propagation_state_history_list[:,1], propagation_state_history_list[:,2], propagation_state_history_list[:,3], label='Vehicle', linewidth=1.5, color='tab:blue')
ax.plot3D(dependent_var[:,4], dependent_var[:,5], dependent_var[:,6], label='Earth', linewidth=0.8, color='tab:green')
ax.plot3D(dependent_var[:,7], dependent_var[:,8], dependent_var[:,9], label='Mars', linewidth=0.8, color='tab:orange')

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20
ax.legend(fontsize='small')
plt.grid(True)

sc_dist_earth = np.linalg.norm(dependent_var[:,4:7] - propagation_state_history_list[:,1:4], axis=1)

print('The initial distance to Earth is', (sc_dist_earth[0]), 'm')

sc_dist_mars = np.linalg.norm(dependent_var[:,7:10] - propagation_state_history_list[:,1:4], axis=1)

print('The final distance to Mars with Mars states from dependent variables is', (sc_dist_mars[-1]), 'm')

hodographic_interpolator = interpolators.create_one_dimensional_vector_interpolator(hodographic_state_history,
                                                                                  interpolators.lagrange_interpolation(
                                                                                      8))
propagation_difference = dict()
for epoch in state_history.keys():
    propagation_difference[epoch] = hodographic_interpolator.interpolate(epoch)[:6] - state_history[epoch][:6]

propagation_difference_list = result2array(propagation_difference)

benchmark_state_difference_norm = np.linalg.norm(propagation_difference_list[:,1:4],axis=1)

plt.figure(figsize=(width, height))
plt.plot((propagation_difference_list[:,0]-propagation_difference_list[0,0]) / constants.JULIAN_DAY, benchmark_state_difference_norm[:])
plt.xlabel('Time')
plt.ylabel('Benchmark difference [m]')
plt.grid(True)
plt.tight_layout()

sc_thrust = dependent_var[:, 11]
sc_acceleration = np.linalg.norm(
    dependent_var[:, 12:15] - dependent_var[:, 15:18] - dependent_var[:, 18:21],
    axis=1)

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_thrust)
plt.plot(time_days, sc_acceleration)
plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [m/s^2]')
plt.legend(
    ['Hodographic thrust profile', 'Accelerations on SC'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
    ncol=3, fancybox=True, shadow=True, fontsize='small')
plt.grid(True)
plt.yscale('log')
plt.tight_layout()

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_dist_earth)

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('SC distance to Earth [m]')
plt.grid(True)
plt.tight_layout()

sc_mass = propagation_state_history_list[:,7]

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_mass)

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('SC Mass [kg]')
plt.grid(True)
plt.tight_layout()

delta_v_total = 20*np.log(sc_mass[0]/sc_mass[-1])

print('The total delta-V for this maneuver is', delta_v_total, 'km/s')

max_acc = max(dependent_var[:,11])*1000

print('The maximum thrust acceleration is', max_acc, 'mN')

plt.show()
