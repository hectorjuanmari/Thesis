"""
Copyright (c) 2010-2022, Delft University of Technology
All rights reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

AE4866 Propagation and Optimization in Astrodynamics
Low Thrust
First name: ***COMPLETE HERE***
Last name: ***COMPLETE HERE***
Student number: ***COMPLETE HERE***

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

# import sys
# sys.path.insert(0, "/home/dominic/Tudat/tudat-bundle/tudat-bundle/cmake-build-default/tudatpy")

# General imports
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
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

initial_time = time_conversion.calendar_date_to_julian_day(
    datetime.datetime(2028, 11, 17)) - constants.JULIAN_DAY_ON_J2000
time_of_flight = 272

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
use_benchmark = 0
run_environment_analysis = 0

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput'

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
time_buffer = 40.0 * constants.JULIAN_DAY
# Time at which to start propagation
initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                            time_buffer)
###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

# Set number of models
number_of_models = 17

# Initialize dictionary to store the results of the simulation
simulation_results = dict()

# Set the interpolation step at which different runs are compared
output_interpolation_step = constants.JULIAN_DAY  # s

if run_environment_analysis:
    for model_test in range(number_of_models):
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
        # For case 4, the ephemeris of Jupiter is generated by solving the 2-body problem of the Sun and Jupiter
        # (in the other cases, the ephemeris of Jupiter from SPICE take into account all the perturbations)
        # if (model_test == 4):
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Jupiter')
        #     body_settings.get('Jupiter').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Jupiter', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 17:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Mercury')
        #     body_settings.get('Mercury').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Mercury', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 18:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Venus')
        #     body_settings.get('Venus').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Venus', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 19:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Earth')
        #     body_settings.get('Earth').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Earth', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 20:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Mars')
        #     body_settings.get('Mars').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Mars', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 21:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Jupiter')
        #     body_settings.get('Jupiter').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Jupiter', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 22:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Saturn')
        #     body_settings.get('Saturn').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Saturn', initial_propagation_time, effective_gravitational_parameter, 'Sun', global_frame_orientation)
        # if model_test == 23:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Uranus')
        #     body_settings.get('Uranus').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Uranus_BARYCENTER', initial_propagation_time, effective_gravitational_parameter, 'Sun',
        #         global_frame_orientation)
        # if model_test == 24:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Sun') + \
        #                                         spice_interface.get_body_gravitational_parameter('Neptune')
        #     body_settings.get('Neptune').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Neptune_BARYCENTER', initial_propagation_time, effective_gravitational_parameter, 'Sun',
        #         global_frame_orientation)
        # if model_test == 25:
        #     effective_gravitational_parameter = spice_interface.get_body_gravitational_parameter('Earth') + \
        #                                         spice_interface.get_body_gravitational_parameter('Moon')
        #     body_settings.get('Moon').ephemeris_settings = environment_setup.ephemeris.keplerian_from_spice(
        #         'Moon', initial_propagation_time, effective_gravitational_parameter, 'Earth', global_frame_orientation)

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
                lambda time: np.array([1, 0, 0]), global_frame_orientation, 'VehcleFixed'))

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

        # Create propagator settings for benchmark (Cowell)
        propagator_settings = Util.get_propagator_settings(
            trajectory_parameters,
            bodies,
            initial_propagation_time,
            vehicle_mass,
            termination_settings,
            dependent_variables_to_save,
            current_propagator=propagation_setup.propagator.cowell,
            model_choice=model_test)

        propagator_settings.integrator_settings = Util.get_integrator_settings(
            0, 7, 1, initial_propagation_time)
        # Propagate dynamics
        dynamics_simulator = numerical_simulation.create_dynamics_simulator(
            bodies, propagator_settings)

        ### OUTPUT OF THE SIMULATION ###
        # Retrieve propagated state and dependent variables
        # NOTE TO STUDENTS, the following retrieve the propagated states, converted to Cartesian states
        state_history = dynamics_simulator.state_history
        dependent_variable_history = dynamics_simulator.dependent_variable_history

        # Save results to a dictionary
        simulation_results[model_test] = [state_history, dependent_variable_history]

        # Get output path
        if model_test == 0:
            subdirectory = '/NominalCase/'
        else:
            subdirectory = '/Model_' + str(model_test) + '/'

        # Decide if output writing is required
        if write_results_to_file:
            output_path = current_dir + subdirectory
        else:
            output_path = None

        # If desired, write output to a file
        if write_results_to_file:
            save2txt(state_history, 'state_history.dat', output_path)
            save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)

    """
    NOTE TO STUDENTS
    The first index of the dictionary simulation_results refers to the model case, while the second index can be 0 (states)
    or 1 (dependent variables).
    You can use this dictionary to make all the cross-comparison that you deem necessary. The code below currently compares
    every case with respect to the "nominal" one.
    """
    # Compare all the model settings with the nominal case
    for model_test in range(1, number_of_models):
        # Get output path
        output_path = current_dir + '/Model_' + str(model_test) + '/'

        # Set time limits to avoid numerical issues at the boundaries due to the interpolation
        nominal_state_history = simulation_results[0][0]
        nominal_dependent_variable_history = simulation_results[0][1]
        nominal_times = list(nominal_state_history.keys())

        # Retrieve current state and dependent variable history
        current_state_history = simulation_results[model_test][0]
        current_dependent_variable_history = simulation_results[model_test][1]
        current_times = list(current_state_history.keys())

        # Get limit times at which both histories can be validly interpolated
        interpolation_lower_limit = max(nominal_times[3], current_times[3])
        interpolation_upper_limit = min(nominal_times[-3], current_times[-3])

        # Create vector of epochs to be compared (boundaries are referred to the first case)
        unfiltered_interpolation_epochs = np.arange(current_times[0], current_times[-1], output_interpolation_step)
        unfiltered_interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n <= interpolation_upper_limit]
        interpolation_epochs = [n for n in unfiltered_interpolation_epochs if n >= interpolation_lower_limit]

        nominal = 0
        if model_test == 11: nominal = 3
        elif model_test == 12: nominal = 4
        elif model_test == 13: nominal = 9
        elif model_test == 15: nominal = 3
        elif model_test == 16: nominal = 4
        # elif model_test == 17: nominal = 1
        # elif model_test == 18: nominal = 2
        # elif model_test == 19: nominal = 3
        # elif model_test == 20: nominal = 4
        # elif model_test == 21: nominal = 5
        # elif model_test == 22: nominal = 6
        # elif model_test == 23: nominal = 7
        # elif model_test == 24: nominal = 8
        # elif model_test == 25: nominal = 9

        interpolation_epochs = unfiltered_interpolation_epochs
        # Compare state history
        state_difference_wrt_nominal = Util.compare_models(current_state_history,
                                                           simulation_results[nominal][0],
                                                           interpolation_epochs,
                                                           output_path,
                                                           'state_difference_wrt_nominal_case.dat')
        # Compare dependent variable history
        dependent_variable_difference_wrt_nominal = Util.compare_models(current_dependent_variable_history,
                                                                        simulation_results[0][1],
                                                                        interpolation_epochs,
                                                                        output_path,
                                                                        'dependent_variable_difference_wrt_nominal_case.dat')

plt.rc('font', size=9)  # controls default text size
plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

number_of_models = 16

states_nominal = np.loadtxt("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/NominalCase/state_history.dat")

dependent_variables_case3 = np.loadtxt("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Model_3/dependent_variable_history.dat")



states_base = np.loadtxt("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Model_1/state_difference_wrt_nominal_case.dat")

states = np.zeros((np.append(states_base.shape, [number_of_models])))

for i in range(number_of_models):
    states[:, :, i] = np.loadtxt("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Model_" + str(i + 1) + "/state_difference_wrt_nominal_case.dat")

time = states[:, 0, 0]
time_days = (time - states_nominal[0, 0]) / constants.JULIAN_DAY

sc_dist = dependent_variables_case3[:,1]

print('Initial distance to Earth is', sc_dist[0], 'm')

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_dist)

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('SC distance to Mars [m]')
plt.grid(True)
plt.tight_layout()

sc_thrust = dependent_variables_case3[:,11]
sc_acceleration = np.linalg.norm(dependent_variables_case3[:,12:15] - dependent_variables_case3[:,15:18] - dependent_variables_case3[:,18:21],axis=1)

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
ax = plt.subplot()
for i in [2, 9, 10, 13]:
    plt.plot(time_days, np.linalg.norm(states[:, 1:4, i], axis=1))
    print(i)
plt.axhline(y=1e3, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.axhline(y=1e2, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.xlabel('Time [days]')
plt.ylabel('$||\epsilon_r||$ [m]')
plt.yscale('log')
ax.set_ymargin(0.2)
plt.grid(True, which="both", ls="--", linewidth=0.3)
plt.legend(
    ['Third bodies', 'Solar radiation pressure', 'Third body spherical harmonics', 'Relativistic effects',
     'Requirement'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
    ncol=3, fancybox=True, shadow=True, fontsize='small')
plt.tight_layout()

plt.figure(figsize=(width, height))
ax = plt.subplot()
for i in (range(9)):
    plt.plot(time_days, np.linalg.norm(states[:, 1:4, i], axis=1))
    print(i)
plt.axhline(y=1e3, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.axhline(y=1e2, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.xlabel('Time [days]')
plt.ylabel('$||\epsilon_r||$ [m]')
plt.yscale('log')
ax.set_ymargin(0.2)
plt.grid(True, which="both", ls="--", linewidth=0.3)
plt.legend(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Moon', 'Requirement'],
           loc='upper center',
           bbox_to_anchor=(0.5, 1.05),
           ncol=5, fancybox=True, shadow=True, fontsize='small')
plt.tight_layout()

plt.figure(figsize=(width, height))
ax = plt.subplot()
for i in [10, 11, 12]:
    plt.plot(time_days, np.linalg.norm(states[:, 1:4, i], axis=1))
    print(i)
plt.axhline(y=1e3, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.axhline(y=1e2, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.xlabel('Time [days]')
plt.ylabel('$||\epsilon_r||$ [m]')
plt.yscale('log')
ax.set_ymargin(0.1)
plt.grid(True, which="both", ls="--", linewidth=0.3)
plt.legend(['Earth J2', 'Mars J2', 'Moon J2', 'Requirement'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fancybox=True, shadow=True, fontsize='small')
plt.tight_layout()

# plt.figure(figsize=(width, height))
# ax = plt.subplot()
# for i in range(16, 25):
#     plt.plot(time_days, np.linalg.norm(states[:, 1:4, i], axis=1))
#     print(i)
# plt.axhline(y=1e3, color='tab:orange', linestyle='dashdot', linewidth=0.8)
# plt.axhline(y=1e2, color='tab:orange', linestyle='dashdot', linewidth=0.8)
# plt.xlabel('Time [days]')
# plt.ylabel('$||\epsilon_r||$ [m]')
# plt.yscale('log')
# ax.set_ymargin(0.2)
# plt.grid(True, which="both", ls="--", linewidth=0.3)
# plt.legend(['Mercury', 'Venus', 'Earth', 'Mars', 'Jupiter', 'Saturn', 'Uranus', 'Neptune', 'Moon', 'Requirement'],
#            loc='upper center',
#            bbox_to_anchor=(0.5, 1.05),
#            ncol=5, fancybox=True, shadow=True, fontsize='small')
# plt.tight_layout()

plt.figure(figsize=(width, height))
ax = plt.subplot()
for i in range(13, 16):
    plt.plot(time_days, np.linalg.norm(states[:, 1:4, i], axis=1))
    print(i)
plt.axhline(y=1e3, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.axhline(y=1e2, color='tab:orange', linestyle='dashdot', linewidth=0.8)
plt.xlabel('Time [days]')
plt.ylabel('$||\epsilon_r||$ [m]')
plt.yscale('log')
ax.set_ymargin(0.1)
plt.grid(True, which="both", ls="--", linewidth=0.3)
plt.legend(['Sun', 'Earth', 'Mars', 'Requirement'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
           ncol=4, fancybox=True, shadow=True, fontsize='small')
plt.tight_layout()

# plt.figure(figsize=(8,8))
# for i in range(33,36):
#     if i==33:plt.plot(time_days, np.linalg.norm(states[:,1:4,i] , axis = 1 ), linewidth=2.5)
#     print(i)
# plt.axhline(y = 1e3, color = 'tab:orange', linestyle = 'dashdot', linewidth = 0.8)
# plt.axhline(y = 1e2, color = 'tab:orange', linestyle = 'dashdot', linewidth = 0.8)
# plt.xlabel( 'Time [days]' )
# plt.ylabel('$||\epsilon_r||$ [m]')
# plt.yscale('log')
# plt.grid(True, which="both", ls="--", linewidth = 0.3)
# plt.legend(['Sun', 'Earth', 'Mars'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True, fontsize='x-small')
# plt.tight_layout()

# plt.figure(figsize=(8,8))
# for i in range(37,41):
#     plt.plot(time_days, np.linalg.norm(states[:,1:4,i] , axis = 1 ), linewidth=2.5)
#     print(i)
# plt.axhline(y = 1e3, color = 'tab:orange', linestyle = 'dashdot', linewidth = 0.8)
# plt.axhline(y = 1e2, color = 'tab:orange', linestyle = 'dashdot', linewidth = 0.8)
# plt.xlabel( 'Time [days]' )
# plt.ylabel('$||\epsilon_r||$ [m]')
# plt.yscale('log')
# plt.grid(True, which="both", ls="--", linewidth = 0.3)
# plt.legend(['Sun J4','Earth J22','Earth J4','Mars J22'],loc='upper center', bbox_to_anchor=(0.5, 1.05),
#           ncol=3, fancybox=True, shadow=True, fontsize='x-small')
# plt.tight_layout()

plt.show()
