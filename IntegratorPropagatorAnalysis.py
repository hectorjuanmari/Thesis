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

trajectory_parameters = [initial_time,
                         time_of_flight,
                         0,
                         8661.95,
                         8425.79,
                         -8632.97,
                         5666.72,
                         -3567.68,
                         -2806.92]

# trajectory_parameters = [7454.4212962963,
#                          320.5648148148,
#                          0,
#                          8661.95,
#                          8425.79,
#                          -8632.97,
#                          5666.72,
#                          -3567.68,
#                          -2806.92]

# trajectory_parameters = [303.3622685185,	385.03125,	0,	-999.466,	-6807.06,	-3851.67,	1519.75,	7624.45,	9524.28]

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = False
run_plots = True

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
time_buffer = 0.0 * constants.JULIAN_DAY
# Time at which to start propagation
initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                            time_buffer)
###########################################################################
# CREATE ENVIRONMENT ######################################################
###########################################################################

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

###########################################################################
# IF DESIRED, GENERATE AND COMPARE BENCHMARKS #############################
###########################################################################

if use_benchmark:
    # Define benchmark interpolator settings to make a comparison between the two benchmarks
    benchmark_interpolator_settings = interpolators.lagrange_interpolation(
        8, boundary_interpolation=interpolators.extrapolate_at_boundary)

    # Create propagator settings for benchmark (Cowell)
    propagator_settings = Util.get_propagator_settings(
        trajectory_parameters,
        bodies,
        initial_propagation_time,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save)

    benchmark_output_path = current_dir + '/SimulationOutput/benchmarks/' if write_results_to_file else None

    # Generate benchmarks
    benchmark_step_size = 86400.0
    benchmark_list = Util.generate_benchmarks(benchmark_step_size,
                                              initial_propagation_time,
                                              bodies,
                                              propagator_settings,
                                              are_dependent_variables_to_save,
                                              benchmark_output_path)
    # Extract benchmark states
    first_benchmark_state_history = benchmark_list[0]
    second_benchmark_state_history = benchmark_list[1]
    # Create state interpolator for first benchmark
    benchmark_state_interpolator = interpolators.create_one_dimensional_vector_interpolator(
        first_benchmark_state_history,
        benchmark_interpolator_settings)

    # Compare benchmark states, returning interpolator of the first benchmark
    benchmark_state_difference = Util.compare_benchmarks(first_benchmark_state_history,
                                                         second_benchmark_state_history,
                                                         benchmark_output_path,
                                                         'benchmarks_state_difference.dat')

    # Extract benchmark dependent variables, if present
    if are_dependent_variables_to_save:
        first_benchmark_dependent_variable_history = benchmark_list[2]
        second_benchmark_dependent_variable_history = benchmark_list[3]
        # Create dependent variable interpolator for first benchmark
        benchmark_dependent_variable_interpolator = interpolators.create_one_dimensional_vector_interpolator(
            first_benchmark_dependent_variable_history,
            benchmark_interpolator_settings)

        # Compare benchmark dependent variables, returning interpolator of the first benchmark, if present
        benchmark_dependent_difference = Util.compare_benchmarks(first_benchmark_dependent_variable_history,
                                                                 second_benchmark_dependent_variable_history,
                                                                 benchmark_output_path,
                                                                 'benchmarks_dependent_variable_difference.dat')

###########################################################################
# # WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
# ###########################################################################

# Create problem without propagating
hodographic_shaping_object = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                bodies)

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
else:
    output_path = None
# Retrieves analytical results and write them to a file
Util.get_hodographic_trajectory(hodographic_shaping_object,
                                output_path)

###########################################################################
# RUN SIMULATION FOR VARIOUS SETTINGS #####################################
###########################################################################
"""
Code below propagates states using each propagator (propagator_index=0..6), four multi-stage variable step-size
integrators (integrator_index j=0..3) and an RK4 integrator (j=4). For the variable-step integrators, 4 different
tolerances are used (step_size_index=0..3). For the RK4, 6 different step sizes are used (step_size_index=0..5),
see use of number_of_integrator_step_size_settings variable. See get_integrator_settings function for more details.

For each combination of i, j, and k, results are written to directory:
    LunarAscent/SimulationOutput/prop_i/int_j/setting_k/

Specifically:
     state_History.dat                                  Cartesian states as function of time
     dependent_variable_history.dat                     Dependent variables as function of time
     state_difference_wrt_benchmark.dat                 Difference of dependent variables w.r.t. benchmark
     dependent_variable_difference_wrt_benchmark.dat    Difference of states w.r.t. benchmark
     ancillary_simulation_info.dat                      Other information about the propagation (number of function
                                                        evaluations, etc...)

NOTE TO STUDENTS: THE NUMBER, TYPES, SETTINGS OF PROPAGATORS/INTEGRATORS/INTEGRATOR STEPS,TOLERANCES,ETC. SHOULD BE
MODIFIED FOR ASSIGNMENT 1, BOTH IN THIS FILE, AND IN FUNCTIONS CALLED BY THIS FILE (MAINLY, BUT NOT NECESSARILY
EXCLUSIVELY, THE get_integrator_settings FUNCTION)
"""
if run_integrator_analysis:

    # Define list of propagators
    available_propagators = [propagation_setup.propagator.cowell,
                             propagation_setup.propagator.encke,
                             propagation_setup.propagator.gauss_keplerian,
                             propagation_setup.propagator.gauss_modified_equinoctial,
                             propagation_setup.propagator.unified_state_model_quaternions,
                             propagation_setup.propagator.unified_state_model_modified_rodrigues_parameters,
                             propagation_setup.propagator.unified_state_model_exponential_map]
    # Define settings to loop over
    number_of_propagators = 6
    number_of_integrators = 16

    # Loop over propagators
    for propagator_index in range(number_of_propagators):
        # Get current propagator, and define translational state propagation settings
        current_propagator = available_propagators[propagator_index]

        # Define propagation settings
        current_propagator_settings = Util.get_propagator_settings(
            trajectory_parameters,
            bodies,
            initial_propagation_time,
            vehicle_mass,
            termination_settings,
            dependent_variables_to_save,
            current_propagator)

        # Loop over different integrators
        for integrator_index in range(number_of_integrators):
            # For RK4, more step sizes are used. NOTE TO STUDENTS, MODIFY THESE AS YOU SEE FIT!
            if integrator_index == 8:
                number_of_integrator_step_size_settings = 3
            else:
                number_of_integrator_step_size_settings = 5
            # Loop over all tolerances / step sizes
            for step_size_index in range(number_of_integrator_step_size_settings):
                # Print status
                to_print = 'Current run: \n propagator_index = ' + str(propagator_index) + \
                           '\n integrator_index = ' + str(integrator_index) \
                           + '\n step_size_index = ' + str(step_size_index)
                print(to_print)
                # Set output path
                output_path = current_dir + '/SimulationOutput/prop_' + str(propagator_index) + \
                              '/int_' + str(integrator_index) + '/step_size_' + str(step_size_index) + '/'
                # Create integrator settings
                current_integrator_settings = Util.get_integrator_settings(propagator_index,
                                                                           integrator_index,
                                                                           step_size_index,
                                                                           initial_propagation_time)
                current_propagator_settings.integrator_settings = current_integrator_settings

                # Propagate dynamics
                dynamics_simulator = numerical_simulation.create_dynamics_simulator(
                    bodies, current_propagator_settings)

                ### OUTPUT OF THE SIMULATION ###
                # Retrieve propagated state and dependent variables
                # NOTE TO STUDENTS, the following retrieve the propagated states, converted to Cartesian states
                state_history = dynamics_simulator.state_history
                unprocessed_state_history = dynamics_simulator.unprocessed_state_history
                dependent_variable_history = dynamics_simulator.dependent_variable_history

                # Get the number of function evaluations (for comparison of different integrators)
                function_evaluation_dict = dynamics_simulator.cumulative_number_of_function_evaluations
                number_of_function_evaluations = list(function_evaluation_dict.values())[-1]
                # Add it to a dictionary
                dict_to_write = {
                    'Number of function evaluations (ignore the line above)': number_of_function_evaluations}
                save2txt(function_evaluation_dict, 'neval.dat', output_path)
                # Check if the propagation was run successfully
                propagation_outcome = dynamics_simulator.integration_completed_successfully
                dict_to_write['Propagation run successfully'] = propagation_outcome
                # Note if results were written to files
                dict_to_write['Results written to file'] = write_results_to_file
                # Note if benchmark was run
                dict_to_write['Benchmark run'] = use_benchmark
                # Note if dependent variables were present
                dict_to_write['Dependent variables present'] = are_dependent_variables_to_save

                # Save results to a file
                if write_results_to_file:
                    save2txt(state_history, 'state_history.dat', output_path)
                    save2txt(dependent_variable_history, 'dependent_variable_history.dat', output_path)
                    save2txt(dict_to_write, 'ancillary_simulation_info.txt', output_path)

                # Compare the simulation to the benchmarks and write differences to files
                if use_benchmark:
                    # Initialize containers
                    state_difference = dict()
                    # Loop over the propagated states and use the benchmark interpolators
                    # NOTE TO STUDENTS: it can happen that the benchmark ends earlier than the regular simulation, due to
                    # the shorter step size. Therefore, the following lines of code will be forced to extrapolate the
                    # benchmark states (or dependent variables), producing a warning. Be aware of it!
                    for epoch in state_history.keys():
                        state_difference[epoch] = state_history[epoch] - benchmark_state_interpolator.interpolate(epoch)
                    state_difference_list = result2array(state_difference)
                    max_error = dict()
                    # position_difference = np.linalg.norm(state_difference_list[:,1:4],axis=1) max_error[0] = max(
                    # position_difference[:]) max_error_interpolator =
                    # interpolators.create_one_dimensional_vector_interpolator(state_difference,
                    # interpolators.lagrange_interpolation( 8)) max_error[0] = np.linalg.norm(
                    # max_error_interpolator.interpolate(list(state_difference.keys())[-1] - 30 * 86400 * 2)[0:3])

                    # lim = -1
                    # while abs(np.linalg.norm(state_difference_list[lim][1:4]) - np.linalg.norm(state_difference_list[lim - 1][1:4])) > 100:
                    #     lim -= 1
                    # max_error[0] = np.linalg.norm(state_difference_list[lim][1:4])

                    limit_epoch = list(first_benchmark_state_history.keys())[-16]
                    max_error_interpolator = interpolators.create_one_dimensional_vector_interpolator(state_difference,
                                                                                                      interpolators.lagrange_interpolation(
                                                                                                          2))
                    aux = np.linalg.norm(max_error_interpolator.interpolate(limit_epoch)[0:3])
                    max_error[0] = aux
                    save2txt(max_error, 'max_error.dat', output_path)
                    # Write differences with respect to the benchmarks to files
                    if write_results_to_file:
                        save2txt(state_difference, 'state_difference_wrt_benchmark.dat', output_path)
                    # Do the same for dependent variables, if present
                    if are_dependent_variables_to_save:
                        # Initialize containers
                        dependent_difference = dict()
                        # Loop over the propagated dependent variables and use the benchmark interpolators
                        for epoch in dependent_variable_history.keys():
                            dependent_difference[epoch] = dependent_variable_history[
                                                              epoch] - benchmark_dependent_variable_interpolator.interpolate(
                                epoch)
                        # Write differences with respect to the benchmarks to files
                        if write_results_to_file:
                            save2txt(dependent_difference, 'dependent_variable_difference_wrt_benchmark.dat',
                                     output_path)

    # Print the ancillary information
    print('\n### ANCILLARY SIMULATION INFORMATION ###')
    for (elem, (info, result)) in enumerate(dict_to_write.items()):
        if elem > 1:
            print(info + ': ' + str(result))

if run_plots:

    plt.rc('font', size=9)  # controls default text size
    plt.rc('font', family='serif')
    # plt.rc('text', usetex=True)

    fig_width_pt = 478 * 1

    inches_per_pt = 1 / 72.27

    golden_ratio = (5 ** .5 - 1) / 2

    width = fig_width_pt * inches_per_pt

    height = width * golden_ratio

    states1 = np.loadtxt('C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_0/int_0/step_size_0/state_difference_wrt_benchmark.dat')

    nint = 16
    nprop = 6
    nset = 5
    max_error = np.zeros((2, nset, nint))

    for i in range(nint):
        nset = 5
        if i == 8:
            nset = 3
        for j in range(nset):
            path = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_0/int_' + str(i) + '/step_size_' + str(
                j) + '/state_difference_wrt_benchmark.dat'
            path2 = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_0/int_' + str(i) + '/step_size_' + str(j) + '/neval.dat'
            path3 = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_0/int_' + str(i) + '/step_size_' + str(j) + '/max_error.dat'
            states = np.loadtxt(path)
            ancillary = np.loadtxt(path2)
            max_error[0, j, i] = ancillary[-1, 1]
            max_error[1, j, i] = np.loadtxt(path3)[1]

    plt.figure(figsize=(width, height))
    plt.xlabel('Number of evaluations')
    plt.xscale('log')
    plt.ylabel('Maximum position error [m]')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.3)
    plt.plot(max_error[0, :, 0], max_error[1, :, 0], marker='o')
    plt.plot(max_error[0, :, 1], max_error[1, :, 1], marker='v')
    plt.plot(max_error[0, :, 2], max_error[1, :, 2], marker='^')
    plt.plot(max_error[0, :, 3], max_error[1, :, 3], marker='<')
    plt.plot(max_error[0, :, 4], max_error[1, :, 4], marker='>')
    plt.plot(max_error[0, :, 5], max_error[1, :, 5], marker='s')
    plt.plot(max_error[0, :, 6], max_error[1, :, 6], marker='p')
    plt.plot(max_error[0, :, 7], max_error[1, :, 7], marker='*')
    plt.axhline(y=10, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.axhline(y=1, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.legend(['RKF4(5)', 'RKF5(6)', 'RKF7(8)', 'RKDP8(7)', 'RK4', 'RK5', 'RK7', 'RK8'], fontsize='small')
    plt.tight_layout()

    plt.figure(figsize=(width, height))
    plt.xlabel('Number of evaluations')
    plt.xscale('log')
    plt.ylabel('Maximum position error [m]')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.3)
    plt.plot(max_error[0, :3, 8], max_error[1, :3, 8], marker='o')
    plt.plot(max_error[0, :, 9], max_error[1, :, 9], marker='v')
    plt.plot(max_error[0, :, 10], max_error[1, :, 10], marker='^')
    plt.plot(max_error[0, :, 11], max_error[1, :, 11], marker='<')
    plt.plot(max_error[0, :, 12], max_error[1, :, 12], marker='>')
    plt.plot(max_error[0, :, 13], max_error[1, :, 13], marker='s')
    plt.plot(max_error[0, :, 14], max_error[1, :, 14], marker='p')
    plt.plot(max_error[0, :, 15], max_error[1, :, 15], marker='*')
    plt.axhline(y=10, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.axhline(y=1, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.legend(
        ['ABM var. order, var. $\Delta t$', 'ABM3 var. $\Delta t$', 'ABM4 var. $\Delta t$', 'ABM5 var. $\Delta t$',
         'ABM4 fix. $\Delta t$', 'ABM5 fix. $\Delta t$', 'ABM6 fix. $\Delta t$', 'ABM7 fix. $\Delta t$'],
        fontsize='small')
    plt.tight_layout()

    # plt.figure(figsize=(width, height))
    # plt.xlabel('Number of evaluations')
    # plt.xscale('log')
    # plt.ylabel('Maximum position error [m]')
    # plt.yscale('log')
    # plt.grid(True, which="major", ls="--", linewidth=0.3)
    # plt.plot(max_error[0, :, 16], max_error[1, :, 16], marker='>')
    # plt.plot(max_error[0, :, 17], max_error[1, :, 17], marker='s')
    # plt.plot(max_error[0, :, 18], max_error[1, :, 18], marker='p')
    # plt.plot(max_error[0, :, 19], max_error[1, :, 19], marker='*')
    # plt.legend(['BS4', 'BS5', 'BS6', 'BS7'])
    # plt.tight_layout()

    max_error = np.zeros((2, nset, nprop))

    for i in range(nprop):
        for j in range(nset):
            path = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_' + str(i) + '/int_7/step_size_' + str(
                j) + '/state_difference_wrt_benchmark.dat'
            path2 = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_' + str(i) + '/int_7/step_size_' + str(j) + '/neval.dat'
            path3 = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_' + str(i) + '/int_7/step_size_' + str(j) + '/max_error.dat'
            states = np.loadtxt(path)
            ancillary = np.loadtxt(path2)
            max_error[0, j, i] = ancillary[-1, 1]
            max_error[1, j, i] = np.loadtxt(path3)[1]

    plt.figure(figsize=(width, height))
    plt.xlabel('Time [days]')
    plt.ylabel('$||\epsilon_r||$ [m]')
    plt.yscale('log')
    # plt.xlim([margin*bm_time_step, (states[-1,0]-states[0,0]-margin*86400*bm_time_step)/constants.JULIAN_DAY])
    plt.grid(True, which="both", ls="--", linewidth=0.3)
    for i in range(nprop):
        path = 'C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/prop_' + str(i) + '/int_7/step_size_1/state_difference_wrt_benchmark.dat'
        states = np.loadtxt(path)
        lim_1 = 0
        lim_2 = -1
        while abs(np.linalg.norm(states[lim_1][1:4]) - np.linalg.norm(states[lim_1 + 1][1:4])) > 100:
            lim_1 += 1
        while abs(np.linalg.norm(states[lim_2][1:4]) - np.linalg.norm(states[lim_2 - 1][1:4])) > 100:
            lim_2 -= 1
        plt.plot((states[lim_1:lim_2, 0] - states[0, 0]) / constants.JULIAN_DAY,
                 np.linalg.norm(states[lim_1:lim_2, 1:4], axis=1))
        # plt.xlim([margin*bm_time_step, (states[-1,0]-states[0,0]-margin*86400*bm_time_step)/constants.JULIAN_DAY])
    plt.axhline(y=10, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.axhline(y=1, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.legend(['Cowell', 'Encke', 'Gauss Keplerian', 'Gauss Modified Equinoctial', 'USM Quaternions',
                'USM Rodrigues Parameters', 'USM Exponential Map'], loc='best', fontsize='small')
    plt.tight_layout()

    max_error[1, 1, 0] = max_error[1, 0, 0]

    plt.figure(figsize=(width, height))
    plt.xlabel('Number of evaluations')
    plt.xscale('log')
    plt.ylabel('Maximum position error [m]')
    plt.yscale('log')
    plt.grid(True, which="both", ls="--", linewidth=0.3)
    plt.plot(max_error[0, :, 0], max_error[1, :, 0], marker='o')
    plt.plot(max_error[0, :, 1], max_error[1, :, 1], marker='v')
    plt.plot(max_error[0, :, 2], max_error[1, :, 2], marker='^')
    plt.plot(max_error[0, :, 3], max_error[1, :, 3], marker='<')
    plt.plot(max_error[0, :, 4], max_error[1, :, 4], marker='>')
    plt.plot(max_error[0, :, 5], max_error[1, :, 5], marker='s')
    # plt.plot(max_error[0, :, 6], max_error[1, :, 6], marker='p')
    plt.axhline(y=10, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.axhline(y=1, color='tab:orange', linestyle='dashdot', linewidth=0.8)
    plt.legend(['Cowell', 'Encke', 'Gauss Keplerian', 'Gauss Modified Equinoctial', 'USM Quaternions',
                'USM Rodrigues Parameters'], loc='best', fontsize='small')
    plt.tight_layout()

    nprop = 3
    nint = 4
    nset = 2

    integrators = [3, 7, 8, 15]

    propagators = [0, 1, 4]

    max_error = np.zeros((2, nset, nprop, nint))

    # plt.figure(figsize=(width, height*1.2))
    #
    # for i in range(nint):
    #     ax = plt.subplot(2, 2, i+1)
    #     ax.set_xlabel('Number of evaluations')
    #     ax.set_xscale('log')
    #     ax.set_ylabel('Maximum position error [m]')
    #     ax.set_yscale('log')
    #     ax.set_xmargin(0.4)
    #     ax.set_ymargin(0.4)
    #     ax.grid(True, which="both", ls="--", linewidth=0.3)
    #     for j in range(nprop):
    #         for k in range(nset):
    #             path = './SimulationOutput/prop_' + str(propagators[j]) + '/int_' + str(integrators[i]) + '/step_size_' + str(
    #                 k) + '/state_difference_wrt_benchmark.dat'
    #             path2 = './SimulationOutput/prop_' + str(propagators[j]) + '/int_' + str(integrators[i]) + '/step_size_' + str(
    #                 k) + '/neval.dat'
    #             path3 = './SimulationOutput/prop_' + str(propagators[j]) + '/int_' + str(integrators[i]) + '/step_size_' + str(
    #                 k) + '/max_error.dat'
    #             states = np.loadtxt(path)
    #             ancillary = np.loadtxt(path2)
    #             max_error[0, k, j, i] = ancillary[-1, 1]
    #             max_error[1, k, j, i] = np.loadtxt(path3)[1]
    #             # if i == 3: max_error[1, k, j, i] = np.linalg.norm(states[-3, 1:4])
    #     ax.scatter(max_error[0, 0, 0, i], max_error[1, 0, 0, i], color='m', marker='o', s=50)
    #     ax.scatter(max_error[0, 0, 1, i], max_error[1, 0, 1, i], color='c', marker='o', s=50)
    #     ax.scatter(max_error[0, 0, 2, i], max_error[1, 0, 2, i], color='y', marker='o', s=50)
    #     ax.scatter(max_error[0, 1, 0, i], max_error[1, 1, 0, i], color='m', marker='s', s=50)
    #     ax.scatter(max_error[0, 1, 1, i], max_error[1, 1, 1, i], color='c', marker='s', s=50)
    #     ax.scatter(max_error[0, 1, 2, i], max_error[1, 1, 2, i], color='y', marker='s', s=50)
    #     m_patch = mpatches.Patch(color='m', label='Cowell')
    #     c_patch = mpatches.Patch(color='c', label='Encke')
    #     y_patch = mpatches.Patch(color='y', label='USM Q.')
    #
    #     ax.legend(handles=[m_patch, c_patch, y_patch], fontsize='small')
    #     plt.tight_layout()
    # plt.legend(['Cowell', 'USM Q', 'USM EM'])

    plt.show()
