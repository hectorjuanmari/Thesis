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


# General imports
import numpy as np
import os

# Tudatpy imports
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators

# Problem-specific imports
import LowThrustUtilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()
# NOTE TO STUDENTS: INPUT YOUR PARAMETER SET HERE, FROM THE INPUT FILES
# ON BRIGHTSPACE, FOR YOUR SPECIFIC STUDENT NUMBER.
trajectory_parameters = [570727221.2273525 / constants.JULIAN_DAY,
                         37073942.58665284 / constants.JULIAN_DAY,
                         0,
                         2471.19649906354,
                         4207.587982407276,
                         -5594.040587888714,
                         8748.139268525232,
                         -3449.838496679572,
                         0.0]

# Choose whether benchmark is run
use_benchmark = True
run_integrator_analysis = True

# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 4.0E3
specific_impulse = 3000.0
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 30.0 * constants.JULIAN_DAY
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

# NOTE TO STUDENTS: MODIFY THE CODE INSIDE THIS "IF STATEMENT" (AND CALLED FUNCTIONS, IF NEEDED)
# TO ASSESS THE QUALITY OF VARIOUS BENCHMARK SETTINGS
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
    number_of_propagators = len(available_propagators)
    number_of_integrators = 5

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
            if integrator_index > 3:
                number_of_integrator_step_size_settings = 6
            else:
                number_of_integrator_step_size_settings = 4

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