'''
Copyright (c) 2010-2021, Delft University of Technology
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

This module defines useful functions that will be called by the main script, where the optimization is executed.
'''

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np

# Tudatpy imports
import tudatpy
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.math import interpolators
from tudatpy.kernel.trajectory_design import shape_based_thrust
from tudatpy.kernel.trajectory_design import transfer_trajectory
from tudatpy.kernel.numerical_simulation import environment_setup


###########################################################################
# USEFUL FUNCTIONS ########################################################
###########################################################################


def get_termination_settings(trajectory_parameters,
                             minimum_mars_distance: float,
                             time_buffer: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings:
    """
    Get the termination settings for the simulation.

    Termination settings currently include:
    - simulation time (propagation stops if it is greater than the one provided by the hodographic trajectory)
    - distance to Mars (propagation stops if the relative distance is lower than the target distance)

    Parameters
    ----------
    trajectory_parameters : list[floats]
        List of trajectory parameters.
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.

    Returns
    -------
    hybrid_termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object.
    """
    # Create single PropagationTerminationSettings objects
    # Time
    final_time = get_trajectory_final_time(trajectory_parameters,
                                           time_buffer)
    time_termination_settings = propagation_setup.propagator.time_termination(
        final_time,
        terminate_exactly_on_final_condition=False)
    # Altitude
    relative_distance_termination_settings = propagation_setup.propagator.dependent_variable_termination(
        dependent_variable_settings=propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
        limit_value=minimum_mars_distance,
        use_as_lower_limit=True,
        terminate_exactly_on_final_condition=False)
    # Define list of termination settings
    termination_settings_list = [time_termination_settings,
                                 relative_distance_termination_settings]
    # Create termination settings object
    hybrid_termination_settings = propagation_setup.propagator.hybrid_termination(termination_settings_list,
                                                                                  fulfill_single_condition=True)
    return hybrid_termination_settings


# NOTE TO STUDENTS: this function can be modified to save more/less dependent variables.
def get_dependent_variable_save_settings() -> list:
    """
    Retrieves the dependent variables to save.

    Currently, the dependent variables saved include the relative distance between the spacecraft and:
    - Earth
    - the Sun
    - Mars

    Parameters
    ----------
    none

    Returns
    -------
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    """
    dependent_variables_to_save = [propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
                                   propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
                                   propagation_setup.dependent_variable.relative_position("Mars", "Sun"),
                                   propagation_setup.dependent_variable.body_mass("Vehicle")]
    return dependent_variables_to_save


# NOTE TO STUDENTS: THIS FUNCTION SHOULD BE EXTENDED TO USE MORE INTEGRATORS FOR ASSIGNMENT 1.
def get_integrator_settings(propagator_index: int,
                            integrator_index: int,
                            settings_index: int,
                            simulation_start_epoch: float) \
        -> tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings:
    """

    Retrieves the integrator settings.

    It selects a combination of integrator to be used (first argument) and
    the related setting (tolerance for variable step size integrators
    or step size for fixed step size integrators). The code, as provided, runs the following:
    - if j=0,1,2,3: a variable-step-size, multi-stage integrator is used (see multiStageTypes list for specific type),
                     with tolerances 10^(-10+*k)
    - if j=4      : a fixed-step-size RK4 integrator is used, with step-size 7200*2^(k)

    Parameters
    ----------
    propagator_index : int
        Index that selects the propagator type (currently not used).
        NOTE TO STUDENTS: this argument can be used to select specific combinations of propagator and integrators
        (provided that the code is expanded).
    integrator_index : int
        Index that selects the integrator type as follows:
            0 -> RK4(5)
            1 -> RK5(6)
            2 -> RK7(8)
            3 -> RKDP7(8)
            4 -> RK4
    settings_index : int
        Index that selects the tolerance or the step size
        (depending on the integrator type).
    simulation_start_epoch : float
        Start of the simulation [s] with t=0 at J2000.

    Returns
    -------
    integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
        Integrator settings to be provided to the dynamics simulator.

    """
    # Define list of multi-stage integrators
    multi_stage_integrators = [propagation_setup.integrator.RKCoefficientSets.rkf_45,
                               propagation_setup.integrator.RKCoefficientSets.rkf_56,
                               propagation_setup.integrator.RKCoefficientSets.rkf_78,
                               propagation_setup.integrator.RKCoefficientSets.rkdp_87]

    # Use variable step-size integrator
    if integrator_index < 4:
        # Select variable-step integrator
        current_coefficient_set = multi_stage_integrators[integrator_index]
        # Compute current tolerance
        current_tolerance = 10.0 ** (-14.0 + settings_index)
        # Create integrator settings
        integrator = propagation_setup.integrator
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        8640.0,
                                                                        current_coefficient_set,
                                                                        np.finfo(float).eps,
                                                                        np.inf,
                                                                        current_tolerance,
                                                                        current_tolerance)
    # Use fixed step-size integrator
    elif integrator_index >= 4 and integrator_index < 8:
        # Compute time step
        fixed_step_size = 86400 / 4 * 2.0 ** (settings_index)
        # Select variable-step integrator
        current_coefficient_set = multi_stage_integrators[integrator_index - 4]
        # Create integrator settings
        integrator = propagation_setup.integrator
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.runge_kutta_variable_step_size(simulation_start_epoch,
                                                                        fixed_step_size,
                                                                        current_coefficient_set,
                                                                        fixed_step_size,
                                                                        fixed_step_size,
                                                                        np.inf,
                                                                        np.inf)

    elif integrator_index == 8:
        # Create integrator settings
        integrator = propagation_setup.integrator
        current_tolerance = 10.0 ** (-11.0 + settings_index)
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                 20.0,
                                                                 10 ** -15,
                                                                 np.inf,
                                                                 current_tolerance,
                                                                 current_tolerance,
                                                                 6,
                                                                 11)

    elif integrator_index >= 9 and integrator_index < 12:
        # Create integrator settings
        integrator = propagation_setup.integrator
        current_tolerance = 10.0 ** (-11.0 + settings_index)
        if integrator_index == 9:
            current_order = 3
        elif integrator_index == 10:
            current_order = 4
        elif integrator_index == 11:
            current_order = 5

        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                 100.0,
                                                                 np.finfo(float).eps,
                                                                 np.inf,
                                                                 current_tolerance,
                                                                 current_tolerance,
                                                                 current_order,
                                                                 current_order)

    elif integrator_index >= 12 and integrator_index < 16:
        # Create integrator settings
        integrator = propagation_setup.integrator
        fixed_step_size = 86400.0 / 4 * 2 ** (settings_index)
        current_order = integrator_index - 8
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.adams_bashforth_moulton(simulation_start_epoch,
                                                                 fixed_step_size,
                                                                 fixed_step_size,
                                                                 fixed_step_size,
                                                                 np.inf,
                                                                 np.inf,
                                                                 current_order,
                                                                 current_order)

    elif integrator_index >= 16 and integrator_index < 20:
        # Create integrator settings
        integrator = propagation_setup.integrator
        current_tolerance = 10.0 ** (-14.0 + settings_index)
        current_order = integrator_index - 12
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.bulirsch_stoer(simulation_start_epoch,
                                                        100,
                                                        propagation_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
                                                        current_order,
                                                        10 ** -15,
                                                        np.inf,
                                                        current_tolerance,
                                                        current_tolerance)

    return integrator_settings


def get_propagator_settings(trajectory_parameters,
                            bodies,
                            initial_propagation_time,
                            constant_specific_impulse,
                            vehicle_initial_mass,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator=propagation_setup.propagator.cowell):
    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']
    # Create radiation pressure interface
    reference_area_radiation = 0.2*0.2
    radiation_pressure_coefficient = 1.2
    occulting_bodies = []
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient, occulting_bodies
    )

    environment_setup.add_radiation_pressure_interface(
        bodies, "Vehicle", radiation_pressure_settings
    )
    # Retrieve thrust acceleration
    thrust_settings = get_hodograph_thrust_acceleration_settings(trajectory_parameters,
                                                                 bodies,
                                                                 constant_specific_impulse)
    # Define accelerations acting on capsule
    acceleration_settings_on_vehicle = {
        'Sun': [propagation_setup.acceleration.point_mass_gravity(),
                #        propagation_setup.acceleration.cannonball_radiation_pressure(),
                ],
        # 'Earth': [propagation_setup.acceleration.point_mass_gravity()],
        # 'Mars': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [thrust_settings]
    }
    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = get_hodograph_state_at_epoch(trajectory_parameters,
                                                 bodies,
                                                 initial_propagation_time)

    # Create propagation settings for the benchmark
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        current_propagator,
        output_variables=dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_vehicle,
                                                                 acceleration_models)
    # Create mass propagator settings (same for all propagations)
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([vehicle_initial_mass]),
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 termination_settings,
                                                                 dependent_variables_to_save)

    return propagator_settings


###########################################################################
# HODOGRAPH-SPECIFIC FUNCTIONS ############################################
###########################################################################


def get_trajectory_time_of_flight(trajectory_parameters: list) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.

    Returns
    -------
    float
        Time of flight [s].
    """
    return trajectory_parameters[1] * constants.JULIAN_DAY


def get_trajectory_initial_time(trajectory_parameters: list,
                                buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Initial time of the hodographic trajectory [s].
    """
    return trajectory_parameters[0] * constants.JULIAN_DAY + buffer_time


def get_trajectory_final_time(trajectory_parameters: list,
                              buffer_time: float = 0.0) -> float:
    """
    Returns the time of flight in seconds.

    Parameters
    ----------
    trajectory_parameters : list of floats
        List of trajectory parameters to optimize.
    buffer_time : float (default: 0.0)
        Delay between start of the hodographic trajectory and the start of the propagation.

    Returns
    -------
    float
        Final time of the hodographic trajectory [s].
    """
    # Get initial time
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    return initial_time + get_trajectory_time_of_flight(trajectory_parameters) - buffer_time


def get_hodographic_trajectory(shaping_object: tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping,
                               trajectory_parameters: list,
                               specific_impulse: float,
                               output_path: str = None):
    """
    It computes the analytical hodographic trajectory and saves the results to a file, if desired.

    This function analytically calculates the hodographic trajectory from the Hodographic Shaping object. It
    retrieves both the trajectory and the acceleration profile; if desired, both are saved to files as follows:

    * hodographic_trajectory.dat: Cartesian states of semi-analytical trajectory;
    * hodographic_thrust_acceleration.dat: Thrust acceleration in inertial, Cartesian, coordinates, along the
    semi-analytical trajectory.

    NOTE: The independent variable (first column) does not represent the usual time (seconds since J2000), but instead
    denotes the time since departure.

    Parameters
    ----------
    shaping_object: tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    trajectory_parameters : list of floats
        List of trajectory parameters to be optimized.
    specific_impulse : float
        Constant specific impulse of the spacecraft.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    none
    """
    # Set time parameters
    start_time = 0.0
    final_time = get_trajectory_time_of_flight(trajectory_parameters)
    # Set number of data points
    number_of_data_points = 10000
    # Compute step size
    step_size = (final_time - start_time) / (number_of_data_points - 1)
    # Create epochs vector
    epochs = np.linspace(start_time,
                         final_time,
                         number_of_data_points)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Retrieve thrust acceleration profile from shaping object
    # NOTE TO THE STUDENTS: do not uncomment
    # thrust_acceleration_profile = shaping_object.get_thrust_acceleration_profile(
    #     epochs,
    #     specific_impulse_function)
    # Retrieve trajectory from shaping object
    trajectory_shape = shaping_object.get_trajectory(epochs)
    # If desired, save results to files
    if output_path is not None:
        # NOTE TO THE STUDENTS: do not uncomment
        # save2txt(thrust_acceleration_profile,
        #          'hodographic_thrust_acceleration.dat',
        #          output_path)
        save2txt(trajectory_shape,
                 'hodographic_trajectory.dat',
                 output_path)


def get_radial_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the radial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the radial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions = shape_based_thrust.recommended_radial_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    radial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[3:5]
    return (radial_velocity_shaping_functions,
            free_coefficients)


def get_normal_velocity_shaping_functions(trajectory_parameters: list,
                                          frequency: float,
                                          scale_factor: float,
                                          time_of_flight: float,
                                          number_of_revolutions: int) -> tuple:
    """
    Retrieves the normal velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun (currently unused).

    Returns
    -------
    tuple
        A tuple composed by two lists: the normal velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions = shape_based_thrust.recommended_normal_hodograph_functions(time_of_flight)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    normal_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=1.0,
        frequency=0.5 * frequency,
        scale_factor=scale_factor))
    # Set free parameters
    free_coefficients = trajectory_parameters[5:7]
    return (normal_velocity_shaping_functions,
            free_coefficients)


def get_axial_velocity_shaping_functions(trajectory_parameters: list,
                                         frequency: float,
                                         scale_factor: float,
                                         time_of_flight: float,
                                         number_of_revolutions: int) -> tuple:
    """
    Retrieves the axial velocity shaping functions (lowest and highest order in Gondelach and Noomen, 2015) and returns
    them together with the free coefficients.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to optimize.
    frequency: float
        Frequency of the highest-order methods.
    scale_factor: float
        Scale factor of the highest-order methods.
    time_of_flight: float
        Time of flight of the trajectory.
    number_of_revolutions: int
        Number of revolutions around the Sun.

    Returns
    -------
    tuple
        A tuple composed by two lists: the axial velocity shaping functions and their free coefficients.
    """
    # Retrieve default methods (lowest-order in Gondelach and Noomen, 2015)
    axial_velocity_shaping_functions = shape_based_thrust.recommended_axial_hodograph_functions(
        time_of_flight,
        number_of_revolutions)
    # Add degrees of freedom (highest-order in Gondelach and Noomen, 2015)
    exponent = 4.0
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_cosine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    axial_velocity_shaping_functions.append(shape_based_thrust.hodograph_scaled_power_sine(
        exponent=exponent,
        frequency=(number_of_revolutions + 0.5) * frequency,
        scale_factor=scale_factor ** exponent))
    # Set free parameters
    free_coefficients = trajectory_parameters[7:9]
    return (axial_velocity_shaping_functions,
            free_coefficients)


def create_hodographic_shaping_object(trajectory_parameters: list,
                                      bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) \
        -> tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping:
    """
    It creates and returns the hodographic shaping object, based on the trajectory parameters.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    hodographic_shaping_object : tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping
        Hodographic shaping object.
    """
    # Time settings
    initial_time = get_trajectory_initial_time(trajectory_parameters)
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
    final_time = get_trajectory_final_time(trajectory_parameters)
    # Number of revolutions
    number_of_revolutions = int(trajectory_parameters[2])
    # Compute relevant frequency and scale factor for shaping functions
    frequency = 2.0 * np.pi / time_of_flight
    scale_factor = 1.0 / time_of_flight
    # Retrieve shaping functions and free parameters
    radial_velocity_shaping_functions, radial_free_coefficients = get_radial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    normal_velocity_shaping_functions, normal_free_coefficients = get_normal_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    axial_velocity_shaping_functions, axial_free_coefficients = get_axial_velocity_shaping_functions(
        trajectory_parameters,
        frequency,
        scale_factor,
        time_of_flight,
        number_of_revolutions)
    # Retrieve boundary conditions and central body gravitational parameter
    initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(initial_time)
    final_state = bodies.get_body('Mars').state_in_base_frame_from_ephemeris(final_time)
    gravitational_parameter = bodies.get_body('Sun').gravitational_parameter
    # Create and return shape-based method
    hodographic_shaping_object = shape_based_thrust.HodographicShaping(initial_state,
                                                                       final_state,
                                                                       time_of_flight,
                                                                       gravitational_parameter,
                                                                       number_of_revolutions,
                                                                       radial_velocity_shaping_functions,
                                                                       normal_velocity_shaping_functions,
                                                                       axial_velocity_shaping_functions,
                                                                       radial_free_coefficients,
                                                                       normal_free_coefficients,
                                                                       axial_free_coefficients)
    return hodographic_shaping_object


def get_hodograph_thrust_acceleration_settings(trajectory_parameters: list,
                                               bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                               specific_impulse: float) \
        -> tudatpy.kernel.trajectory_design.shape_based_thrust.HodographicShaping:
    """
    It extracts the acceleration settings resulting from the hodographic trajectory and returns the equivalent thrust
    acceleration settings object.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    specific_impulse : float
        Constant specific impulse of the spacecraft.

    Returns
    -------
    tudatpy.kernel.numerical_simulation.propagation_setup.acceleration.ThrustAccelerationSettings
        Thrust acceleration settings object.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Compute offset, which is the time since J2000 (when t=0 for tudat) at which the simulation starts
    # N.B.: this is different from time_buffer, which is the delay between the start of the hodographic
    # trajectory and the beginning of the simulation
    time_offset = get_trajectory_initial_time(trajectory_parameters)
    # Create specific impulse lambda function
    specific_impulse_function = lambda t: specific_impulse
    # Return acceleration settings
    return transfer_trajectory.get_low_thrust_acceleration_settings(shaping_object,
                                                                    bodies,
                                                                    'Vehicle',
                                                                    specific_impulse_function,
                                                                    time_offset)


def get_hodograph_state_at_epoch(trajectory_parameters: list,
                                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                 epoch: float) -> np.ndarray:
    """
    It retrieves the Cartesian state, expressed in the inertial frame, at a given epoch of the analytical trajectory.

    Parameters
    ----------
    trajectory_parameters : list
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    np.ndarray
        Cartesian state in the inertial frame of the spacecraft at the given epoch.
    """
    # Create shaping object
    shaping_object = create_hodographic_shaping_object(trajectory_parameters,
                                                       bodies)
    # Define current hodograph time
    hodograph_time = epoch - get_trajectory_initial_time(trajectory_parameters)
    return shaping_object.get_state(hodograph_time)


###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################


# NOTE TO STUDENTS: THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
def generate_benchmarks(benchmark_step_size: float,
                        simulation_start_epoch: float,
                        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                        benchmark_propagator_settings:
                        tudatpy.kernel.numerical_simulation.propagation_setup.propagator.MultiTypePropagatorSettings,
                        are_dependent_variables_present: bool,
                        output_path: str = None):
    """
    Function to generate to accurate benchmarks.

    This function runs two propagations with two different integrator settings that serve as benchmarks for
    the nominal runs. The state and dependent variable history for both benchmarks are returned and, if desired, 
    they are also written to files (to the directory ./SimulationOutput/benchmarks/) in the following way:
    * benchmark_1_states.dat, benchmark_2_states.dat
        The numerically propagated states from the two benchmarks.
    * benchmark_1_dependent_variables.dat, benchmark_2_dependent_variables.dat
        The dependent variables from the two benchmarks.

    Parameters
    ----------
    simulation_start_epoch : float
        The start time of the simulation in seconds.
    constant_specific_impulse : float
        Constant specific impulse of the vehicle.  
    minimum_mars_distance : float
        Minimum distance from Mars at which the propagation stops.
    time_buffer : float
        Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        System of bodies present in the simulation.
    benchmark_propagator_settings
        Propagator settings object which is used to run the benchmark propagations.
    trajectory_parameters
        List that represents the trajectory parameters for the spacecraft.
    are_dependent_variables_present : bool
        If there are dependent variables to save.
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).

    Returns
    -------
    return_list : list
        List of state and dependent variable history in this order: state_1, state_2, dependent_1_ dependent_2.
    """
    ### CREATION OF THE TWO BENCHMARKS ###
    # Define benchmarks' step sizes
    first_benchmark_step_size = benchmark_step_size
    second_benchmark_step_size = 2.0 * first_benchmark_step_size

    # Create integrator settings for the first benchmark, using a fixed step size RKDP8(7) integrator
    # (the minimum and maximum step sizes are set equal, while both tolerances are set to inf)
    benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        first_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        first_benchmark_step_size,
        first_benchmark_step_size,
        np.inf,
        np.inf)

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies,
        benchmark_integrator_settings,
        benchmark_propagator_settings, print_dependent_variable_data=True)

    # Create integrator settings for the second benchmark in the same way
    benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        simulation_start_epoch,
        second_benchmark_step_size,
        propagation_setup.integrator.RKCoefficientSets.rkdp_87,
        second_benchmark_step_size,
        second_benchmark_step_size,
        np.inf,
        np.inf)

    print('Running second benchmark...')
    second_dynamics_simulator = numerical_simulation.SingleArcSimulator(
        bodies,
        benchmark_integrator_settings,
        benchmark_propagator_settings, print_dependent_variable_data=False)

    ### WRITE BENCHMARK RESULTS TO FILE ###
    # Retrieve state history
    first_benchmark_states = first_dynamics_simulator.state_history
    second_benchmark_states = second_dynamics_simulator.state_history
    # Write results to files
    if output_path is not None:
        save2txt(first_benchmark_states, 'benchmark_1_states.dat', output_path)
        save2txt(second_benchmark_states, 'benchmark_2_states.dat', output_path)
    # Add items to be returned
    return_list = [first_benchmark_states,
                   second_benchmark_states]

    ### DO THE SAME FOR DEPENDENT VARIABLES ###
    if are_dependent_variables_present:
        # Retrieve dependent variable history
        first_benchmark_dependent_variable = first_dynamics_simulator.dependent_variable_history
        second_benchmark_dependent_variable = second_dynamics_simulator.dependent_variable_history
        # Write results to file
        if output_path is not None:
            save2txt(first_benchmark_dependent_variable, 'benchmark_1_dependent_variables.dat', output_path)
            save2txt(second_benchmark_dependent_variable, 'benchmark_2_dependent_variables.dat', output_path)
        # Add items to be returned
        return_list.append(first_benchmark_dependent_variable)
        return_list.append(second_benchmark_dependent_variable)

    return return_list


def compare_benchmarks(first_benchmark: dict,
                       second_benchmark: dict,
                       output_path: str,
                       filename: str) -> dict:
    """
    It compares the results of two benchmark runs.

    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.

    Parameters
    ----------
    first_benchmark : dict
        State (or dependent variable history) from the first benchmark.
    second_benchmark : dict
        State (or dependent variable history) from the second benchmark.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.

    Returns
    -------
    benchmark_difference : dict
        Interpolated difference between the two benchmarks' state (or dependent variable) history.
    """
    # Create 8th-order Lagrange interpolator for first benchmark
    benchmark_interpolator = interpolators.create_one_dimensional_vector_interpolator(first_benchmark,
                                                                                      interpolators.lagrange_interpolation(
                                                                                          8))
    # Calculate the difference between the benchmarks
    print('Calculating benchmark differences...')
    # Initialize difference dictionaries
    benchmark_difference = dict()
    # Calculate the difference between the states and dependent variables in an iterative manner
    for second_epoch in second_benchmark.keys():
        benchmark_difference[second_epoch] = benchmark_interpolator.interpolate(second_epoch) - \
                                             second_benchmark[second_epoch]
    # Write results to files
    if output_path is not None:
        save2txt(benchmark_difference, filename, output_path)
    # Return the interpolator
    return benchmark_difference