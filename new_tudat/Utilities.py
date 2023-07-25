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


def get_hodographic_trajectory(shaping_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
                               output_path: str ):
    """
    It computes the analytical hodographic trajectory and saves the results to a file
    This function analytically calculates the hodographic trajectory from the Hodographic Shaping object.
    * hodographic_trajectory.dat: Cartesian states of semi-analytical trajectory;
    Parameters
    ----------
    shaping_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
        TransferTrajectory object with a single leg: the hodographic shaping leg from Earth to Mars
    output_path : str (default: None)
        If and where to save the benchmark results (if None, results are NOT written).
    Returns
    -------
    none
    """

    # Set number of data points
    number_of_data_points = 10000

    # Extract trajectory shape
    trajectory_shape = shaping_object.states_along_trajectory(number_of_data_points)
    # If desired, save results to files
    save2txt(trajectory_shape,
                 'hodographic_trajectory.dat',
                 output_path)
    return trajectory_shape


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



def create_hodographic_trajectory(trajectory_parameters: list,
                                  bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) \
        -> tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory:
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
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
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

    # Create settings for transfer trajectory (zero excess velocity on departure and arrival)
    hodographic_leg_settings = transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_shaping_functions,
        normal_velocity_shaping_functions,
        axial_velocity_shaping_functions )
    node_settings = list()
    node_settings.append( transfer_trajectory.departure_node( 1.0E8, 0.0 ) )
    node_settings.append( transfer_trajectory.capture_node( 1.0E8, 0.0 ) )

    # Create and return transfer trajectory
    trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies, [hodographic_leg_settings], node_settings, ['Earth','Mars'],'Sun' )

    # Extract node times
    node_times = list( )
    node_times.append( get_trajectory_initial_time( trajectory_parameters ) )
    node_times.append( get_trajectory_final_time( trajectory_parameters ) )

    #transfer_trajectory.print_parameter_definitions( [hodographic_leg_settings], node_settings )
    hodograph_free_parameters = trajectory_parameters[2:9]

    # Depart and arrive with 0 excess velocity
    node_parameters = list()
    node_parameters.append( np.zeros([3,1]))
    node_parameters.append( np.zeros([3,1]))

    # Update trajectory to given times, node settings, and hodograph parameters
    trajectory_object.evaluate( node_times, [hodograph_free_parameters], node_parameters )
    return trajectory_object

def create_hodographic_trajectory(trajectory_parameters: list,
                                  bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies) \
        -> tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory:
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
    time_of_flight = get_trajectory_time_of_flight(trajectory_parameters)
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

    # Create settings for transfer trajectory (zero excess velocity on departure and arrival)
    hodographic_leg_settings = transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_shaping_functions,
        normal_velocity_shaping_functions,
        axial_velocity_shaping_functions )
    node_settings = list()
    node_settings.append( transfer_trajectory.departure_node( 1.0E8, 0.0 ) )
    node_settings.append( transfer_trajectory.capture_node( 1.0E8, 0.0 ) )

    # Create and return transfer trajectory
    trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies, [hodographic_leg_settings], node_settings, ['Earth','Mars'],'Sun' )

    # Extract node times
    node_times = list( )
    node_times.append( get_trajectory_initial_time( trajectory_parameters ) )
    node_times.append( get_trajectory_final_time( trajectory_parameters ) )

    #transfer_trajectory.print_parameter_definitions( [hodographic_leg_settings], node_settings )
    hodograph_free_parameters = trajectory_parameters[2:9]

    # Depart and arrive with 0 excess velocity
    node_parameters = list()
    node_parameters.append( np.zeros([3,1]))
    node_parameters.append( np.zeros([3,1]))

    # Update trajectory to given times, node settings, and hodograph parameters
    trajectory_object.evaluate( node_times, [hodograph_free_parameters], node_parameters )
    return trajectory_object

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

