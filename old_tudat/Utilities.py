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
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.util import result2array


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

    # Create Lambert arc state model
    lambert_arc_ephemeris = get_lambert_problem_result(bodies, 'Mars', initial_time, final_time)
    # Create propagation settings and propagate dynamics
    dynamics_simulator = propagate_trajectory(initial_time, final_time, bodies, lambert_arc_ephemeris,
                                              use_perturbations=False)
    # Extract state history from dynamics simulator
    state_history = dynamics_simulator.state_history
    state_history_matrix = result2array(state_history)

    # Retrieve boundary conditions and central body gravitational parameter
    initial_state = state_history_matrix[0,1:]
    # initial_state = bodies.get_body('Earth').state_in_base_frame_from_ephemeris(initial_time)
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

###########################################################################
# LAMBERT UTILITIES #####################################################
###########################################################################

def get_lambert_problem_result(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float ) -> tudatpy.kernel.numerical_simulation.environment.Ephemeris:

    """"
    This function solved Lambert's problem for a transfer from Earth (at departure epoch) to
    a target body (at arrival epoch), with the states of Earth and the target body defined
    by ephemerides stored inside the SystemOfBodies object (bodies). Note that this solver
    assumes that the transfer departs/arrives to/from the center of mass of Earth and the target body

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    target_body : The name (string) of the body to which the Lambert arc is to be computed

    departure_epoch : Epoch at which the departure from Earth's center of mass is to take place

    arrival_epoch : Epoch at which the arrival at he target body's center of mass is to take place

    Return
    ------
    Ephemeris object defining a purely Keplerian trajectory. This Keplerian trajectory defines the transfer
    from Earth to the target body according to the inputs to this function. Note that this Ephemeris object
    is valid before the departure epoch, and after the arrival epoch, and simply continues (forwards and backwards)
    the unperturbed Sun-centered orbit, as fully defined by the unperturbed transfer arc
    """

    # Gravitational parameter of the Sun
    central_body_gravitational_parameter = bodies.get_body("Sun").gravitational_parameter

    # Set initial and final positions for Lambert targeter
    initial_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name="Earth",
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=departure_epoch)

    final_state = spice_interface.get_body_cartesian_state_at_epoch(
        target_body_name=target_body,
        observer_body_name="Sun",
        reference_frame_name='ECLIPJ2000',
        aberration_corrections="NONE",
        ephemeris_time=arrival_epoch)

    # Create Lambert targeter
    lambertTargeter = two_body_dynamics.LambertTargeterIzzo(
        initial_state[:3], final_state[:3], arrival_epoch - departure_epoch, central_body_gravitational_parameter)

    # Compute initial Cartesian state of Lambert arc
    lambert_arc_initial_state = initial_state
    lambert_arc_initial_state[3:] = lambertTargeter.get_departure_velocity()

    # Compute Keplerian state of Lambert arc
    lambert_arc_keplerian_elements = element_conversion.cartesian_to_keplerian(lambert_arc_initial_state,
                                                                       central_body_gravitational_parameter)

    # Setup Keplerian ephemeris model that describes the Lambert arc
    kepler_ephemeris = environment_setup.create_body_ephemeris(
        environment_setup.ephemeris.keplerian(lambert_arc_keplerian_elements, departure_epoch,
                                              central_body_gravitational_parameter), "Vehicle")

    return kepler_ephemeris

def propagate_trajectory(
        initial_time: float,
        final_time: float,
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        lambert_arc_ephemeris: tudatpy.kernel.numerical_simulation.environment.Ephemeris,
        use_perturbations: bool,
        initial_state_correction=np.array([0, 0, 0, 0, 0, 0]),
        use_rsw_acceleration = False,
        rsw_acceleration_magnitude = np.array([0,0,0])) -> numerical_simulation.SingleArcSimulator:

    """
    This function will be repeatedly called throughout the assignment. Propagates the trajectory based
    on several input parameters

    Parameters
    ----------
    initial_time : Epoch since J2000 at which the propagation starts

    final_time : Epoch since J2000 at which the propagation will be terminated

    bodies : Body objects defining the physical simulation environment

    lambert_arc_ephemeris : Lambert arc state model as returned by the get_lambert_problem_result() function

    use_perturbations : Boolean to indicate whether a perturbed (True) or unperturbed (False) trajectory
                        is propagated

    initial_state_correction : Cartesian state which is added to the Lambert arc state when computing the numerical initial state

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.
    Return
    ------
    Dynamics simulator object from which the state- and dependent variable history can be extracted

    """
    # Compute initial state along Lambert arc (and apply correction if needed)
    lambert_arc_initial_state = lambert_arc_ephemeris.cartesian_state(initial_time) + initial_state_correction

    # Get propagator settings for perturbed/unperturbed forwards/backwards arcs
    if use_perturbations:
        propagator_settings = get_perturbed_propagator_settings(
            bodies, lambert_arc_initial_state, final_time,use_rsw_acceleration,rsw_acceleration_magnitude)
    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, final_time)

    # If propagation is backwards in time, make initial time step negative
    if initial_time > final_time:
        signed_fixed_step_size = -3600.0*24
    else:
        signed_fixed_step_size = 3600.0*24

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_4(initial_time, signed_fixed_step_size)

    # Propagate forward/backward perturbed/unperturbed arc and save results to files
    dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies, integrator_settings, propagator_settings)

    return dynamics_simulator

def get_unperturbed_propagator_settings(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        initial_state: np.array,
        termination_time: float ) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for an unperturbed trajectory.

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    initial_state : Cartesian initial state of the vehicle in the simulation

    termination_time : Epoch since J2000 at which the propagation will be terminated


    Return
    ------
    Propagation settings of the unperturbed trajectory.
    """

    bodies.get_body( 'Vehicle' ).mass = 24.0

    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']

    # Define accelerations acting on vehicle.
    acceleration_settings_on_spacecraft = dict(
    Sun=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ]
    )

    acceleration_settings = {'Vehicle': acceleration_settings_on_spacecraft}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    termination_settings = propagation_setup.propagator.time_termination( termination_time )

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
        propagation_setup.dependent_variable.relative_position("Mars", "Sun"),
        propagation_setup.dependent_variable.body_mass("Vehicle"),
        propagation_setup.dependent_variable.relative_speed("Vehicle", "Earth")
    ]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables = dependent_variables_to_save
    )
    return propagator_settings

def get_perturbed_propagator_settings(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        initial_state: np.array,
        termination_time: float,
        use_rsw_acceleration = False,
        rsw_acceleration_magnitude = np.array([0,0,0])) -> propagation_setup.propagator.SingleArcPropagatorSettings:
    """
    Creates the propagator settings for a perturbed trajectory.

    Parameters
    ----------
    bodies : Body objects defining the physical simulation environment

    initial_state : Cartesian initial state of the vehicle in the simulation

    termination_time : Epoch since J2000 at which the propagation will be terminated

    use_rsw_acceleration: Boolean defining whether an RSW acceleration (used to denote thrust) is to be used

    rsw_acceleration_magnitude: Magnitude of RSW acceleration, to be used if use_rsw_acceleration == True
                                the entries of this vector denote the acceleration in radial, normal and cross-track,
                                respectively.

    Return
    ------
    Propagation settings of the perturbed trajectory.
    """

    bodies.get_body( 'Vehicle' ).mass = 24.0

    reference_area_radiation = 0.2 * 0.2
    radiation_pressure_coefficient = 1.2
    radiation_pressure_settings = environment_setup.radiation_pressure.cannonball(
        "Sun", reference_area_radiation, radiation_pressure_coefficient
    )
    environment_setup.add_radiation_pressure_interface(
            bodies, "Vehicle", radiation_pressure_settings
    )

    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']

    # Define accelerations acting on vehicle.
    acceleration_settings_on_spacecraft = dict(
    Venus=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Earth=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Moon=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Mars=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Jupiter=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Saturn=
    [
        propagation_setup.acceleration.point_mass_gravity()
    ],
    Sun=
    [
        propagation_setup.acceleration.point_mass_gravity(),
        propagation_setup.acceleration.cannonball_radiation_pressure()
    ]
    )

    # DO NOT MODIFY, and keep AFTER creation of acceleration_settings_on_spacecraft
    # (line is added for compatibility with question 4)
    if use_rsw_acceleration:
        acceleration_settings_on_spacecraft["Sun"].append(
            propagation_setup.acceleration.empirical(rsw_acceleration_magnitude))

    acceleration_settings = {'Vehicle': acceleration_settings_on_spacecraft}

    acceleration_models = propagation_setup.create_acceleration_models(
        bodies, acceleration_settings, bodies_to_propagate, central_bodies)

    termination_settings = propagation_setup.propagator.time_termination( termination_time )

    dependent_variables_to_save = [
            propagation_setup.dependent_variable.total_acceleration( "Vehicle" )
    ]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables = dependent_variables_to_save
    )
    return propagator_settings

def write_propagation_results_to_file(
        dynamics_simulator: numerical_simulation.SingleArcSimulator,
        lambert_arc_ephemeris: tudatpy.kernel.numerical_simulation.environment.Ephemeris,
        file_output_identifier: str,
        output_directory: str):
    """
    This function will write the results of a numerical propagation, as well as the Lambert arc states at the epochs of the
    numerical state history, to a set of files. Two files are always written when calling this function (numerical state history, a
    and Lambert arc state history). If any dependent variables are saved during the propagation, those are also saved to a file

    Parameters
    ----------
    dynamics_simulator : Object that was used to propagate the dynamics, and which contains the numerical state and dependent
                         variable results

    lambert_arc_ephemeris : Lambert arc state model as returned by the get_lambert_problem_result() function

    file_output_identifier : Name that will be used to correctly save the output data files

    output_directory : Directory to which the files will be written

    Files written
    -------------

    <output_directory/file_output_identifier>_numerical_states.dat
    <output_directory/file_output_identifier>_dependent_variables.dat
    <output_directory/file_output_identifier>_lambert_statess.dat


    Return
    ------
    None

    """

    # Save numerical states
    simulation_result = dynamics_simulator.state_history
    save2txt(solution=simulation_result,
             filename=output_directory + file_output_identifier + "_numerical_states.dat",
             directory="./")

    # Save dependent variables
    dependent_variables = dynamics_simulator.dependent_variable_history
    if len(dependent_variables.keys()) > 0:
        save2txt(solution=dependent_variables,
                 filename=output_directory + file_output_identifier + "_dependent_variables.dat",
                 directory="./")

    # Save Lambert arc states
    lambert_arc_states = get_lambert_arc_history(lambert_arc_ephemeris, simulation_result)

    save2txt(solution=lambert_arc_states,
             filename=output_directory + file_output_identifier + "_lambert_states.dat",
             directory="./")

    return

def get_lambert_arc_history(
        lambert_arc_ephemeris: tudatpy.kernel.numerical_simulation.environment.Ephemeris,
        simulation_result: dict ) -> dict:
    """"
    This function extracts the state history (as a dict with time as keys, and Cartesian states as values)
    from an Ephemeris object defined by a lambert solver. This function takes a dictionary of states (simulation_result)
    as input, iterates over the keys of this dict (which represent times) to ensure that the times
    at which this function returns the states of the lambert arcs are identical to those at which the
    simulation_result has (numerically calculated) states


    Parameters
    ----------
    lambert_arc_ephemeris : Ephemeris object from which the states are to be extracted

    simulation_result : Dictionary of (numerically propagated) states, from which the keys
                        are used to determine the times at which this funcion is to extract states
                        from the lambert arc
    Return
    ------
    Variational equations solver object, from which the state-, state transition matrix-, and
    sensitivity matrix history can be extracted.
    """

    lambert_arc_states = dict()
    for state in simulation_result:
        lambert_arc_states[state] = lambert_arc_ephemeris.cartesian_state(state)

    return lambert_arc_states

