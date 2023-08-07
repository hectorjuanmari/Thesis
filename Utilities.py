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
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import two_body_dynamics
from tudatpy.kernel.astro import element_conversion
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.util import result2array


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
        terminate_exactly_on_final_condition=True)
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
    # dependent_variables_to_save = [propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
    #                                propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
    #                                propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
    #                                propagation_setup.dependent_variable.single_acceleration_norm(
    #                                    propagation_setup.acceleration.thrust_acceleration_type,'Vehicle','Vehicle'),
    #                                propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth')]
    dependent_variables_to_save = [propagation_setup.dependent_variable.relative_distance('Vehicle', 'Earth'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Sun'),
                                   propagation_setup.dependent_variable.relative_distance('Vehicle', 'Mars'),
                                   propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
                                   propagation_setup.dependent_variable.relative_position("Mars", "SSB"),
                                   propagation_setup.dependent_variable.body_mass("Vehicle"),
                                   propagation_setup.dependent_variable.single_acceleration_norm(
                                       propagation_setup.acceleration.thrust_acceleration_type, 'Vehicle', 'Vehicle'),
                                   propagation_setup.dependent_variable.total_acceleration("Vehicle"),
                                   propagation_setup.dependent_variable.single_acceleration(
                                       propagation_setup.acceleration.point_mass_gravity_type, 'Vehicle', 'Sun'),
                                   propagation_setup.dependent_variable.single_acceleration(
                                       propagation_setup.acceleration.thrust_acceleration_type, 'Vehicle', 'Vehicle'),
                                   propagation_setup.dependent_variable.relative_position("Vehicle", "SSB"),
                                   propagation_setup.dependent_variable.relative_position("Vehicle", "Sun")
                                   ]

    return dependent_variables_to_save


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
    multi_stage_integrators = [propagation_setup.integrator.CoefficientSets.rkf_45,
                               propagation_setup.integrator.CoefficientSets.rkf_56,
                               propagation_setup.integrator.CoefficientSets.rkf_78,
                               propagation_setup.integrator.CoefficientSets.rkdp_87]
    tolerance_1 = [10 ** -15, 10 ** -14, 10 ** -13, 10 ** -12, 10 ** -11]
    tolerance_2 = [10 ** -14, 10 ** -13, 10 ** -12, 10 ** -11, 10 ** -10]
    # Use variable step-size integrator
    if integrator_index < 4:
        # Select variable-step integrator
        current_coefficient_set = multi_stage_integrators[integrator_index]
        # Compute current tolerance
        current_tolerance = tolerance_1[settings_index]
        if integrator_index == 3: current_tolerance = tolerance_2[settings_index]
        # Create integrator settings
        integrator = propagation_setup.integrator
        # Here (epsilon, inf) are set as respectively min and max step sizes
        # also note that the relative and absolute tolerances are the same value
        integrator_settings = integrator.runge_kutta_variable_step_size(
            1.0,
            current_coefficient_set,
            np.finfo(float).eps,
            np.inf,
            current_tolerance,
            current_tolerance)
    # Use fixed step-size integrator
    elif integrator_index >= 4 and integrator_index < 8:
        # Compute time step
        fixed_step_size = 7200 * 2.0 ** (settings_index)
        if integrator_index > 5: fixed_step_size = 7200 * 2.0 ** (settings_index + 3)
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
        current_tolerance = 10.0 ** (-10.0 + settings_index)
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
        fixed_step_size = 7200 * 2.0 ** (settings_index)
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
    # elif integrator_index >= 16 and integrator_index < 20:
    #     # Create integrator settings
    #     integrator = propagation_setup.integrator
    #     current_tolerance = 10.0 ** (-12.0 + settings_index)
    #     current_order = integrator_index - 12
    #     print(current_order)
    #     # Here (epsilon, inf) are set as respectively min and max step sizes
    #     # also note that the relative and absolute tolerances are the same value
    #     integrator_settings = integrator.bulirsch_stoer(simulation_start_epoch,
    #                                                     100,
    #                                                     propagation_setup.integrator.ExtrapolationMethodStepSequences.bulirsch_stoer_sequence,
    #                                                     current_order,
    #                                                     10 ** -15,
    #                                                     np.inf,
    #                                                     current_tolerance,
    #                                                     current_tolerance)
    return integrator_settings


def get_propagator_settings(trajectory_parameters,
                            bodies,
                            initial_propagation_time,
                            vehicle_initial_mass,
                            termination_settings,
                            dependent_variables_to_save,
                            current_propagator=propagation_setup.propagator.cowell,
                            model_choice=0,
                            vinf=np.zeros([3, 1])):
    """
    Creates the propagator settings.

    This function creates the propagator settings for translational motion and mass, for the given simulation settings
    Note that, in this function, the thrust_parameters are used to update the engine model and rotation model of the
    vehicle. The propagator settings that are returned as output of this function are not yet usable: they do not
    contain any integrator settings, which should be set at a later point by the user

    Parameters
    ----------
    trajectory_parameters : list[ float ]
        List of free parameters for the low-thrust model, which will be used to update the vehicle properties such that
        the new thrust/magnitude direction are used. The meaning of the parameters in this list is stated at the
        start of the *Propagation.py file
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.
    initial_propagation_time : float
        Start of the simulation [s] with t=0 at J2000.
    vehicle_initial_mass : float
        Mass of the vehicle to be used at the initial time
    termination_settings : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.PropagationTerminationSettings
        Propagation termination settings object to be used
    dependent_variables_to_save : list[tudatpy.kernel.numerical_simulation.propagation_setup.dependent_variable]
        List of dependent variables to save.
    current_propagator : tudatpy.kernel.numerical_simulation.propagation_setup.propagator.TranslationalPropagatorType
        Type of propagator to be used for translational dynamics

    Returns
    -------
    propagator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.MultiTypePropagatorSettings
        Propagator settings to be provided to the dynamics simulator.
    """

    # Define bodies that are propagated and their central bodies of propagation
    bodies_to_propagate = ['Vehicle']
    central_bodies = ['Sun']
    # Update vehicle rotation model and thrust magnitude model
    transfer_trajectory = set_hodograph_thrust_model(trajectory_parameters, bodies, vinf)
    # Define accelerations acting on capsule
    acceleration_settings_on_vehicle = {
        'Sun': [propagation_setup.acceleration.point_mass_gravity()],
        # 'Earth': [propagation_setup.acceleration.point_mass_gravity()],
        # 'Mars': [propagation_setup.acceleration.point_mass_gravity()],
        'Vehicle': [propagation_setup.acceleration.thrust_from_engine('LowThrustEngine')]
    }

    ####
    # Solar System Point Mass
    ####

    if model_choice == 1:
        acceleration_settings_on_vehicle['Mercury'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 2:
        acceleration_settings_on_vehicle['Venus'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 3:
        acceleration_settings_on_vehicle['Earth'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 4:
        acceleration_settings_on_vehicle['Mars'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 5:
        acceleration_settings_on_vehicle['Jupiter'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 6:
        acceleration_settings_on_vehicle['Saturn'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 7:
        acceleration_settings_on_vehicle['Uranus'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 8:
        acceleration_settings_on_vehicle['Neptune'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
    elif model_choice == 9:
        acceleration_settings_on_vehicle['Moon'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]

    ####
    # Solar Radiation Pressure
    ####

    elif model_choice == 10:
        acceleration_settings_on_vehicle['Sun'] = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.cannonball_radiation_pressure()
        ]

    ####
    # Spherical Harmonics
    ####

    elif model_choice == 11:
        acceleration_settings_on_vehicle['Earth'] = [
            propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
        ]
    elif model_choice == 12:
        acceleration_settings_on_vehicle['Mars'] = [
            propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
        ]
    elif model_choice == 13:
        acceleration_settings_on_vehicle['Moon'] = [
            propagation_setup.acceleration.spherical_harmonic_gravity(2, 0),
        ]

    ####
    # Relativistic Corrections
    ####

    elif model_choice == 14:
        use_schwarzschild = True
        use_lense_thirring = False
        use_de_sitter = False
        acceleration_settings_on_vehicle['Sun'] = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(
                use_schwarzschild,
                use_lense_thirring,
                use_de_sitter
            )
        ]
    elif model_choice == 15:
        use_schwarzschild = True
        use_lense_thirring = False
        use_de_sitter = False
        acceleration_settings_on_vehicle['Earth'] = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(
                use_schwarzschild,
                use_lense_thirring,
                use_de_sitter
            )
        ]
    elif model_choice == 16:
        use_schwarzschild = True
        use_lense_thirring = False
        use_de_sitter = False
        acceleration_settings_on_vehicle['Mars'] = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.relativistic_correction(
                use_schwarzschild,
                use_lense_thirring,
                use_de_sitter
            )
        ]

    elif model_choice == 17:
        use_schwarzschild = True
        use_lense_thirring = False
        use_de_sitter = False
        acceleration_settings_on_vehicle['Sun'] = [
            propagation_setup.acceleration.point_mass_gravity(),
            propagation_setup.acceleration.cannonball_radiation_pressure(),
            propagation_setup.acceleration.relativistic_correction(
                use_schwarzschild,
                use_lense_thirring,
                use_de_sitter
            )
        ]
        acceleration_settings_on_vehicle['Mercury'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Venus'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Earth'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Mars'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Jupiter'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Saturn'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Uranus'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Neptune'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]
        acceleration_settings_on_vehicle['Moon'] = [
            propagation_setup.acceleration.point_mass_gravity()
        ]

    # ####
    # # Two-body ephemeris
    # ####
    #
    # elif model_choice == 17:
    #     acceleration_settings_on_vehicle['Mercury'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 18:
    #     acceleration_settings_on_vehicle['Venus'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 19:
    #     acceleration_settings_on_vehicle['Earth'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 20:
    #     acceleration_settings_on_vehicle['Mars'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 21:
    #     acceleration_settings_on_vehicle['Jupiter'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 22:
    #     acceleration_settings_on_vehicle['Saturn'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 23:
    #     acceleration_settings_on_vehicle['Uranus'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 24:
    #     acceleration_settings_on_vehicle['Neptune'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]
    # elif model_choice == 25:
    #     acceleration_settings_on_vehicle['Moon'] = [
    #         propagation_setup.acceleration.point_mass_gravity()
    #     ]

    # Create global accelerations dictionary
    acceleration_settings = {'Vehicle': acceleration_settings_on_vehicle}
    acceleration_models = propagation_setup.create_acceleration_models(
        bodies,
        acceleration_settings,
        bodies_to_propagate,
        central_bodies)

    # Retrieve initial state
    initial_state = transfer_trajectory.legs[0].state_along_trajectory(initial_propagation_time)

    # Create propagation settings for the translational dynamics
    translational_propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        initial_propagation_time,
        None,
        termination_settings,
        current_propagator,
        output_variables=dependent_variables_to_save)

    # Create mass rate model
    mass_rate_settings_on_vehicle = {'Vehicle': [propagation_setup.mass_rate.from_thrust()]}
    mass_rate_models = propagation_setup.create_mass_rate_models(bodies,
                                                                 mass_rate_settings_on_vehicle,
                                                                 acceleration_models)
    # Create mass propagator settings
    mass_propagator_settings = propagation_setup.propagator.mass(bodies_to_propagate,
                                                                 mass_rate_models,
                                                                 np.array([vehicle_initial_mass]),
                                                                 initial_propagation_time,
                                                                 None,
                                                                 termination_settings)

    # Create multi-type propagation settings list
    propagator_settings_list = [translational_propagator_settings,
                                mass_propagator_settings]

    # Create multi-type propagation settings object for translational dynamics and mass.
    # NOTE: these are not yet 'valid', as no integrator settings are defined yet
    propagator_settings = propagation_setup.propagator.multitype(propagator_settings_list,
                                                                 None,
                                                                 initial_propagation_time,
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


def get_hodographic_trajectory(shaping_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
                               output_path: str):
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
    trajectory_parameters : list[ float ]
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
                                  bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                                  vinf=np.zeros([3, 1])) \
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

    # Create settings for transfer trajectory (zero excess velocity on departure and arrival)
    hodographic_leg_settings = transfer_trajectory.hodographic_shaping_leg(
        radial_velocity_shaping_functions,
        normal_velocity_shaping_functions,
        axial_velocity_shaping_functions)
    node_settings = list()
    node_settings.append(transfer_trajectory.departure_node(np.inf, 0.99))
    node_settings.append(transfer_trajectory.capture_node(29e6, 0.99))

    # Create and return transfer trajectory
    trajectory_object = transfer_trajectory.create_transfer_trajectory(
        bodies, [hodographic_leg_settings], node_settings, ['Earth', 'Mars'], 'Sun')

    # Extract node times
    node_times = list()
    node_times.append(get_trajectory_initial_time(trajectory_parameters))
    node_times.append(get_trajectory_final_time(trajectory_parameters))

    # transfer_trajectory.print_parameter_definitions( [hodographic_leg_settings], node_settings )
    hodograph_free_parameters = trajectory_parameters[2:9]

    node_parameters = list()
    node_parameters.append(vinf)
    node_parameters.append(np.zeros([3, 1]))

    # Update trajectory to given times, node settings, and hodograph parameters
    trajectory_object.evaluate(node_times, [hodograph_free_parameters], node_parameters)
    return trajectory_object


def set_hodograph_thrust_model(trajectory_parameters: list,
                               bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                               vinf=np.zeros([3, 1])):
    """
    It extracts the acceleration settings resulting from the hodographic trajectory and returns the equivalent thrust
    acceleration settings object. In addition, it returns teh transfer trajectory object for later use in the code

    Parameters
    ----------
    trajectory_parameters : list[float]
        List of trajectory parameters to be optimized.
    bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies
        System of bodies present in the simulation.

    Returns
    -------
    None
    """
    # Create shaping object
    trajectory_object = create_hodographic_trajectory(trajectory_parameters, bodies, vinf)
    transfer_trajectory.set_low_thrust_acceleration(trajectory_object.legs[0], bodies, 'Vehicle', 'LowThrustEngine')

    # Return trajectory object
    return trajectory_object


###########################################################################
# BENCHMARK UTILITIES #####################################################
###########################################################################

# THIS FUNCTION CAN BE EXTENDED TO GENERATE A MORE ROBUST BENCHMARK (USING MORE THAN 2 RUNS)
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
        first_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87,
        first_benchmark_step_size,
        first_benchmark_step_size,
        np.inf,
        np.inf)
    benchmark_propagator_settings.integrator_settings = benchmark_integrator_settings

    print('Running first benchmark...')
    first_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

    # Create integrator settings for the second benchmark in the same way
    benchmark_integrator_settings = propagation_setup.integrator.runge_kutta_variable_step_size(
        second_benchmark_step_size,
        propagation_setup.integrator.CoefficientSets.rkdp_87,
        second_benchmark_step_size,
        second_benchmark_step_size,
        np.inf,
        np.inf)
    benchmark_propagator_settings.integrator_settings = benchmark_integrator_settings

    print('Running second benchmark...')
    second_dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies,
        benchmark_propagator_settings)

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


###########################################################################
# LAMBERT UTILITIES #######################################################
###########################################################################

def get_lambert_problem_result(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        target_body: str,
        departure_epoch: float,
        arrival_epoch: float) -> tudatpy.kernel.numerical_simulation.environment.Ephemeris:
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
        use_rsw_acceleration=False,
        rsw_acceleration_magnitude=np.array([0, 0, 0])) -> numerical_simulation.SingleArcSimulator:
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
            bodies, lambert_arc_initial_state, final_time, use_rsw_acceleration, rsw_acceleration_magnitude)
    else:
        propagator_settings = get_unperturbed_propagator_settings(
            bodies, lambert_arc_initial_state, final_time)

    # If propagation is backwards in time, make initial time step negative
    if initial_time > final_time:
        signed_fixed_step_size = -3600.0 * 24
    else:
        signed_fixed_step_size = 3600.0 * 24

    # Create numerical integrator settings
    integrator_settings = propagation_setup.integrator.runge_kutta_4(initial_time, signed_fixed_step_size)

    # Propagate forward/backward perturbed/unperturbed arc and save results to files
    dynamics_simulator = numerical_simulation.SingleArcSimulator(bodies, integrator_settings, propagator_settings)

    return dynamics_simulator


def get_unperturbed_propagator_settings(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        initial_state: np.array,
        termination_time: float) -> propagation_setup.propagator.SingleArcPropagatorSettings:
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

    bodies.get_body('Vehicle').mass = 24.0

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

    termination_settings = propagation_setup.propagator.time_termination(termination_time)

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.relative_position("Earth", "Sun"),
        propagation_setup.dependent_variable.relative_position("Mars", "Sun"),
        propagation_setup.dependent_variable.body_mass("Vehicle"),
        propagation_setup.dependent_variable.relative_speed("Vehicle", "Earth"),
    ]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables=dependent_variables_to_save
    )
    return propagator_settings


def get_perturbed_propagator_settings(
        bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
        initial_state: np.array,
        termination_time: float,
        use_rsw_acceleration=False,
        rsw_acceleration_magnitude=np.array([0, 0, 0])) -> propagation_setup.propagator.SingleArcPropagatorSettings:
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

    bodies.get_body('Vehicle').mass = 24.0

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

    termination_settings = propagation_setup.propagator.time_termination(termination_time)

    dependent_variables_to_save = [
        propagation_setup.dependent_variable.total_acceleration("Vehicle")
    ]

    # Create propagation settings.
    propagator_settings = propagation_setup.propagator.translational(
        central_bodies,
        acceleration_models,
        bodies_to_propagate,
        initial_state,
        termination_settings,
        output_variables=dependent_variables_to_save
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
        simulation_result: dict) -> dict:
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


def compare_models(first_model: dict,
                   second_model: dict,
                   interpolation_epochs: np.ndarray,
                   output_path: str,
                   filename: str) -> dict:
    """
    It compares the results of two runs with different model settings.
    It uses an 8th-order Lagrange interpolator to compare the state (or the dependent variable, depending on what is
    given as input) history. The difference is returned in form of a dictionary and, if desired, written to a file named
    filename and placed in the directory output_path.
    Parameters
    ----------
    first_model : dict
        State (or dependent variable history) from the first run.
    second_model : dict
        State (or dependent variable history) from the second run.
    interpolation_epochs : np.ndarray
        Vector of epochs at which the two runs are compared.
    output_path : str
        If and where to save the benchmark results (if None, results are NOT written).
    filename : str
        Name of the output file.
    Returns
    -------
    model_difference : dict
        Interpolated difference between the two simulations' state (or dependent variable) history.
    """
    # # Create interpolator settings
    # interpolator_settings = interpolators.lagrange_interpolation(
    #     8, boundary_interpolation=interpolators.use_boundary_value)
    # # Create 8th-order Lagrange interpolator for both cases
    # first_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    #     first_model, interpolator_settings)
    # second_interpolator = interpolators.create_one_dimensional_vector_interpolator(
    #     second_model, interpolator_settings)
    # # Calculate the difference between the first and second model at specific epochs
    # model_difference = {epoch: second_interpolator.interpolate(epoch) - first_interpolator.interpolate(epoch)
    #                     for epoch in interpolation_epochs}
    model_difference = {epoch: second_model[epoch] - first_model[epoch]
                        for epoch in first_model.keys()}
    # Write results to files
    if output_path is not None:
        save2txt(model_difference,
                 filename,
                 output_path)
    # Return the model difference
    return model_difference

###########################################################################
# OPTIMISATION UTILITIES ##################################################
###########################################################################

def get_penalty_function(constraints, bounds):
    p = 0
    blow_up = 1e1
    for k in range(len(constraints)):
        g_k = constraints[k]-bounds[0, k]
        g_lim = bounds[1, k] - bounds[0, k]

        if k == 2:
            g_k = -g_k
            g_lim = -g_lim

        G_k = max(0.0, g_k)/g_lim  # This is positive by definition!
        if G_k > 1: G_k = blow_up

        p = p + G_k

    return p

def get_fx_dict(population, fitness, generation):
    population_size = population.shape[0]
    # number_of_decision_variables = population.shape[1]
    # number_of_objectives = fitness.shape[1]

    population_dict = dict()
    fitness_dict = dict()
    for individual in range(population_size):
        track_id = generation + individual/10000
        population_dict[track_id] = population[individual, :]
        fitness_dict[track_id] = fitness[individual, :]

    return population_dict, fitness_dict

def get_convergence(fitness, convergence_rate, current_state):

    keys = np.array(list(fitness.keys()))
    population_size = sum(np.floor(keys) == 0)
    new_gen = np.floor(keys[-1])
    old_gen = new_gen - 1

    if old_gen == 0:
        old_compliance = 0

    compliance_dict = dict()

    old_fitness = np.zeros((population_size, 3))
    new_fitness = np.zeros((population_size, 3))
    compliance = 0
    for individual in range(population_size):
        old_track = old_gen + individual / 10000
        new_track = new_gen + individual / 10000

        old_fitness[individual, :] = np.array(list(fitness[old_track]))
        new_fitness[individual, :] = np.array(list(fitness[new_track]))

        if old_gen == 0 and old_fitness[individual, 2] == 0:
            old_compliance = old_compliance + 1
            compliance_dict[old_track] = old_fitness[individual, :]

        if new_fitness[individual, 2] == 0:
            compliance = compliance + 1
            compliance_dict[new_track] = new_fitness[individual, :]

    d_old = get_COM(old_fitness)
    d_new = get_COM(new_fitness)

    increment_d = np.linalg.norm(d_new - d_old)/np.linalg.norm(d_old) * 100
    convergence = increment_d < convergence_rate

    if convergence: current_state = current_state + 1
    else: current_state = 0


    print('Generation: %i   increment: %f   Convergence: %i/5' % (new_gen, increment_d, current_state))

    if old_gen == 0:
        print('Individuals meeting ALL requirements in generation 0: %i' % (old_compliance))

    print('Individuals meeting ALL requirements in generation %i: %i' % (new_gen, compliance))

    return current_state, compliance_dict

def get_COM(fitness_array):
    if fitness_array.shape[1] == 1:
        population_size = fitness_array.shape[0]
        cum = 0
        for individual in range(population_size):
            cum = cum + fitness_array[individual]

        d = cum / population_size
        d = np.array([abs(d)])

    else:
        m_c = 500
        tof_c = 300
        delta_m = fitness_array[:, 0]/m_c
        tof = fitness_array[:, 1]/tof_c
        p = fitness_array[:, 2]

        # for just_in_case in range(p.shape[0]):
        #     if p[just_in_case] > 1:
        #         p[just_in_case] = 100*np.log10(p[just_in_case])

        population_size = delta_m.shape[0]
        cum = np.zeros(3)
        for individual in range(population_size):
            r_k = np.array([delta_m[individual], tof[individual], p[individual]])
            # d_k = np.sqrt(delta_m[individual]**2 + tof[individual]**2)
            cum = cum + r_k

        d = cum / population_size

    return d
