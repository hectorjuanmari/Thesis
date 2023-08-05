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

import numpy as np

# Problem-specific imports
import Utilities as Util

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
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class LowThrustProblem_SO:
    """
    Class to initialize, simulate and optimize the Low Thrust trajectory.
    The class is created specifically for this problem. This is done to provide better integration with Pagmo/Pygmo,
    where the optimization process (assignment 3) will be done. For the first two assignments, the presence of this
    class is not strictly needed, but it was chosen to create the code so that the same version could be used for all
    three assignments.
    Attributes
    ----------
    bodies
    integrator_settings
    propagator_settings
    specific_impulse
    minimum_mars_distance
    time_buffer
    perform_propagation
    Methods
    -------
    get_last_run_propagated_state_history()
    get_last_run_dependent_variable_history()
    get_last_run_dynamics_simulator()
    fitness(trajectory_parameters)
    get_hodographic_shaping()
    """

    def __init__(self,
                 bodies: tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
                 specific_impulse: float,
                 minimum_mars_distance: float,
                 time_buffer: float,
                 vehicle_mass: float,
                 vinf: list,
                 decision_variable_range,
                 constraint_bounds,
                 perform_propagation: bool = True):
        """
        Constructor for the LowThrustProblem class.
        Parameters
        ----------
        bodies : tudatpy.kernel.numerical_simulation.environment.SystemOfBodies,
            System of bodies present in the simulation.
        integrator_settings : tudatpy.kernel.numerical_simulation.propagation_setup.integrator.IntegratorSettings
            Integrator settings to be provided to the dynamics simulator.
        specific_impulse : float
            Constant specific impulse of the vehicle.
        minimum_mars_distance : float
            Minimum distance from Mars at which the propagation stops.
        time_buffer : float
            Time interval between the simulation start epoch and the beginning of the hodographic trajectory.
        perform_propagation : bool (default: True)
            If true, the propagation is performed.
        Returns
        -------
        none
        """
        # Copy arguments as attributes
        self.hodographic_shaping_function = None
        self.bodies_function = lambda: bodies
        self.specific_impulse = specific_impulse
        self.minimum_mars_distance = minimum_mars_distance
        self.time_buffer = time_buffer
        self.vehicle_mass = vehicle_mass
        self.vinf = vinf
        self.decision_variable_range = decision_variable_range
        self.constraint_bounds = constraint_bounds
        self.perform_propagation = perform_propagation

    def get_nobj(self):
        return 1

    def get_nec(self):
        return 0

    def get_nic(self):
        return 0

    def get_bounds(self):
        return self.decision_variable_range

    def get_last_run_propagated_cartesian_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, converted to Cartesian states
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_equations_of_motion_numerical_solution()

    def get_last_run_propagated_state_history(self) -> dict:
        """
        Returns the full history of the propagated state, not converted to Cartesian state
        (i.e. in the actual formulation that was used during the numerical integration).
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_equations_of_motion_numerical_solution_raw()

    def get_last_run_dependent_variable_history(self) -> dict:
        """
        Returns the full history of the dependent variables.
        Parameters
        ----------
        none
        Returns
        -------
        dict
        """
        return self.dynamics_simulator_function().get_dependent_variable_history()

    def get_last_run_dynamics_simulator(self) -> tudatpy.kernel.numerical_simulation.SingleArcSimulator:
        """
        Returns the dynamics simulator object.
        Parameters
        ----------
        none
        Returns
        -------
        tudatpy.kernel.numerical_simulation.SingleArcSimulator
        """
        return self.dynamics_simulator_function()

    def fitness(self,
                trajectory_parameters) -> float:
        """
        Propagate the trajectory using the hodographic method with the parameters given as argument.
        This function uses the trajectory parameters to create a new hodographic shaping object, from which a new
        thrust acceleration profile is extracted. Subsequently, the trajectory is propagated numerically, if desired.
        The fitness, currently set to zero, can be computed here: it will be used during the optimization process.
        Parameters
        ----------
        trajectory_parameters : list of floats
            List of trajectory parameters to optimize.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        # Create hodographic shaping object
        bodies = self.bodies_function()
        vinf = self.vinf
        hodographic_shaping = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                 bodies, vinf)
        self.hodographic_shaping_function = lambda: hodographic_shaping

        # Add the objective and constraint values into the fitness vector

        # LOW THRUST ΔV
        dV_LT = hodographic_shaping.delta_v_per_leg[0]

        # MAX THRUST (constraint)
        thrust = np.linalg.norm(list(hodographic_shaping.rsw_thrust_accelerations_along_trajectory(100).values()),axis=1)*self.vehicle_mass
        T_max = np.max(thrust)

        constraints = [T_max]

        penalty_function = Util.get_penalty_function(constraints, self.constraint_bounds)

        weight = 1000

        return np.array([dV_LT + weight * penalty_function])

    def fitness_MO(self,
                   trajectory_parameters) -> float:
        """
        Propagate the trajectory using the hodographic method with the parameters given as argument.
        This function uses the trajectory parameters to create a new hodographic shaping object, from which a new
        thrust acceleration profile is extracted. Subsequently, the trajectory is propagated numerically, if desired.
        The fitness, currently set to zero, can be computed here: it will be used during the optimization process.
        Parameters
        ----------
        trajectory_parameters : list of floats
            List of trajectory parameters to optimize.
        Returns
        -------
        fitness : float
            Fitness value (for optimization, see assignment 3).
        """
        # Create hodographic shaping object
        bodies = self.bodies_function()
        vinf = self.vinf
        hodographic_shaping = Util.create_hodographic_trajectory(trajectory_parameters,
                                                                 bodies, vinf)
        self.hodographic_shaping_function = lambda: hodographic_shaping

        # Add the objective and constraint values into the fitness vector

        # LOW THRUST ΔV
        dV_LT = hodographic_shaping.delta_v_per_leg[0]

        # MAX THRUST (constraint)
        thrust = np.linalg.norm(list(hodographic_shaping.rsw_thrust_accelerations_along_trajectory(100).values()),
                                axis=1) * self.vehicle_mass
        T_max = np.max(thrust)

        constraints = [T_max]

        penalty_function = Util.get_penalty_function(constraints, self.constraint_bounds)

        return [dV_LT, penalty_function]
