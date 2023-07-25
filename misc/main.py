'''
Copyright (c) 2010-2020, Delft University of Technology
All rigths reserved

This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.
'''

from interplanetary_transfer_helper_functions import *
from matplotlib import pyplot as plt
from matplotlib import rc
import os


# Load spice kernels.
spice_interface.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# RUN CODE FOR QUESTION 1 #################################################
###########################################################################

# Create body objects
bodies = create_simulation_bodies()

# Create Lambert arc state model
lambert_arc_ephemeris = get_lambert_problem_result(bodies, target_body, departure_epoch, arrival_epoch)

# Create propagation settings and propagate dynamics
dynamics_simulator = propagate_trajectory( departure_epoch, arrival_epoch, bodies, lambert_arc_ephemeris,
                     use_perturbations = False)

write_propagation_results_to_file(
    dynamics_simulator, lambert_arc_ephemeris, "Q1", output_directory)

# Extract state history from dynamics simulator
state_history = dynamics_simulator.state_history
dependent_variables = dynamics_simulator.dependent_variable_history

# Evaluate the Lambert arc model at each of the epochs in the state_history
lambert_history = get_lambert_arc_history( lambert_arc_ephemeris, state_history )

state_history_matrix = result2array(state_history)
lambert_history_matrix = result2array(lambert_history)
dependent_var = np.vstack(list(dependent_variables.values()))
time = dependent_variables.keys()
time_days = [ t / constants.JULIAN_DAY - departure_epoch / constants.JULIAN_DAY for t in time ]


plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(lambert_history_matrix[:,1], lambert_history_matrix[:,2], lambert_history_matrix[:,3], label='Lambert Trajectory')
ax.plot3D(dependent_var[:,0], dependent_var[:,1], dependent_var[:,2], label='Earth', linewidth=0.5)
ax.plot3D(dependent_var[:,3], dependent_var[:,4], dependent_var[:,5], label='Mars', linewidth=0.5)

ax.set_xlabel('x [m]')
ax.set_ylabel('y [m]')
ax.set_zlabel('z [m]')
ax.legend()
plt.grid(True)

sc_mass = dependent_var[:,6]

plt.figure()

plt.plot(time_days, sc_mass)

plt.xlim( [min(time_days), max(time_days)])
plt.xlabel( 'Time [days]' )
plt.ylabel('SC Mass [kg]')
plt.grid(True)

plt.show()