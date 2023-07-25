"""
Hector Juan Mari (St. number: 5620325)

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
This function is then used to define a thrust model in the numerical propagation.

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

"""

###########################################################################
# IMPORT STATEMENTS #######################################################
###########################################################################

# General imports
import numpy as np
import os
from matplotlib import pyplot as plt


# Tudatpy imports
from tudatpy.kernel.interface import spice
from tudatpy.io import save2txt
from tudatpy.kernel import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel import numerical_simulation
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.numerical_simulation import propagation_setup
from tudatpy.kernel.math import interpolators
from tudatpy.util import result2array
from tudatpy.kernel.astro import time_conversion
import datetime




# Problem-specific imports
import Utilities as Util

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

initial_time = time_conversion.calendar_date_to_julian_day(datetime.datetime(2026, 11, 15))-constants.JULIAN_DAY_ON_J2000

trajectory_parameters = [initial_time,
                         225,
                         0,
                         8661.95,
                         8425.79,
                         -8632.97,
                         5666.72,
                         -3567.68,
                         -2806.92]

# Choose whether benchmark is run
use_benchmark = True
# Choose whether output of the propagation is written to files
write_results_to_file = True
# Get path of current directory
current_dir = os.path.dirname(__file__)

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 22
specific_impulse = 2400.0
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

###########################################################################
# WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
###########################################################################

# Create problem without propagating
hodographic_shaping_object = Util.create_hodographic_shaping_object(trajectory_parameters,
                                                                    bodies)

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicSemiAnalytical/'
else:
    output_path = None

# Retrieves analytical results and write them to a file
states = Util.get_hodographic_trajectory(hodographic_shaping_object,
                                trajectory_parameters,
                                specific_impulse,
                                output_path)

states_list = result2array(states)

# Retrieve the Earth trajectory over vehicle propagation epochs from spice
earth_states_from_spice = {
    epoch+trajectory_parameters[0]*constants.JULIAN_DAY:bodies.get_body('Earth').state_in_base_frame_from_ephemeris(epoch+trajectory_parameters[0]*constants.JULIAN_DAY)
    for epoch in list(states.keys())
}
# Convert the dictionary to a multi-dimensional array
earth_array = result2array(earth_states_from_spice)

# Retrieve the Mars trajectory over vehicle propagation epochs from spice
mars_states_from_spice = {
    epoch+trajectory_parameters[0]*constants.JULIAN_DAY:bodies.get_body('Mars').state_in_base_frame_from_ephemeris(epoch+trajectory_parameters[0]*constants.JULIAN_DAY)
    for epoch in list(states.keys())
}
# Convert the dictionary to a multi-dimensional array
mars_array = result2array(mars_states_from_spice)


plt.figure()
ax = plt.axes(projection='3d')

ax.plot3D(states_list[:,1], states_list[:,2], states_list[:,3], label='Vehicle', linewidth=1.5)
ax.plot3D(earth_array[:,1], earth_array[:,2], earth_array[:,3], label='Earth', linewidth=0.8)
ax.plot3D(mars_array[:,1], mars_array[:,2], mars_array[:,3], label='Mars', linewidth=0.8)

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20


ax.legend(fontsize='small')
plt.grid(True)

final_dist = np.linalg.norm(states_list[-1,1:4]-mars_array[-1,1:4])

print("The final distance to Mars is", final_dist, "m")

delta_v = hodographic_shaping_object.compute_delta_v()

print("The total delta-V for the transfer is", delta_v, "m/s")

final_dist_earth_mars = np.linalg.norm(earth_array[-1,1:4]-mars_array[-1,1:4])/1.495978707e11

print("The final distance between Earth and Mars is ", final_dist_earth_mars, "AU")

plt.show()
