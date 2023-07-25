import datetime

from Utilities import *
from tudatpy.kernel.numerical_simulation import environment_setup
from tudatpy.kernel.interface import spice_interface
from tudatpy.kernel.astro import time_conversion
from tudatpy.util import result2array
from matplotlib import pyplot as plt
import matplotlib.cm as cm
from matplotlib import rc
import os
import pickle

compute = False

# Load spice kernels.
spice_interface.load_standard_kernels()

# Define directory where simulation output will be written
output_directory = "./SimulationOutput/"

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 22
specific_impulse = 2500.0
# Fixed parameters
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = 0.0 * constants.JULIAN_DAY

# Time at which to start propagation
n = 200
departure_min = time_conversion.calendar_date_to_julian_day(datetime.datetime(2028, 6, 1))
departure_max = time_conversion.calendar_date_to_julian_day(datetime.datetime(2029, 6, 1))
departure_epoch_range = [(departure_min-constants.JULIAN_DAY_ON_J2000)*constants.JULIAN_DAY, (departure_max-constants.JULIAN_DAY_ON_J2000)*constants.JULIAN_DAY]
departure_epoch_list = np.linspace(departure_epoch_range[0], departure_epoch_range[1], n)
time_of_flight = [150*constants.JULIAN_DAY, 500*constants.JULIAN_DAY]
time_of_flight_list = np.linspace(time_of_flight[0]/constants.JULIAN_DAY, time_of_flight[1]/constants.JULIAN_DAY, n)
C3 = np.zeros([np.size(departure_epoch_list), np.size(departure_epoch_list)])
target_body = 'Mars'
global_frame_orientation = 'ECLIPJ2000'
fixed_step_size = 3600.0

if compute:
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

    # plt.figure()
    # ax = plt.axes(projection='3d')
    #
    # ax.set_xlabel('x [m]')
    # ax.set_ylabel('y [m]')
    # ax.set_zlabel('z [m]')
    # ax.legend()
    # plt.grid(True)

    i = 0

    for departure_epoch in departure_epoch_list:

        j = 0

        arrival_epoch_range = [departure_epoch + time_of_flight[0], departure_epoch + time_of_flight[1]]
        arrival_epoch_list = np.linspace(arrival_epoch_range[0], arrival_epoch_range[1], n)

        for arrival_epoch in arrival_epoch_list:

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
            time_days = [t / constants.JULIAN_DAY - departure_epoch / constants.JULIAN_DAY for t in time]

            # ax.plot3D(lambert_history_matrix[:,1], lambert_history_matrix[:,2], lambert_history_matrix[:,3], label='Lambert Trajectory', color="black")
            # ax.plot3D(dependent_var[:,0], dependent_var[:,1], dependent_var[:,2], label='Earth', linewidth=0.5, color="tab:blue")
            # ax.plot3D(dependent_var[:,3], dependent_var[:,4], dependent_var[:,5], label='Mars', linewidth=0.5, color="orange")

            vinf = dependent_var[0, 7]/1000

            C3[j, i] = vinf**2

            j += 1

            # print("The initial C3 for this transfer is ", C3)

            # plt.show()
        i += 1

    # plt.figure()
    #
    # plt.plot(time_of_flight_list, C3)

    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/departure_epoch_list_4.pickle", "wb") as f:
            pickle.dump(departure_epoch_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/time_of_flight_list_4.pickle", "wb") as f:
            pickle.dump(time_of_flight_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/C3_4.pickle", "wb") as f:
            pickle.dump(C3, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
else:
    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/departure_epoch_list_2.pickle", "rb") as f:
            departure_epoch_list = pickle.load(f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/time_of_flight_list_2.pickle", "rb") as f:
            time_of_flight_list = pickle.load(f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
    try:
        with open("C:/Users/hecto/Desktop/TU Delft/Thesis/SimulationOutput/Lambert/C3_2.pickle", "rb") as f:
            C3 = pickle.load(f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

departure_date_list = [time_conversion.julian_day_to_calendar_date(epoch/constants.JULIAN_DAY+constants.JULIAN_DAY_ON_J2000) for epoch in departure_epoch_list]

X, Y = np.meshgrid(departure_date_list, time_of_flight_list)
Z = C3

plt.rc('font', size=9)  # controls default text size
plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

fig, ax = plt.subplots(figsize=(width, height))
CS = ax.contour(X, Y, Z,levels = [10,15,20,25,30,40,60,100], colors='tab:blue')
CS2 = ax.contourf(X,Y,Z,levels = [0, 15], colors='cornflowerblue')
ax.clabel(CS, inline=True, fontsize=10)
ax.set_xlabel('Departure date')
ax.set_ylabel('Time of flight [days]')


C3_min = np.min(C3)

print("Minimum C3 is ", C3_min)

plt.show()