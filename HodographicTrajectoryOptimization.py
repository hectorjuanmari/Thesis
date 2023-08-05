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
import datetime
import pickle


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


# Problem-specific imports
import Utilities as Util
from LowThrustProblem_SO import LowThrustProblem_SO
import pygmo as pg
AU = 1.495978707e11

###########################################################################
# DEFINE HOUSEKEEPING SETTINGS ############################################
###########################################################################

# '1' -
# '2' -
# '3' - n = 7, (2028, 10, 15) - (2029, 4, 15), m = 50, 200 - 700, N_revs = 0, vinf = 4000
# '4' - n = 7, (2028, 10, 15) - (2029, 4, 15), m = 50, 200 - 700, N_revs = 1, vinf = 4000
# '5' - n = 7, (2028, 10, 15) - (2029, 4, 15), m = 50, 200 - 700, N_revs = 0, vinf = 0
# '6' - n = 20, (2028, 10, 1) - (2029, 3, 1), m = 20, 200 - 550, N_revs = 0, vinf = 4000, seed = 666
# '7' - n = 20, (2028, 10, 1) - (2029, 3, 1), m = 20, 200 - 550, N_revs = 0, vinf = 0
# '8' - n = 20, (2028, 10, 1) - (2029, 3, 1), m = 20, 200 - 550, N_revs = 0, vinf = 4000, seed = 42
# '9' - n = 20, (2028, 10, 15) - (2029, 2, 1), m = 20, 200 - 550, N_revs = 0, vinf = 4000, seed = 42
# '10' - n = 20, (2028, 10, 1) - (2029, 2, 1), m = 20, 150 - 550, N_revs = 0, vinf = 4000, seed = 42
# '11' - n = 20, (2028, 10, 1) - (2029, 2, 1), m = 20, 150 - 550, N_revs = 0, vinf = 4000, seed = 221018
# '12' - n = 2, (2028, 11, 20) - (2028, 12, 1), m = 2, 320 - 340, N_revs = 0, vinf = 4000, seed = 42, weight = 0
# '14' - n = 20, (2028, 10, 15) - (2028, 2, 1), m = 20, 150 - 550, N_revs = 0, vinf = 4000, seed = 42, weight = 1000, blowup=1e4, new min. DeltaV
# '15' - n = 20, (2028, 10, 15) - (2028, 2, 1), m = 20, 150 - 550, N_revs = 0, vinf = 4000, seed = 42, weight = 1000, blowup=1e2, new min. DeltaV
# '16' - n = 20, (2028, 10, 15) - (2028, 2, 1), m = 20, 150 - 550, N_revs = 0, vinf = 4000, seed = 221018, weight = 1000, blowup=1e2, new min. DeltaV


exercise = '16'
N_revs = 0
convergence_rate = 1.0  # IN PERCENTAGE! So this is 1%
buffer_time_days = 15
original_population_size = 100
max_number_of_generations = 100

###########################################################################
# DEFINE GLOBAL SETTINGS ##################################################
###########################################################################

# Load spice kernels
spice_interface.load_standard_kernels()

n = 20
initial_time_1 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2028, 10, 15))-constants.JULIAN_DAY_ON_J2000
initial_time_2 = time_conversion.calendar_date_to_julian_day(datetime.datetime(2029, 2, 1))-constants.JULIAN_DAY_ON_J2000
initial_time_list = np.linspace(initial_time_1, initial_time_2, n)
# initial_time_list = [initial_time_1]


m = 20
time_of_flight_1 = 150
time_of_flight_2 = 550
time_of_flight_list = np.linspace(time_of_flight_1, time_of_flight_2, m)
# time_of_flight_list = [time_of_flight_1]


hard_increase = 100/100
soft_increase = 2000/100
constraint_bounds = np.array([1.25e-3])
increase = [soft_increase]

ultimate_constraint_bounds = np.zeros([len(constraint_bounds)])
for k in range(len(increase)):
    ultimate_constraint_bounds[k] = constraint_bounds[k]*(1+increase[k])

constraint_bounds = np.vstack([constraint_bounds, ultimate_constraint_bounds])

trajectory_parameters_global = np.zeros([n, m, 9])
delta_v = np.zeros([n, m])

compute = False
write_results_to_file = True
current_dir = 'C:/Users/hecto/Desktop/TU Delft/Thesis'

###########################################################################
# DEFINE SIMULATION SETTINGS ##############################################
###########################################################################

# Vehicle settings
vehicle_mass = 20
specific_impulse = 2500.0
minimum_mars_distance = 5.0E7
# Time since 'departure from Earth CoM' at which propagation starts (and similar
# for arrival time)
time_buffer = buffer_time_days * constants.JULIAN_DAY

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
        lambda time: np.array([1, 0, 0]), global_frame_orientation, 'VehicleFixed'))

# Adding excess velocity
vinf = [[4000], [0], [0]]

count = 0

if compute:
    i = 0
    for initial_time in initial_time_list:
        j = 0
        for time_of_flight in time_of_flight_list:

            print((count/(m*n)*100), '% Completed')

            decision_variable_range = \
                [[initial_time, time_of_flight, N_revs, -10000, -10000, -10000, -10000, -10000, -10000],
                 [initial_time, time_of_flight, N_revs, 10000, 10000, 10000, 10000, 10000, 10000]]

            current_low_thrust_problem = LowThrustProblem_SO(bodies,
                                                             specific_impulse,
                                                             minimum_mars_distance,
                                                             time_buffer,
                                                             vehicle_mass,
                                                             vinf,
                                                             decision_variable_range,
                                                             constraint_bounds,
                                                             True)

            problema = pg.problem(current_low_thrust_problem)

            seed_list = [666, 4444, 42, 221018, 300321]

            seed_list = [42]

            for current_seed in seed_list:

                seed = current_seed
                population_seed = current_seed
                algorithm_seed = current_seed

                available_algorithms = {  # 'GACO': pg.gaco(seed = algorithm_seed),
                    'DE': pg.de(seed=algorithm_seed),
                    # 'SADE': pg.sade(seed = algorithm_seed),
                    # 'DE1220': pg.de1220(seed = algorithm_seed),
                    # 'GWO': pg.gwo(seed = algorithm_seed),
                    # 'SGA': pg.sga(seed = algorithm_seed),
                    # 'IHS': pg.ihs(seed = algorithm_seed),
                    # 'PSO': pg.pso(seed = algorithm_seed),
                    # 'GPSO': pg.pso_gen(seed = algorithm_seed),
                    # 'SEA': pg.sea(seed = algorithm_seed),
                    # 'SA': pg.simulated_annealing(seed = algorithm_seed),
                    # 'ABC': pg.bee_colony(seed = algorithm_seed),
                    # 'CMA-ES': pg.cmaes(seed = algorithm_seed),
                    # 'xNES': pg.xnes(seed = algorithm_seed),
                }

                algorithm_list = list(available_algorithms.keys())

                for current_algorithm in algorithm_list:

                    if current_algorithm == 'MOEAD':
                        population_size = 105
                    else:
                        population_size = original_population_size

                    print('--- PROBLEM DESCRIPTION ---')
                    print('Number of revolutions: %i' % (N_revs))
                    print('Seed: %i' % (current_seed))
                    print('Population size: %i' % (population_size))
                    print('Number of generations: %i' % (max_number_of_generations))
                    print('Algorithm: %s' % current_algorithm)
                    print('Convergence rate: ' + f"{convergence_rate / 100:.1%}")
                    print('-----------------------------\n')

                    convergence_message = '\nEvolution performed for the maximum number of generations.\nConvergence might have not been achieved.\n\n'

                    print('Creating and evaluating generation 0...')
                    population = pg.population(problema, size=population_size, seed=population_seed)

                    population_dict = dict()
                    fitness_dict = dict()
                    compliance_dict = dict()
                    d_dict = dict()

                    population_dict, fitness_dict = Util.get_fx_dict(population.get_x(), population.get_f(), 0)
                    d_dict[0] = Util.get_COM(np.array(list(fitness_dict.values())))
                    current_state = 0

                    algorithm = pg.algorithm(available_algorithms[current_algorithm])

                    for gen in range(1, max_number_of_generations + 1):

                        print('Evolving population. Generation %i/%i --> %i/%i' % (
                        gen - 1, max_number_of_generations, gen, max_number_of_generations))
                        population = algorithm.evolve(population)

                        current_population, current_fitness = Util.get_fx_dict(population.get_x(), population.get_f(),
                                                                               gen)
                        d_dict[gen] = Util.get_COM(np.array(list(current_fitness.values())))

                        population_dict.update(current_population)
                        fitness_dict.update(current_fitness)

                        current_state, compliance = Util.get_convergence(fitness_dict, convergence_rate, current_state)
                        compliance_dict.update(compliance)

                        converged_generation = gen

                        if current_state == 5:
                            convergence_message = '\nEvolution converged at generation %i\n\n' % (converged_generation)
                            break

                    fitness_list = list(fitness_dict.values())[-100:]
                    population_list = list(population_dict.values())[-100:]
                    ind = fitness_list.index(min(fitness_list))
                    print(fitness_list[ind])
                    trajectory_parameters_global[i, j] = population_list[ind]

                    current_low_thrust_problem = LowThrustProblem_SO(bodies,
                                                                     specific_impulse,
                                                                     minimum_mars_distance,
                                                                     time_buffer,
                                                                     vehicle_mass,
                                                                     vinf,
                                                                     decision_variable_range,
                                                                     constraint_bounds,
                                                                     True)

                    fitness = current_low_thrust_problem.fitness_MO(trajectory_parameters_global[i, j])

                    print(fitness[0])

                    delta_v[i, j] = fitness[0]

                    compliance_dict = dict()

                    pop_aux = list(population_dict.keys())[-100:]

                    for pop in pop_aux:
                        trajectory_parameters = population_dict[pop]
                        initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters, time_buffer)

                        current_low_thrust_problem = LowThrustProblem_SO(bodies,
                                                                         specific_impulse,
                                                                         minimum_mars_distance,
                                                                         time_buffer,
                                                                         vehicle_mass,
                                                                         vinf,
                                                                         decision_variable_range,
                                                                         constraint_bounds,
                                                                         True)

                        fitness = current_low_thrust_problem.fitness_MO(trajectory_parameters)
                        # delta_v[i, j] = fitness[0]
                        if fitness[1] == 0:
                            compliance_dict[pop] = fitness

                    print(convergence_message)

                    datadir = '/OptimizationOutput/Exercise_%s_Date_%s_TOF_%s_Revs_%i_Seed_%i_Algorithm_%s' \
                              % (exercise, i, j, N_revs, seed, current_algorithm)

                    output_path = current_dir + datadir
                    # save2txt(population_dict, 'population.dat', output_path)
                    # save2txt(fitness_dict, 'fitness.dat', output_path)
                    # if len(list(compliance_dict.keys())) != 0: save2txt(compliance_dict, 'compliance.dat', output_path)
                    # save2txt(d_dict, 'd.dat', output_path)
            count += 1
            j += 1
        i += 1
    try:
        datadir = '/OptimizationOutput/delta_v_%s.pickle' \
                  % (exercise)
        output_path = current_dir + datadir
        with open(output_path,
                  "wb") as f:
            pickle.dump(delta_v, f, protocol=pickle.HIGHEST_PROTOCOL)
        datadir = '/OptimizationOutput/trajectory_parameters_%s.pickle' \
                  % (exercise)
        output_path = current_dir + datadir
        with open(output_path,
                  "wb") as f:
            pickle.dump(trajectory_parameters_global, f, protocol=pickle.HIGHEST_PROTOCOL)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)
else:
    try:
        datadir = '/OptimizationOutput/delta_v_%s.pickle' \
                  % (exercise)
        output_path = current_dir + datadir
        with open(output_path,
                  "rb") as f:
            delta_v = pickle.load(f)
        datadir = '/OptimizationOutput/trajectory_parameters_%s.pickle' \
                  % (exercise)
        output_path = current_dir + datadir
        with open(output_path,
                  "rb") as f:
            trajectory_parameters_global = pickle.load(f)
    except Exception as ex:
        print("Error during pickling object (Possibly unsupported):", ex)

# Prepares output path
if write_results_to_file:
    output_path = current_dir + '/SimulationOutput/HodographicOptimization/'
else:
    output_path = None

###########################################################################
# WRITE RESULTS FOR SEMI-ANALYTICAL METHOD ################################
###########################################################################

delta_v = delta_v.transpose()/1000

ind = np.unravel_index(np.argmin(delta_v[:, :], axis=None), delta_v[:, :].shape)

trajectory_parameters_optimum = trajectory_parameters_global[ind[1], ind[0]]

hodographic_shaping_object = Util.create_hodographic_trajectory(trajectory_parameters_optimum,
                                                                                bodies,vinf)

# Retrieves analytical results and write them to a file
hodographic_states = Util.get_hodographic_trajectory(hodographic_shaping_object,
                                current_dir)

hodographic_states_list = result2array(hodographic_states)

# Retrieve the Earth trajectory over vehicle propagation epochs from spice
earth_states_from_spice = {
    epoch:bodies.get_body('Earth').state_in_base_frame_from_ephemeris(epoch)
    for epoch in list(hodographic_states.keys())
}
# Convert the dictionary to a multi-dimensional array
earth_array = result2array(earth_states_from_spice)

# Retrieve the Mars trajectory over vehicle propagation epochs from spice
mars_states_from_spice = {
    epoch:bodies.get_body('Mars').state_in_base_frame_from_ephemeris(epoch)
    for epoch in list(hodographic_states.keys())
}
# Convert the dictionary to a multi-dimensional array
mars_array = result2array(mars_states_from_spice)

plt.rc('font', size=9)  # controls default text size
plt.rc('font', family='serif')
# plt.rc('text', usetex=True)

fig_width_pt = 478 * 1

inches_per_pt = 1 / 72.27

golden_ratio = (5 ** .5 - 1) / 2

width = fig_width_pt * inches_per_pt

height = width * golden_ratio

departure_date_list = [time_conversion.julian_day_to_calendar_date(epoch + constants.JULIAN_DAY_ON_J2000) for epoch in
                       initial_time_list]

fig, ax = plt.subplots(figsize=(width, height))
c1 = ax.pcolormesh(departure_date_list, time_of_flight_list, delta_v[:, :], cmap='coolwarm', vmax=12)
c2 = ax.scatter(departure_date_list[ind[1]], time_of_flight_list[ind[0]], color='tab:green')
fig.colorbar(c1, ax=ax, label='Delta-V [km/s]')
ax.set_xlabel('Date [yyyy-mm]')
ax.set_ylabel('TOF [days]')
fig.tight_layout()


plt.figure(figsize=(width, height))
ax = plt.axes(projection='3d')

ax.plot3D(hodographic_states_list[:,1], hodographic_states_list[:,2], hodographic_states_list[:,3], label='Vehicle', linewidth=1.5)
ax.plot3D(earth_array[:,1], earth_array[:,2], earth_array[:,3], label='Earth', linewidth=0.8, color='tab:green')
ax.plot3D(mars_array[:,1], mars_array[:,2], mars_array[:,3], label='Mars', linewidth=0.8, color='tab:orange')

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20

ax.legend(fontsize='small')
#plt.grid(True)
#plt.tight_layout()
# plt.savefig("hodographic-transfer.png")

print('Minimum Delta-V:', np.min(delta_v[:, :]), "km/s")

print('Initial date for minimum Delta-V:', departure_date_list[ind[1]], ' and TOF:', time_of_flight_list[ind[0]])
print('Initial date for minimum Delta-V:', time_conversion.julian_day_to_calendar_date(trajectory_parameters_optimum[0] + constants.JULIAN_DAY_ON_J2000), ' and TOF:', trajectory_parameters_optimum[1])


# C3 = (np.linalg.norm(states_list[0,4:7]-earth_array[0,4:7])/1000)**2
#
# print("The initial C3 is", C3, "km^2/s^2")

# initial_dist = np.linalg.norm(states_list[0,1:4]-earth_array[0,1:4])
#
# print("The initial distance to Earth is", initial_dist, "m")

# final_dist = np.linalg.norm(states_list[-1,1:4]-mars_array[-1,1:4])
#
# print("The final distance to Mars is", final_dist, "m")

delta_v_leg = hodographic_shaping_object.delta_v_per_leg

print("The delta-V for the transfer is", delta_v_leg[0]/1000, "km/s")

# delta_v_per_node = hodographic_shaping_object.delta_v_per_node
#
# print("The delta-V applied in each node is", delta_v_per_node, "m/s")

# final_dist_earth_mars = np.linalg.norm(earth_array[-1,1:4]-mars_array[-1,1:4])/1.495978707e11
#
# print("The final distance between Earth and Mars is ", final_dist_earth_mars, "AU")

# final_vel_vehicle_mars = np.linalg.norm(states_list[-1,1:4]-mars_array[-1,1:4])
#
# print("The final velocity between the vehicle and Mars is ", final_vel_vehicle_mars, "m/s")

thrust = np.linalg.norm(list(hodographic_shaping_object.rsw_thrust_accelerations_along_trajectory(100).values()),
                                axis=1) * vehicle_mass

# print(thrust*1000)

print("The maximum thrust is ", max(thrust)*1000, "mN")

plt.figure(figsize=(width, height))
ax = plt.axes(projection='3d')

ax.plot3D(earth_array[:,1], earth_array[:,2], earth_array[:,3], label='Earth', linewidth=0.8, color='tab:green')
ax.plot3D(mars_array[:,1], mars_array[:,2], mars_array[:,3], label='Mars', linewidth=0.8, color='tab:orange')

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20
ax.legend(fontsize='small')
plt.grid(True)

for i in range(n):
    ind = np.argmin(delta_v[:, i], axis=None)
    trajectory_parameters = trajectory_parameters_global[i, ind]
    # Time at which to start propagation
    initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters,
                                                                time_buffer)
    # Retrieve termination settings
    termination_settings = Util.get_termination_settings(trajectory_parameters,
                                                         minimum_mars_distance,
                                                         time_buffer)
    # Retrieve dependent variables to save
    dependent_variables_to_save = Util.get_dependent_variable_save_settings()
    # Check whether there is any
    are_dependent_variables_to_save = False if not dependent_variables_to_save else True

    propagator_settings = Util.get_propagator_settings(
        trajectory_parameters,
        bodies,
        initial_propagation_time,
        vehicle_mass,
        termination_settings,
        dependent_variables_to_save,
        current_propagator=propagation_setup.propagator.cowell,
        model_choice=0,
        vinf=vinf)

    propagator_settings.integrator_settings = Util.get_integrator_settings(
        0, 7, 1, initial_propagation_time)
    # Propagate dynamics
    dynamics_simulator = numerical_simulation.create_dynamics_simulator(
        bodies, propagator_settings)

    benchmark_output_path = current_dir + '/SimulationOutput/OptimumTrajectory/' if write_results_to_file else None

    ### OUTPUT OF THE SIMULATION ###
    # Retrieve propagated state and dependent variables
    state_history = dynamics_simulator.state_history
    dependent_variable_history = dynamics_simulator.dependent_variable_history

    # Save results to a dictionary
    simulation_results = [state_history, dependent_variable_history]

    propagation_state_history_list = result2array(state_history)
    dependent_var = result2array(dependent_variable_history)
    time = dependent_var[:, 0]
    time_days = (time - time[0]) / constants.JULIAN_DAY

    ax.plot3D(propagation_state_history_list[:, 1], propagation_state_history_list[:, 2],
              propagation_state_history_list[:, 3], label='Vehicle', linewidth=1.5, color='tab:blue')

    sc_mass = propagation_state_history_list[:, 7]
    delta_v_total = 20 * np.log(sc_mass[0] / sc_mass[-1])

    print('For departure date: ', departure_date_list[i], ' and TOF: ', time_of_flight_list[ind], ', minimum Delta-V (Hodographic): ', delta_v[ind, i], ', minimum Delta-V (propagation): ', delta_v_total)


# Time at which to start propagation
initial_propagation_time = Util.get_trajectory_initial_time(trajectory_parameters_optimum,
                                                            time_buffer)
# Retrieve termination settings
termination_settings = Util.get_termination_settings(trajectory_parameters_optimum,
                                                     minimum_mars_distance,
                                                     time_buffer)
# Retrieve dependent variables to save
dependent_variables_to_save = Util.get_dependent_variable_save_settings()
# Check whether there is any
are_dependent_variables_to_save = False if not dependent_variables_to_save else True

propagator_settings = Util.get_propagator_settings(
    trajectory_parameters_optimum,
    bodies,
    initial_propagation_time,
    vehicle_mass,
    termination_settings,
    dependent_variables_to_save,
    current_propagator=propagation_setup.propagator.cowell,
    model_choice=0,
    vinf=vinf)

propagator_settings.integrator_settings = Util.get_integrator_settings(
    0, 7, 1, initial_propagation_time)
# Propagate dynamics
dynamics_simulator = numerical_simulation.create_dynamics_simulator(
    bodies, propagator_settings)

benchmark_output_path = current_dir + '/SimulationOutput/OptimumTrajectory/' if write_results_to_file else None

### OUTPUT OF THE SIMULATION ###
# Retrieve propagated state and dependent variables
state_history = dynamics_simulator.state_history
dependent_variable_history = dynamics_simulator.dependent_variable_history

# Save results to a dictionary
simulation_results = [state_history, dependent_variable_history]

propagation_state_history_list = result2array(state_history)
dependent_var = result2array(dependent_variable_history)
time = dependent_var[:,0]
time_days = (time-time[0])/constants.JULIAN_DAY

plt.figure(figsize=(width, height))
ax = plt.axes(projection='3d')

ax.plot3D(hodographic_states_list[:,1], hodographic_states_list[:,2], hodographic_states_list[:,3], label='Hodograph', linewidth=1.5, color='tab:red')
ax.plot3D(propagation_state_history_list[:,1], propagation_state_history_list[:,2], propagation_state_history_list[:,3], label='Vehicle', linewidth=1.5, color='tab:blue')
ax.plot3D(earth_array[:,1], earth_array[:,2], earth_array[:,3], label='Earth', linewidth=0.8, color='tab:green')
ax.plot3D(mars_array[:,1], mars_array[:,2], mars_array[:,3], label='Mars', linewidth=0.8, color='tab:orange')

ax.set_xlabel('x [m]')
ax.xaxis.labelpad = 20
ax.set_ylabel('y [m]')
ax.yaxis.labelpad = 20
ax.set_zlabel('z [m]')
ax.zaxis.labelpad = 20
ax.legend(fontsize='small')
plt.grid(True)

sc_mass = propagation_state_history_list[:,7]

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_mass)

plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('SC Mass [kg]')
plt.grid(True)
plt.tight_layout()

sc_thrust = dependent_var[:, 11] * sc_mass
sc_acceleration = np.linalg.norm(
    dependent_var[:, 12:15] - dependent_var[:, 15:18] - dependent_var[:, 18:21],
    axis=1) * sc_mass

plt.figure(figsize=(width, height))

plt.plot(time_days, sc_thrust)
# plt.plot(time_days, sc_acceleration)
plt.xlim([min(time_days), max(time_days)])
plt.xlabel('Time [days]')
plt.ylabel('Acceleration [N]')
plt.legend(
    ['Hodographic thrust profile', 'Accelerations on SC'], loc='upper center', bbox_to_anchor=(0.5, 1.05),
    ncol=3, fancybox=True, shadow=True, fontsize='small')
plt.grid(True)
# plt.yscale('log')
plt.tight_layout()

delta_v_total = 20*np.log(sc_mass[0]/sc_mass[-1])

print('The total delta-V for this maneuver is', delta_v_total, 'km/s')

max_acc = max(dependent_var[:,11]*sc_mass)*1000

print('The maximum thrust force is', max_acc, 'mN')

plt.show()