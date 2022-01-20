# standard imports
from matplotlib import pyplot as plt
import numpy as np
import glob

# reach imports
from commonroad_reach.reach.common.util import project_reachset_and_transform_into_cartesian_coordinate

# drivability checker imports
from commonroad_dc.collision.visualization.draw_dispatch import draw_object as draw_object_cc

# commonroad-io imports
from commonroad.visualization.draw_dispatch_cr import draw_object
from commonroad.common.file_reader import CommonRoadFileReader


def open_and_draw_scenario(full_path, time_step=0):
    """
    Opens scenario file from path and draws scenario, planning problem at specified time step
    :param full_path: full path to scenario file
    :param time_step: time step to draw (default: 0)
    """
    files = sorted(glob.glob(full_path))
    crfr = CommonRoadFileReader(files[0])
    scenario, planning_problem_set = crfr.open()
    planning_problem = list(planning_problem_set.planning_problem_dict.values())[0]
    plt.figure(figsize=(20, 10))
    draw_object(scenario, draw_params={'time_begin': time_step,
                                       'scenario': {'lanelet_network': {'lanelet': {'show_label': True}}}})
    draw_object(planning_problem)
    plt.gca().set_aspect('equal')
    plt.autoscale()
    plt.show()


def draw_scenario_and_ref_path(scenario, planning_problem, ref_path=None, step=0):
    """
    Draws scenario, planning problem and reference path at time step current_count
    :param scenario: CommonRoad scenario object
    :param planning_problem: CommonRoad planning problem object
    :param ref_path: reference route to goal region (default None)
    :param step: time step to visualize (default: 0)
    """
    plt.figure(figsize=(20, 10))
    draw_object(scenario, draw_params={'time_begin': step,
                                       'scenario': {'lanelet_network': {'lanelet': {'show_label': True}}}})
    draw_object(planning_problem)
    if ref_path is not None:
        plt.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=20, linewidth=0.6,
                 label='reference path')
    plt.legend()
    plt.gca().set_aspect('equal')
    plt.autoscale()
    plt.show()


def draw_driving_corridor(driving_corridor, vehicle_configuration, time_step=None, plot_limits=None):
    """
    Draws driving corridors
    :param driving_corridor: driving corridor (dict of pycrreach.ReachSetNodes)
    :param vehicle_configuration: vehicle configuration
    :param time_step: time step
    """
    if time_step is None:
        for reachset in driving_corridor.values():
            draw_object_cc(
                obj=project_reachset_and_transform_into_cartesian_coordinate(
                    reachset,
                    vehicle_configuration.coordinates,
                    vehicle_configuration.curvilinear_coordinate_system),
                plot_limits=plot_limits,
                draw_params={'collision': {'polygon': {'facecolor': 'gray', 'edgecolor': 'dimgray', 'zorder': 20, 'opacity': 0.3}}})
    else:
        draw_object_cc(
            project_reachset_and_transform_into_cartesian_coordinate(
                driving_corridor[time_step],
                vehicle_configuration.coordinates,
                vehicle_configuration.curvilinear_coordinate_system),
            draw_params={'collision': {'polygon': {'facecolor': 'gray', 'edgecolor': 'dimgray', 'zorder': 20, 'opacity': 0.3}}})


def draw_planning_results(scenario, ref_path, planned_traj, step, planning_problem=None):
    """
    Draws scenario, planning problem, ref_path and planned trajectory at a certain time step
    :param scenario: CommonRoad scenario object
    :param ref_path: reference route to goal region
    :param planned_traj: planned trajectory
    :param step: time step
    :param planning_problem: CommonRoad planning problem object (default: None)
    """
    draw_object(scenario, draw_params={'time_begin': step})
    if planning_problem is not None:
        draw_object(planning_problem)
    plt.plot(np.array(planned_traj[0]), np.array(planned_traj[1]), color='k', marker='o', markersize=1, zorder=30,
             linewidth=1.0, label='planned trajectories')
    plt.plot(ref_path[:, 0], ref_path[:, 1], color='g', marker='.', markersize=1, zorder=20, linewidth=1.2,
             label='reference path')

