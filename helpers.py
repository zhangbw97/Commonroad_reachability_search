# standard imports
import os
import time
import yaml

# third party
import matplotlib.pyplot as plt
import numpy as np
from math import sin, cos
# commonroad-io
from commonroad.visualization.draw_dispatch_cr import draw_object

# commonroad_reach
from commonroad_reach.initialization.initialization import set_up, create_reachset_configuration_vehicle
from commonroad_reach.reach.reachability_single_vehicle_cpp import ReachabilitySingleVehicle
from commonroad_reach.common.util import create_path
from commonroad_reach.visualization.draw_util import draw_solution
from commonroad_reach.planning.qp_planner.collision_avoidance_constraints import CollisionAvoidanceConstraints
from scripts_reach.visualization_gw import draw_driving_corridor
import pycrreach
import pycrccosy
# route planner from commonroad-search (SMP)
from SMP.route_planner.route_planner.route_planner import RoutePlanner, RouteType
# commonroad-ccosy
from commonroad_ccosy.geometry.util import resample_polyline
from SMP.motion_planner.search_algorithms.base_class import State
# commonroad-io
from commonroad.common.file_reader import CommonRoadFileReader
from commonroad_ccosy.geometry.util import compute_curvature_from_polyline
from SMP.maneuver_automaton.motion_primitive import MotionPrimitive
from commonroad.scenario.scenario import Scenario


def open_scenario(settings):
    """
    Open a scenario from .xml file and return the scenario and planning problem set
    """
    # ***********************************
    # open CR scenario and config file
    # ***********************************
    scenario_path = settings['scenario_settings']['scenario_path']
    crfr = CommonRoadFileReader(
        scenario_path + settings['scenario_settings']['scenario_name'] + '.xml')
    scenario, planning_problem_set = crfr.open()
    return scenario, planning_problem_set


def extrapolate_ref_path(reference_path: np.ndarray, resample_step: float = 2.0) -> np.ndarray:
    """
    Function to extrapolate the end of the reference path in order to avoid CCosy errors and/or invalid trajectory
    samples when the reference path is too short.
    :param reference_path: original reference path
    :param resample_step: interval for resampling
    :return extrapolated reference path
    """
    p = np.poly1d(np.polyfit(reference_path[-2:, 0], reference_path[-2:, 1], 1))
    x = 2.3 * reference_path[-1, 0] - reference_path[-2, 0]
    new_polyline = np.concatenate((reference_path, np.array([[x, p(x)]])), axis=0)
    return resample_polyline(new_polyline, step=resample_step)


def compute_reference_area(scenario, planning_problem, route_planner: RoutePlanner,
                           ref_path, reachability_reference, configuration):
    """
    Compute Reachability Analysis and retrieve reachable sets or driving corridor of the given scenario

    :param scenario: Scenario to be processed
    :param planning_problem: planning_problem of the given scenario
    :param route_planner: the route planner initialized for the given scenario
    :param ref_path: reference path used by the motion planner (computed using the route planner)
    :param reachability_reference: choose whether to use Reachable Sets or Driving Corridor as reference area
    :param configuration: the vehicle configuration file for computing reachability analysis

    :returns:
        - reference_area: reference area is either reachable set or longitudinal driving corridor. if not specified, the
        function will return -1.
        - curvilinear_cosy: the used curvilinear coordinate system
        - vehicle_configuration: the vehicle configuration created by reachability analysis
    """
    # set up planning_problem_id in config
    configuration['vehicle_settings'][planning_problem.planning_problem_id] = \
        configuration['vehicle_settings']['problem_id']
    del configuration['vehicle_settings']['problem_id']

    # set up directory to store plots
    configuration["debugging_settings"]["directory"] = os.path.join(configuration["debugging_settings"]["folder_dir"],
                                                                    str(scenario.scenario_id))

    # *******************************
    # Setup Vehicle Configuration
    # *******************************

    # retrieve lanelets leading to goal
    lanelets_to_goal = get_lanelets_ids_leading_to_goal(route_planner, scenario)
    # WORKAROUND: extrapolate to avoid CCosy errors in some scenarios
    ref_path_mod = extrapolate_ref_path(ref_path, resample_step=0.2)

    # set up vehicle configuration
    vehicle_configuration = create_reachset_configuration_vehicle(scenario,
                                                                  route_planner,
                                                                  planning_problem,
                                                                  configuration['vehicle_settings'],
                                                                  configuration['general_reachset_settings']
                                                                  ['time_horizon'],
                                                                  reference_path=ref_path_mod,
                                                                  lanelets_leading_to_goal=lanelets_to_goal,
                                                                  consider_traffic=configuration['scenario_settings']
                                                                  ['consider_traffic'])
    # initialize C++ reachability analysis object
    reachability_analysis = pycrreach.ContinuousReachabilityAnalysis(
        scenario.dt, vehicle_configuration.convert_to_pycrreach_vehicle_parameters())

    # initialize python reachability analysis object
    reachability_single_vehicle = ReachabilitySingleVehicle(reachability_analysis, vehicle_configuration)

    # set start and end time for reachable set simulation
    start_time = vehicle_configuration.initial_time_idx + 1
    end_time = vehicle_configuration.initial_time_idx + configuration['general_reachset_settings']['time_horizon']

    # simulate reachable sets
    t = time.time()
    reachability_single_vehicle.compute_next_time_steps(start_time, end_time)
    # print('elapsed time = {}'.format(time.time() - t))

    # extract longitudinal driving corridor for given horizon
    reachable_set = reachability_single_vehicle.reachable_set
    collision_avoidance_constraints = CollisionAvoidanceConstraints(reachable_set,
                                                                    vehicle_configuration.reference_point)
    curvilinear_cosy = reachability_single_vehicle.vehicle_configuration.curvilinear_coordinate_system

    if reachability_reference == "reachable_set":
        return reachable_set, curvilinear_cosy, vehicle_configuration

    elif reachability_reference == "driving_corridor":
        # get long corridors and select first (largest) corridor
        longitudinal_driving_corridors = collision_avoidance_constraints.driving_corridors(reachable_set, verbose=False)
        try:
            lon_driving_corridor = longitudinal_driving_corridors[0]
        except Exception as e:
            print("Error in retrieving lon_driving_corridor")
            print(e)
            return None, curvilinear_cosy, vehicle_configuration
        else:
            return lon_driving_corridor, curvilinear_cosy, vehicle_configuration
    else:
        print("Please Specify which reference area is chosen")
        return -1


def visualize_reachable_sets(scenario, planning_problem, configuration, vehicle_configuration, start_time, end_time,
                             reachability_single_vehicle, lon_driving_corridor=None):
    # visualize reachable sets  + store in /evaluation
    if (configuration['debugging_settings']['draw_solution'] or
            configuration['debugging_settings']['store_config']):
        full_path = create_path(configuration['debugging_settings']['directory'] + '/')

    if configuration['debugging_settings']['draw_solution']:
        draw_solution(reachability_single_vehicle, scenario, end_time,
                      configuration['debugging_settings']['draw_window'],
                      full_path + '/reachset_')

        if lon_driving_corridor is not None:
            # visualize driving corridor + store in /evaluation
            plt.figure(figsize=(20, 10))
            draw_driving_corridor(lon_driving_corridor, vehicle_configuration)
            draw_object(scenario, draw_params={'time_begin': start_time})
            draw_object(planning_problem)
            plt.gca().set_aspect('equal')
            plt.autoscale()
            plt.savefig(full_path + '/lon_driving_corridor.svg', format='svg', dpi=300, bbox_inches='tight')



def convert_position_to_curvilinear_coordinate(state: State, curvilinear_cosy: pycrccosy.CurvilinearCoordinateSystem):
    """
    Convert the position of a given state from cartesian coordinate to curvilinear coordinate

    :param state: the state to be transformed
    :param curvilinear_cosy: curvilinear coordinate system

    :returns p_curvilinear: the position of the state in curvilinear coordinate system
    """
    [p_x, p_y] = state.position
    p_curvilinear = curvilinear_cosy.convert_to_curvilinear_coords(p_x, p_y)
    # print('Converted p_cartesian in curvilinear coordinates: {}'.format(p_curvilinear))
    return p_curvilinear


def position_within_reachable_set(state: State, reachset_node: pycrreach.ReachSetNode,
                                  curvilinear_cosy: pycrccosy.CurvilinearCoordinateSystem) -> bool:
    """
    check in curvilinear coordinate system if the position of a given state is within the reachable set.

    :param state: the state to be tested
    :param reachset_node: the reference area given as ReachSetNode, either the Reachable Sets or the Driving Corridor at
    the time_step of the given state
    :param curvilinear_cosy: the used curvilinear coordinate system


    :returns    True: when the position is within the ReachSetNode
                False: when the position is outside the ReachSetNode
    """

    p_curvilinear = convert_position_to_curvilinear_coordinate(state, curvilinear_cosy)
    r = reachset_node
    p_curvilinear_min = np.array([r.x_min(), r.y_min()])
    p_curvilinear_max = np.array([r.x_max(), r.y_max()])

    if (p_curvilinear_min[0] <= p_curvilinear[0] <= p_curvilinear_max[0]
            and p_curvilinear_min[1] <= p_curvilinear[1] <= p_curvilinear_max[1]):
        return True
    else:
        return False



def velocity_within_reachable_set(state: State, reachset_node: pycrreach.ReachSetNode,
                                  curvilinear_cosy: pycrccosy.CurvilinearCoordinateSystem) -> bool:
    """
    Check if the velocity of a given state is within reachable set.
    Transfer the velocity of the given state from curvilinear to cartesian coordinate system, and compare the velocity
    constraint of the given REachSetNode with the given state in cartesian coordinate system.

    :param state: the state to be tested
    :param reachset_node: Reachable Sets or Driving Corridor at the time_step of the given state
    :param curvilinear_cosy: used curvilinear coordinate system
    """

    velocity_cartesian = [state.velocity * cos(state.orientation),
                          state.velocity * sin(state.orientation)]

    r = reachset_node
    v_curvilinear_min = [r.v_x_min(), r.v_y_min()]
    v_curvilinear_max = [r.v_x_max(), r.v_y_max()]
    p_curvilinear = curvilinear_cosy.convert_to_curvilinear_coords(state.position[0],
                                                                   state.position[1])
    # tangent and normal vector at the center of drivable area in catesian coordinate
    tangent = curvilinear_cosy.tangent(p_curvilinear[0])
    normal = curvilinear_cosy.normal(p_curvilinear[0])

    # canonical inner product
    longitudinal_velocity = velocity_cartesian[0] * tangent[0] + velocity_cartesian[1] * tangent[1]
    lateral_velocity = velocity_cartesian[0] * normal[0] + velocity_cartesian[1] * normal[1]

    if (v_curvilinear_min[0] <= longitudinal_velocity <= v_curvilinear_max[0] and
            v_curvilinear_min[1] <= lateral_velocity <= v_curvilinear_max[1]):
        return True
    else:
        return False


def get_lanelets_ids_leading_to_goal(route_planner: RoutePlanner, scenario: Scenario):
    """
    Get the ids of the lanelets leading to goal.

    :param route_planner: Route Planner used for the given scenario
    :param scenario: a chosen scenario
    :returns lanelets_to_goal: the ids of lanelets leading to goal
                REGULAR scenarios: the planned lanelets by route and their adjacent lanelets.
                SURVIVAL scenario, the adjacent lanelets and all successors of initial lanelet.
    """
    lanelets_to_goal = []
    id_lanelet_start = route_planner.id_lanelets_start
    if route_planner.route_type == RouteType.REGULAR:

        id_lanelet_goal = route_planner.ids_lanelets_goal
        # choose the route planner type you want to utilize
        route_lanelets_to_goal = route_planner._find_routes_networkx(id_lanelet_start[0], id_lanelet_goal[0])[0]
        lanelet = scenario.lanelet_network.find_lanelet_by_id(route_lanelets_to_goal[0])
        for i in range(len(route_lanelets_to_goal)+1):
            lanelets_to_goal.append(lanelet.lanelet_id)

            if lanelet.adj_right:
                # try to go right
                lanelets_to_goal.append(lanelet.adj_right)
            if lanelet.adj_left:
                # try to go left
                lanelets_to_goal.append(lanelet.adj_left)

            if i < len(route_lanelets_to_goal) - 1:
                lanelet = scenario.lanelet_network.find_lanelet_by_id(route_lanelets_to_goal[i + 1])
            else:
                # no possible route to advance
                break
        lanelets_to_goal = set(lanelets_to_goal)

    elif route_planner.route_type == RouteType.SURVIVAL:
        lanelets = [scenario.lanelet_network.find_lanelet_by_id(id_lanelet_start[0])]
        for lanelet in lanelets:
            lanelets_to_goal.append(lanelet.lanelet_id)
        for step_forward in range(1):
            lanelets_cache = []
            for lanelet in lanelets:
                lanelets_to_goal.append(lanelet.lanelet_id)

                if lanelet.adj_right:
                    # try to go right
                    lanelets_to_goal.append(lanelet.adj_right)
                if lanelet.adj_left:
                    # try to go right
                    lanelets_to_goal.append(lanelet.adj_left)
                if lanelet.successor:
                    # select the successors as the analysed lanelets of next step and add them to the lanelets_to_goal
                    for successor in lanelet.successor:
                        lanelets_cache.append(scenario.lanelet_network.find_lanelet_by_id(successor))
                        lanelets_to_goal.append(successor)

                else:
                    # no possible route to advance
                    break
            lanelets = lanelets_cache
        lanelets_to_goal = set(lanelets_to_goal)
    else:
        raise ValueError("route_type:", route_planner.route_type, "is not defined")
    return lanelets_to_goal
