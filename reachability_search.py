import copy
import numpy as np
import time
import os
import yaml
from typing import Tuple, Dict, Any, List, Union

from commonroad.scenario.trajectory import State

from SMP.motion_planner.node import PriorityNode
from SMP.motion_planner.plot_config import DefaultPlotConfig
from SMP.motion_planner.search_algorithms.best_first_search import AStarSearch
from SMP.motion_planner.utility import MotionPrimitiveStatus, initial_visualization, update_visualization

from SMP.maneuver_automaton.motion_primitive import MotionPrimitive

from SMP.route_planner.route_planner.utils_route import compute_polyline_length
from SMP.route_planner.route_planner.utils_visualization import get_plot_limits_from_routes
from SMP.route_planner.route_planner.route_planner import RoutePlanner

from scripts_reach.helpers import compute_reference_area, position_within_reachable_set, \
    velocity_within_reachable_set, extrapolate_ref_path


def get_vehicle_configuration():
    """
    Load vehicle configuration file for reachability analysis
    """
    # Read configuration
    config_name = "../../batch_processing/reachability_analysis_vehicle_config.yaml"
    config_path = os.path.join(os.path.dirname(__file__), config_name)
    with open(config_path, 'r') as stream:
        try:
            configuration = yaml.load(stream, Loader=yaml.Loader)
        except yaml.YAMLError as exc:
            print(exc)
            return -1
        else:
            return configuration


class ReachabilitySearchPlanner(AStarSearch):
    """
    Motion planner implementation for using Reachability_Analysis.
    This planner is inherited from the AStarSearch planner.
    """

    def __init__(self, scenario, planningProblem, automata, plot_config=DefaultPlotConfig,
                 number_heuristics=10, max_speed=20,
                 reachability_reference="driving_corridor", change_survival_to_goal=True):
        """
        Initialize MotionPlanner

        :param number_heuristics: Number of heuristic used. This only has to be set larger than the actual used number
            of heuristics. Since it all values are initialized with zeros, the unused places will simply have no effect
            on the cost.
        :param max_speed: maximal allowed velocity of the vehicle. Used for computing the fastest time required to reach
            goal.
        :param reachability_reference: Choose which reachability analysis result to use as reference area.
            "reachable_set": using Reachable Sets
            "driving_corridor": using Driving Corridor
        :param change_survival_to_goal: choose whether to manually give survival scenario a goal lanelet for computing
            reachable set.
        """
        super().__init__(scenario=scenario, planningProblem=planningProblem, automaton=automata,
                         plot_config=plot_config)
        ############################ SETTINGS FOR VISUALIZATION ##############################
        # Switch visualization
        plot_config.DO_PLOT = True  # Enable/Disable Plot
        plot_config.REACHABILITY = True  # Plot Driving Corridor
        plot_config.PLOT_REF_PATH = True  # Plot Reference Path
        plot_config.PLOT_ROUTE = False  # Plot Planned Route

        # Comment out to show plot during runtime
        # Uncomment to save plots to directory path_fig
        # self.path_fig = f"../../outputs/fig/{scenario.scenario_id}/"
        ######################################################################################

        self.mode = "Goal"
        self.max_speed = max_speed
        self.reachability_reference = reachability_reference
        self.reachability_region = None
        self.average_velocity_required = None
        self.change_survival_to_goal = change_survival_to_goal
        # Retrieve goal information and chose which heuristic to use based on scenario type
        try:
            goal = self.planningProblem.goal.state_list[0]
        except Exception as e:
            print("Something went wrong with fetching the Goal Information")
            print(e)
        else:
            if not hasattr(goal, 'position'):
                # Survival
                self.mode = "Survival"
            elif hasattr(goal.position, 'shapes'):
                # Goal_Region is Shape
                self.position_desired = self.calc_goal_interval(goal.position.shapes[0].vertices)

        # Initialize weights and costs
        self.weights = np.ones(number_heuristics)
        self.init_weights()
        self.cost = np.zeros(number_heuristics)

        # Set up Route Planner
        # Choose whether to change SURVIVAL scenarios to REGULAR by setting a successor lanelet as the goal
        self.route_planner = RoutePlanner(scenario, planningProblem,
                                          backend=RoutePlanner.Backend.NETWORKX_REVERSED,
                                          change_survival_to_regular=self.change_survival_to_goal)
        candidate_holder = self.route_planner.plan_routes()
        self.route = candidate_holder.retrieve_first_route()
        # TODO: Maybe replace WORKAROUND?
        self.ref_path_mod = extrapolate_ref_path(self.route.reference_path, resample_step=0.2)

        # Calculate Required Average Velocity if goal position and target time_step are specified.
        if self.position_desired and self.time_desired.end is not np.inf:
            ref_path_length = compute_polyline_length(self.route.reference_path)
            self.average_velocity_required = ref_path_length / self.time_desired.end

        # Reachability Analysis
        # Timer
        ra_time_start = time.time()
        configuration = get_vehicle_configuration()
        try:
            self.reachability_region, self.curvilinear_cosy, self.vehicle_configuration = \
                compute_reference_area(scenario,
                                       planningProblem,
                                       self.route_planner,
                                       self.route.reference_path,
                                       self.reachability_reference,
                                       configuration)
        except Exception as e:
            print("Reachability Analysis - ERROR IN SCENARIO:", scenario.scenario_id)
            print(e)

        # Timer
        self.RA_time = time.time() - ra_time_start
        print('Reachability_Analysis Timer = {:.4f} s'.format(self.RA_time))

        # Setting PLot Limits using the RoutePlanner
        self.plot_limits = None
        if plot_config.DO_PLOT and plot_config.REACHABILITY:
            # option 1: plot limits from reference path
            # self.plot_limits = get_plot_limits_from_reference_path(self.route)
            # option 2: plot limits from lanelets in the route
            self.plot_limits = get_plot_limits_from_routes(self.route)

    def execute_search(self) -> Tuple[Union[None, List[List[State]]], Union[None, List[MotionPrimitive]], Any, Any]:
        """
        Implementation of tree search using a Priority queue.
        The evaluation function of each child class is implemented individually.
        """
        # for visualization in jupyter notebook
        list_status_nodes = []
        dict_status_nodes: Dict[int, Tuple] = {}

        node_initial = PriorityNode(list_paths=[[self.state_initial]],
                                    list_primitives=[self.motion_primitive_initial], depth_tree=0, priority=0)

        initial_visualization(self.scenario, self.state_initial, self.shape_ego,
                              self.planningProblem, self.config_plot, self.path_fig,
                              lon_driving_corridor=self.reachability_region,
                              vehicle_configuration=self.vehicle_configuration,
                              plot_limits=self.plot_limits,
                              route=self.route)

        # add current node (i.e., current path and primitives) to the frontier
        f_initial = self.evaluation_function(node_initial)
        self.frontier.insert(item=node_initial, priority=f_initial)

        dict_status_nodes = update_visualization(primitive=node_initial.list_paths[-1],
                                                 status=MotionPrimitiveStatus.IN_FRONTIER,
                                                 dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                 config=self.config_plot,
                                                 count=len(list_status_nodes),
                                                 lon_driving_corridor=self.reachability_region,
                                                 vehicle_configuration=self.vehicle_configuration,
                                                 plot_limits=self.plot_limits,
                                                 time_step=node_initial.list_paths[-1][-1].time_step)
        list_status_nodes.append(copy.copy(dict_status_nodes))

        while not self.frontier.empty():
            # pop the last node
            node_current = self.frontier.pop()

            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.CURRENTLY_EXPLORED,
                                                     dict_node_status=dict_status_nodes,
                                                     path_fig=self.path_fig, config=self.config_plot,
                                                     count=len(list_status_nodes),
                                                     lon_driving_corridor=self.reachability_region,
                                                     vehicle_configuration=self.vehicle_configuration,
                                                     plot_limits=self.plot_limits,
                                                     time_step=node_current.list_paths[-1][-1].time_step)
            list_status_nodes.append(copy.copy(dict_status_nodes))

            # goal test
            if self.reached_goal(node_current.list_paths[-1]):
                path_solution = self.remove_states_behind_goal(node_current.list_paths)
                list_status_nodes = self.plot_solution(path_solution=path_solution, node_status=dict_status_nodes,
                                                       list_states_nodes=list_status_nodes)
                return path_solution, node_current.list_primitives, list_status_nodes, self.RA_time

            # check all possible successor primitives(i.e., actions) for current node
            for primitive_successor in node_current.get_successors():

                # translate/rotate motion primitive to current position
                list_primitives_current = copy.copy(node_current.list_primitives)
                path_translated = self.translate_primitive_to_current_state(primitive_successor,
                                                                            node_current.list_paths[-1])
                
                # TODO: Use Reachability Analysis to skip nodes directly: PART 1
                ########### UNCOMMENT THIS TO USE NODE PRUNING METHOD #############
                # check for reachability, if is not reachable it is skipped
                # if not self.check_reachability_skip_node(path_translated[-1]) == 0:

                # continue
                ###################################################################

                # check for collision, if is not collision free it is skipped
                if not self.is_collision_free(path_translated):
                    list_status_nodes, dict_status_nodes = \
                        self.plot_colliding_primitives(current_node=node_current,
                                                       path_translated=path_translated,
                                                       node_status=dict_status_nodes,
                                                       list_states_nodes=list_status_nodes)
                    continue

                list_primitives_current.append(primitive_successor)

                path_new = node_current.list_paths + [[node_current.list_paths[-1][-1]] + path_translated]
                node_child = PriorityNode(list_paths=path_new,
                                          list_primitives=list_primitives_current,
                                          depth_tree=node_current.depth_tree + 1,
                                          priority=node_current.priority)
                f_child = self.evaluation_function(node_current=node_child)
                # insert the child to the frontier:
                dict_status_nodes = update_visualization(primitive=node_child.list_paths[-1],
                                                         status=MotionPrimitiveStatus.IN_FRONTIER,
                                                         dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                         config=self.config_plot,
                                                         count=len(list_status_nodes),
                                                         lon_driving_corridor=self.reachability_region,
                                                         vehicle_configuration=self.vehicle_configuration,
                                                         plot_limits=self.plot_limits,
                                                         time_step=node_child.list_paths[-1][-1].time_step)
                list_status_nodes.append(copy.copy(dict_status_nodes))
                self.frontier.insert(item=node_child, priority=f_child)

            dict_status_nodes = update_visualization(primitive=node_current.list_paths[-1],
                                                     status=MotionPrimitiveStatus.EXPLORED,
                                                     dict_node_status=dict_status_nodes, path_fig=self.path_fig,
                                                     config=self.config_plot,
                                                     count=len(list_status_nodes))
            list_status_nodes.append(copy.copy(dict_status_nodes))
        return None, None, list_status_nodes, self.RA_time

    def evaluation_function(self, node_current: PriorityNode) -> float:
        # copied the implementation in AStarSearch
        if self.reached_goal(node_current.list_paths[-1]):
            node_current.list_paths = self.remove_states_behind_goal(node_current.list_paths)
        # calculate g(n)
        node_current.priority += (len(node_current.list_paths[-1]) - 1) * self.scenario.dt

        # f(n) = g(n) + h(n)
        return node_current.priority + self.heuristic_function(node_current=node_current)

    def heuristic_function(self, node_current: PriorityNode) -> float:
        """
        Current heuristic function check firstly the reachability of the node. If the current node is reachable, the
        other heuristics are calculated.
        """
        # Goal Test
        if self.reached_goal(node_current.list_paths[-1]):
            return 0.0

        # TODO: Use Reachability Analysis to skip nodes directly: PART 2
        # For using the reachability analysis to skip nodes directly, comment out this method.
        # Check whether last state of current primitive is within the reference area
        if self.check_reachability_heuristic(node_current) is np.inf:
            return np.inf

        # Calculate Heuristic Cost
        self.calc_heuristic(node_current)

        ################# Print Detailed Costs for DEBUGGING #################
        # self.print_costs()
        # print("Cost:\t\t\t\t\t\t", self.weights @ self.cost)
        # print()
        ######################################################################

        return self.weights @ self.cost

    def check_reachability_skip_node(self, state: State):
        """
        Check Reachability for skipping the node

        :param state: the state which is to be checked
        """
        time_step = state.time_step
        return self.check_reachability(state, time_step)

    def check_reachability_heuristic(self, node_current: PriorityNode):
        """
        Check Reachability for calculating heuristic

        :param node_current: the current node which is to be checked
        """
        final_state = node_current.list_paths[-1][-1]
        time_step = node_current.list_paths[-1][-1].time_step
        return self.check_reachability(final_state, time_step)

    def check_reachability(self, state: State, time_step):
        """
        Check whether the given state of a node is within the reference area of reachability analysis

        :param state: the state which is to be checked
        :param time_step: the time_step of the given state
        """
        # We assume the first initial node is always reachable, this is indeed always true
        # (implemented to workaround the BUG where the first node is always outside of the driving corridor)
        # TODO: Investigate the reason for this bug and do bugfix instead of workaround
        if time_step is 0:
            return 0

        # Chose the time_step of the reference area to compare with
        reachable_set_time_slice = self.reachability_region[time_step]
        # Loop through all polygons in this time_step
        for reachset_node in reachable_set_time_slice:
            position_checker = position_within_reachable_set(state,
                                                             reachset_node,
                                                             self.curvilinear_cosy)

            if not position_checker:
                continue
            elif not velocity_within_reachable_set(state, reachset_node, self.curvilinear_cosy):
                continue
            else:
                return 0
        return np.inf

    def calc_heuristic(self, node_current):
        """
        Calculate the heuristic cost of a given node

        This heuristic use high level planning and compare always the cost of the last state of the primitive
        1 - Euclidean Distance to Reference Path:
            Penalize the ego vehicle for deviating from the reference path. If the vehicle deviate too far from the
            reference path (default 3 m), the node is set to be costing INF.
        2 - Orientation to Reference Path:
            Penalize the ego vehicle for driving in different direction than the reference path. This heuristic aims to
            help the ego vehicle make turns at the desired road crossing.

        Heuristics used for scenarios with goal specified:
        3 - Euclidean Distance to Goal:
            Penalize the ego vehicle for being far away from goal.
        4 - Time Left
            Penalize the ego vehicle for not choosing going back to earlier nodes.
            - If the the average velocity required for the ego vehicle to drive from the current position to goal is
            larger than the maximal velocity of the ego vehicle, the node is considered unsuitable, since the ego
            vehicle can never reach the goal from there in time.
            - If the time left for the current node is smaller than zero, the node is considered unsuitable, since the
            time limit is exceeded and the goal is not reached.
        5 - Average Velocity Gap:
            Penalize the ego vehicle for driving too fast or too slow. The ego vehicle should drive with a desired
            average velocity, so that it reaches goal in time, not too soon or too late.
        6 - Time to Goal:
            Penalize the ego vehicle for needing longer to reach the goal. This heuristic computes the time needed for
            the ego vehicle to drive from current node with the current velocity to goal.
            - This is a bad heuristic which contradicts with (4 - Time Left) and strike for faster driving. Should be
            changed or deleted.
        7 - Angle to Goal:
            Penalize the ego vehicle for driving in a different direction than the desired orientation in goal region.
            This heuristic also helps the ego vehicle to make turns.
            - This heuristic can be further improved by adding activation condition. We can activate this heuristic,
            only when the ego vehicle is on the same lanelet as the goal region.
        8 - Manhattan Distance to Goal:
            Penalize the ego vehicle for being far away from goal.

        Heuristics used for survival scenarios:
        3 - Velocity:
            Penalize the ego vehicle for driving slowly.
        4 - Time Left
            Penalize the ego vehicle for not choosing going back to earlier nodes.
        """
        # Last state of primitive
        end_state = node_current.list_paths[-1][-1]

        # 1 - Euclidean Distance to Reference Path
        dist_to_ref_path = self.calc_distance_to_nearest_point(self.ref_path_mod,
                                                               end_state.position)
        # Prune Node that are too far away from reference path
        if dist_to_ref_path > 3:
            return np.inf
        self.cost[0] = dist_to_ref_path

        # 2 - Orientation to Reference Path
        self.cost[1] = self.calc_orientation_diff_to_route(end_state)

        # Goal is Position or ShapeGroup
        if self.mode is "Goal":
            # 3 - Euclidean Distance to Goal
            dist_to_goal = self.calc_euclidean_distance(current_node=node_current)
            self.cost[2] = dist_to_goal

            # 4 - Time Left
            if self.time_desired is not None:
                time_left = self.time_desired.start - end_state.time_step
                if dist_to_goal / time_left > self.max_speed:
                    return np.inf
                elif time_left < 0:
                    return np.inf
                else:
                    self.cost[3] = time_left
                    # If the ego vehicle is within the goal region, reduce the weights for all other costs
                    if dist_to_goal <= 0:
                        for w in self.weights:
                            w /= 3
                        self.weights[3] = 1

            # 5 - Average Velocity Gap
            if self.average_velocity_required is not None:
                ave_vel = self.calc_average_velocity(node_current)
                self.cost[4] = (abs(self.average_velocity_required - (ave_vel + 1)) * 10) ** 2

            # 6 - Time to Goal
            # TODO: Compare the time needed to reach the goal with the available time left would be more consistent
            velocity = end_state.velocity
            if velocity < 0:
                self.cost[5] = np.inf
            elif np.isclose(velocity, 0):
                self.cost[5] = 10
            else:
                self.cost[5] = dist_to_goal / velocity

            # 7 - Angle to Goal
            angle_to_goal = self.calc_angle_to_goal(end_state)
            self.cost[6] = np.abs(angle_to_goal)

            # 8 - Manhattan Distance to Goal
            dist_to_goal_manhattan = self.calc_manhattan_distance(current_node=node_current)
            self.cost[7] = dist_to_goal_manhattan

        # Survival
        elif self.mode is "Survival":
            # 3 - Velocity
            vel_dif = 100 - end_state.velocity
            self.cost[2] = vel_dif

            # 4 - Time Left
            if self.time_desired is not None:
                time_left = self.time_desired.start - end_state.time_step
                if time_left < 0:
                    self.cost[3] = 0
                else:
                    self.cost[3] = time_left

    def init_weights(self):
        """
        Initialize weights for various heuristic functions
        Details about heuristic functions see calculate_heuristic()
        All weights are currently manually adjusted and should be optimized in the future.
        """
        if self.mode is "Goal":
            # Weight for Distance to Ref_Path
            self.weights[0] = 2
            # Weight for Orientation to ref_path
            self.weights[1] = 3
            # Weight for Time Left
            self.weights[3] = 1
            # Weight for Average Velocity Gap
            self.weights[4] = 5
            # Weight for needed Time to Goal
            self.weights[5] = 2
            # Weight for Angle to Goal
            self.weights[6] = 3

            # If turn is needed
            if np.abs(self.calc_angle_to_goal(self.state_initial)) >= 1.05:
                # Weight for Angle to Goal
                self.weights[6] = 5
                # Weight for Manhattan Distance to Goal
                self.weights[7] = 2


        if self.mode is "Survival":
            # Weight for Distance to Ref_Path
            self.weights[0] = 1
            # Weight for Time Left
            self.weights[3] = 5

    def print_costs(self):
        """
        Print the detailed cost of each node for DEBUGGING
        """
        print("Distance to ref_path:\t\t", "{:.4f}".format(self.cost[0]), "\t\t", self.cost[0] * self.weights[0])
        print("Orientation to ref_path:\t", "{:.4f}".format(self.cost[1]), "\t\t", self.cost[1] * self.weights[1])
        if self.mode is 0 or self.mode is 2:
            print("Average Velocity:\t\t\t", "{:.4f}".format(self.cost[2]), "\t\t", self.cost[2] * self.weights[2])
            print("Time to Goal:\t\t\t\t", "{:.4f}".format(self.cost[3]), "\t\t", self.cost[3] * self.weights[3])
            print("Angle to Goal:\t\t\t\t", "{:.4f}".format(self.cost[4]), "\t\t", self.cost[4] * self.weights[4])
            print("Manhattan Distance:\t\t\t", "{:.4f}".format(self.cost[5]), "\t\t", self.cost[5] * self.weights[5])
            print("Time Left:\t\t\t\t\t", "{:.4f}".format(self.cost[6]), "\t\t", self.cost[6] * self.weights[6])
        else:
            print("Velocity Gap:\t\t\t\t", "{:.4f}".format(self.cost[2]), "\t\t", self.cost[2] * self.weights[2])
            print("Time Left:\t\t\t\t\t", "{:.4f}".format(self.cost[3]), "\t\t", self.cost[3] * self.weights[3])
        print()
