'''
Stores all functionality for DAG, DirGNN, and associated processes
'''

import random
import rng_funcs as rng
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from typing import List
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from typing import Dict


'''
gen_DAG helper function for generating and adding nodes to the current layer, then adding them to the DAG
'''
def gen_layer_nodes(teams, employees, task_baseline_time, num_nodes_to_generate_for_cur_layer, node_id_counter, team_idx_counter):
    next_layer = []

    for node in range(num_nodes_to_generate_for_cur_layer):

            '''
            For each node:
                baseline_delta: the pre-simulation, predicted time for the task's completion (stays static during each simulation)
                local_delta: total time to complete task
                global_delta: time delta from start of project until end of current task
                team: which team can execute this task
                exp_coefficient: how much the deltas will be getting multiplied by
                nc_prob: non-conformance probability | chance of error
                nc_occured: whether or not a non-conformance occured during runthrough
            '''
            # generate task base time, assign employee, adjust according to experience
            baseline_delta = random.choice(task_baseline_time)

            employee = random.sample(sorted(teams[team_idx_counter].employees), 1)[0]

            exp_coefficient, nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[employee].exp_years)
            local_delta = round(exp_coefficient * baseline_delta, 5)

            nc_occured = False
            if random.random() < nc_prob:
                nc_occured = True
                local_delta = rng.adjust_local_delta_based_on_nc(local_delta)

            temp_node = (node_id_counter, {'baseline_delta': baseline_delta,
                                           'local_delta': local_delta,
                                            'global_delta': 0, 
                                            'team': team_idx_counter, 
                                            'emp_ID': employee, 
                                            'exp_coefficient' : round(exp_coefficient, 5), 
                                            'nc_prob': round(nc_prob, 5), 
                                            'nc_occured': nc_occured})

            node_id_counter += 1
            next_layer.append(temp_node)
            team_idx_counter = (team_idx_counter + 1) % len(teams)

    return [next_layer, node_id_counter, team_idx_counter]


'''
gen_DAG helper function for generating and adding edges between layers in the DAG during creation
'''
def gen_layer_edges(DAG, cur_layer, next_layer):
    
    edges_to_save = []

    # creating edges between nodes
    if len(cur_layer) == 1: # case: cur_layer size is 1
        for node in next_layer:
            DAG.add_edge(cur_layer[0][0], node[0])
            edges_to_save.append((cur_layer[0][0], node[0]))

    elif len(cur_layer) > len(next_layer): #case: cur_layer size is larger than next_layer
        chance_of_connectivity = len(next_layer) / len(cur_layer)

        if len(next_layer) <= 4: # situational case for making enough connections in cases where size difference between layers is too great
            chance_of_connectivity = 0.5

        connection_found = False
        for node in cur_layer:
            for next_node in next_layer:
                if (random.random() < chance_of_connectivity):
                    connection_found = True
                    DAG.add_edge(node[0], next_node[0])
                    edges_to_save.append((node[0], next_node[0]))

        # if no chance connection was made
        if not connection_found:
            connection_idx_to_add = random.randint(0, len(cur_layer)-1)
            DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])
            edges_to_save.append((cur_layer[connection_idx_to_add][0], next_node[0]))

    else: # case: cur_layer size smaller than or equal to next_layer size
        chance_of_connectivity = 0.5
        for next_node in next_layer:
            connection_found = False
            for node in cur_layer:
                if(random.random() > chance_of_connectivity):
                    DAG.add_edge(node[0], next_node[0])
                    edges_to_save.append((node[0], next_node[0]))
                    connection_found = True

            if not connection_found:
                connection_idx_to_add = random.randint(0, len(cur_layer)-1)
                DAG.add_edge(cur_layer[connection_idx_to_add][0], next_node[0])
                edges_to_save.append((cur_layer[connection_idx_to_add][0], next_node[0]))

    return edges_to_save


'''
Function for generating DAG nodes and edges
'''
def gen_DAG(num_top_layers: int, teams, employees, task_baseline_time, update_task_baseline_deltas):

    # Step 1: start at first node
    # Step 2: using num nodes to top. layer relationship func, determine how many nodes to generate for next "layer"
    # Step 3: generate next layer's nodes
    # Step 3: generate edges between current layer's nodes and next layer's nodes

    # setup
#region
    DAG = nx.DiGraph()

    cur_layer = []
    next_layer = []

    layer_counter = 0
    node_id_counter = 0

    task_deltas = {}
    task_edges = []

    team_idx_counter = 0 # TODO: refactor so that team IDs get assigned, not just counters
#endregion

    for i in range(num_top_layers):

        # emptying next layer
        next_layer.clear()

        # computing number of nodes to generate for following layer based on relationship function
        num_nodes_to_generate_for_cur_layer = rng.node_count_generation_by_top_layer_small(layer_counter)
        layer_counter += 1

        # generating nodes and creating next topological layer
        par_list = gen_layer_nodes(teams, employees, task_baseline_time, num_nodes_to_generate_for_cur_layer, node_id_counter, team_idx_counter)
        next_layer = par_list[0]
        node_id_counter = par_list[1]
        team_idx_counter = par_list[2]

        if update_task_baseline_deltas:
                for node in next_layer:
                    task_deltas[node[0]] = node[1]['baseline_delta']

        DAG.add_nodes_from(next_layer)

        if len(cur_layer) == 0: # edge case for the first node (cur_layer doesn't exist yet)
            cur_layer = next_layer.copy()
            continue
        
        # generating edges between current and next layer
        layer_task_edges = gen_layer_edges(DAG, cur_layer=cur_layer, next_layer=next_layer)

        if update_task_baseline_deltas:
            task_edges += layer_task_edges

        # assigning next layer to be current layer for next iteration
        cur_layer = next_layer.copy()

    if update_task_baseline_deltas:
        np.save('task_baseline_deltas.npy', task_deltas)
        np.save('task_edges.npy', task_edges)

    return DAG

'''
Function for generating DAG from files task_baseline_deltas and task_edges
'''
def gen_DAG_from_file(nodes_file, edges_file, teams, employees):
    try:
        nodes = np.load('task_baseline_deltas.npy', allow_pickle='TRUE').item()
        edges = np.load('task_edges.npy', allow_pickle='TRUE')

    except:
        print(f'Failed to open {nodes_file} and {edges_file}')
        return None

    DAG = nx.DiGraph()

    team_idx_counter = 0
    node_list = []

    # adding nodes
#region
    for node in nodes:
        employee = random.sample(sorted(teams[team_idx_counter].employees), 1)[0]
        exp_coefficient, nc_prob = rng.adjusted_task_time_and_prob_of_error(employees[employee].exp_years)
        local_delta = round(exp_coefficient * nodes[node], 5) # baseline delta = node[1]

        nc_occured = False
        if random.random() < nc_prob:
            nc_occured = True
            local_delta = rng.adjust_local_delta_based_on_nc(local_delta)

        temp_node = (node, {'baseline_delta': nodes[node],
                                           'local_delta': local_delta,
                                            'global_delta': 0, 
                                            'team': team_idx_counter, 
                                            'emp_ID': employee, 
                                            'exp_coefficient' : round(exp_coefficient, 5), 
                                            'nc_prob': round(nc_prob, 5), 
                                            'nc_occured': nc_occured})
        node_list.append(temp_node)
        team_idx_counter = (team_idx_counter + 1) % len(teams)

    DAG.add_nodes_from(node_list)
#endregion

    # adding edges
    for edge in edges:
        DAG.add_edge(edge[0], edge[1])

    return DAG


'''
Sorts the tasks topologically with a random bias for task selection
'''
def topological_sort_with_random_priority(DAG):

    # initializing in degrees to 0 for each node
    task_in_degrees = {}
    queue = set()
    for node in DAG.nodes:
        node_count = len(list(DAG.predecessors(node)))
        task_in_degrees[node] = node_count

        if node_count < 1: # adding all initial tasks that have 0 predecessors
            queue.add(node)

    sort_results = []

    # processing the queue - bucket style!
    while len(queue) > 0:
        task_to_process = random.sample(list(queue), 1)[0]
        sort_results.append(task_to_process)
        queue.remove(task_to_process)

        for successor in DAG.successors(task_to_process):
            task_in_degrees[successor] -= 1

            if task_in_degrees[successor] < 1:
                queue.add(successor)

    return sort_results


'''
Sorts the tasks topologically with a random bias for task selection
'''
def rcpsp_solver_with_buffer(DAG, min_buffer, max_buffer) -> List[int]:

    final_schedule = []
    resource_availability = {}

    for node in DAG.nodes: # all resources are available to start
        resource_availability[DAG.nodes[node]['emp_ID']] = 0

    sorted_tasks = topological_sort_with_random_priority(DAG)
    for task in sorted_tasks:
        resource = DAG.nodes[task]['emp_ID']

        earliest_start = 0
        for predecessor in DAG.predecessors(task): # finding last predecessor that finished
            if DAG.nodes[predecessor]['global_delta'] > earliest_start:
                earliest_start = DAG.nodes[predecessor]['global_delta']

        rand_delay = random.randint(15,420)
        earliest_start = max(resource_availability[resource], earliest_start) + rand_delay

        DAG.nodes[task]['global_delta'] = earliest_start + DAG.nodes[task]['local_delta']
        resource_availability[resource] = DAG.nodes[task]['global_delta'] # TODO: add random buff here

        final_schedule.append((task, earliest_start))

    return final_schedule


'''
Prints DAG to console
'''
def print_DAG(DAG):
    for node in DAG:
        print(node, DAG._node[node])


'''
Displays DAG using pyplot
'''
def display_DAG(DAG):
    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        for node in nodes:
            DAG.nodes[node]["layer"] = layer

    pos = nx.multipartite_layout(DAG, subset_key="layer")

    fig, ax = plt.subplots(figsize=(10,10))
    nx.draw_networkx(DAG, pos=pos, ax=ax, node_size=10, node_color = 'white' , edge_color = 'white', font_color = 'black', with_labels=True)
    ax.set_facecolor('#1AA7EC')
    fig.tight_layout()
    plt.show()


'''
Schedule visualization function
'''
def visualize_schedule(DAG):
    # Extract task information: calculate start, duration, and sort by start time
    tasks = []
    for node, attributes in DAG.nodes(data=True):
        start_time = attributes['global_delta'] - attributes['local_delta']
        duration = attributes['local_delta']
        emp_id = attributes['emp_ID']
        tasks.append((node, start_time, duration, emp_id))
    
    # Sort tasks by start time in ascending order
    tasks.sort(key=lambda x: x[1], reverse=True)  # Sort by start time

    # Prepare for plotting
    fig, ax = plt.subplots(figsize=(15, 10))

    # Extract unique employee IDs for coloring
    unique_emp_ids = sorted(set(task[3] for task in tasks))
    colors = cm.get_cmap('Set1', len(unique_emp_ids))
    emp_color_map = {emp_id: colors(i / len(unique_emp_ids)) for i, emp_id in enumerate(unique_emp_ids)}

    # Plot each task as a horizontal bar
    for i, (task_id, start_time, duration, emp_id) in enumerate(tasks):
        ax.barh(i, duration, left=start_time, height=0.5, align='center',
                color=emp_color_map[emp_id], alpha=0.8, edgecolor='black')
        ax.text(start_time + duration / 2, i, f'Task {task_id}',
                ha='center', va='center', color='black', fontweight='bold', fontsize = 7.5)
    
    # Setting labels for y-axis as task IDs in order of start time
    ax.set_yticks(range(len(tasks)))
    ax.set_yticklabels([f'Task {task[0]}' for task in tasks])
    ax.set_xlabel('Time')
    ax.set_ylabel('Tasks')
    ax.set_title('Task Schedule')
    plt.tight_layout()
    plt.show()


'''
Method for generating edges between layers that are not right next to each other, TODO: complete
'''
def post_process_edge_generation(DAG):

    layers = {}

    for layer, nodes in enumerate(nx.topological_generations(DAG)):
        layers[layer] = []
        for node in nodes:
            DAG.nodes[node]["layer"] = layer
            layers[layer].append(node)