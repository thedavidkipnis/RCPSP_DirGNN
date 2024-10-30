import math
import random
import numpy as np
import scipy.stats as sc
from scipy.stats import skewnorm

'''
Input: N/A
Output: Years of experience randomly assigned to an employee during employee generation
'''
def emp_exp_years_skewed_dist_gen():

    # creating a skew-normal random variable
    raw_sample = skewnorm.rvs(6, 0, 1) # rvs(alpha, loc, scale)
    target_min, target_max = 2, 10
    current_min, current_max = -0.4, 2.4  # Approximated from original percentiles
    scaled_sample = (raw_sample - current_min) / (current_max - current_min) * (target_max - target_min) + target_min

    # clips all values in scaled_sample to be within the range provided
    final_sample = np.clip(scaled_sample, 0, 25)
    return final_sample


'''
Input: years of experience of a worker
Output: tuple containing (experience coefficient, prob. of error)
'''
def adjusted_task_time_and_prob_of_error(exp_years: int):

    experience_coefficient = (math.exp(-1*(exp_years - 5))) + 0.3
    prob_of_error = math.log(experience_coefficient + 0.7) / 6.0

    return (experience_coefficient,prob_of_error)


'''
Input: 
- Experience coefficient (how experienced an employee is)
- Task delta (perfect-world time to complete a task)
- Probability of error (how likely an employee is at making a mistake during the task)
Output: Adjusted delta of how long that task took given input
'''
def rng_task_time(experience_coefficient: float, task_delta: float, prob_of_error):

    added_delta = 0.0

    # random error happened
    if random.random() <= prob_of_error:
        added_delta = random.random() * task_delta

    rand_cff = round(random.random(), 4)

    total_delta = (experience_coefficient * sc.norm.ppf(q=rand_cff, loc=task_delta, scale=task_delta/8)) + added_delta

    return total_delta


'''
Input: which topological layer the DAG generator is currently attempting to generate nodes for
Output: the number of nodes needed for that topological layer
'''
def node_count_generation_by_top_layer(topological_layer) -> int:

    A = 0.05
    B = 0.3
    C = 0.3
    D = 1.3
    E = 0.6
    F = 14.1
    G = 0
    k = 0.5
    w = 0.2

    return math.ceil(((A * D ** ( (-k * ((topological_layer-F)/C)) * math.sin(w * ((topological_layer-F)/C)) ** 2 )) * E) / B) + G


'''
Input: which topological layer the DAG generator is currently attempting to generate nodes for
Output: the number of nodes needed for that topological layer
'''
def node_count_generation_by_top_layer_alt(topological_layer):

    P = 0.4
    R = 0.72
    W = -0.04
    Z = 1.8
    i = 207.3
    U = 1.1
    t = 3.5
    q = 3

    return math.ceil(P + R * (Z ** (W * (topological_layer - i) * abs(U * math.sin(((topological_layer/t) - i) / q)))))


def node_count_generation_by_top_layer_small(topological_layer):

    return math.ceil(3 * math.sin(topological_layer-1) + 3.5) 


def adjust_local_delta_based_on_nc(local_delta: float):
    return local_delta + float(np.random.normal(120, 15, 1)[0])


'''
Input: number of trials being run, probability of error
'''
def error_count_for_iterations_and_experience(num_trials: int, prob_of_error: float):

    # TODO: turn q to rng, generate num between 0 and 1 instead of constant .95
    return sc.binom.ppf(q=0.95, n=num_trials, p=prob_of_error)


# diq in butt
