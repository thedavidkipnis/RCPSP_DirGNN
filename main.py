import rng_funcs as rng
import dirgnn as dgn
import networkx as nx

# to install stuff: py -m pip install scipy

# List representing the attandance probability of given employee 
# Each index represents a day of the week
# 5 indexes - 0 is Monday, 1 is Tuesday, etc.

emp_attendance_probability = [0.65, 0.73, 0.86, 0.89, .8]

'''
During task (node) generation, a time will be pulled at random from this list and assigned to a task
'''
task_baseline_times = []
for i in range(1,29):
    task_baseline_times.append(i*15)


TEAM_COUNT = 4 # used to be 20
EMPLOYEE_COUNT = 10 # used to be 200


class Employee:

    def __init__(self, ID, is_busy, team, exp_years, resource_type):
        self.ID = ID
        self.is_busy = is_busy
        self.team = team
        self.exp_years = exp_years
        self.resource_type = resource_type


class Team:

    def __init__(self, ID, employees, shop_type):
        self.ID = ID
        self.employees = employees
        self.shop_type = shop_type
        # self.shift_start = shift_start
        # self.shift_end = shift_end


'''
Generates employee list
'''
def gen_employees(num_employees):

    # experience years 0-25yrs, skewed towards 0

    employees = {}

    for i in range(num_employees):

        exp = round(rng.emp_exp_years_skewed_dist_gen(), 2)

        temp = Employee(i, False, None, exp, 'B')
        if i > num_employees / 2:
            temp.resource_type = 'A'

        employees[i] = temp

    return employees


'''
Generates teams based on employee list
'''
def gen_teams(num_teams, employees):

    teams = {}
    
    # generating individual teams
    for i in range(num_teams):
        temp_team = Team(i,set(),1)
        if i % 2 == 0:
            temp_team.shop_type = 2

        teams[i] = temp_team

    # assigning employees to teams
    team_counter = 0
    for emp in employees:
        employees[emp].team = team_counter
        teams[team_counter].employees.add(emp)
        team_counter = (team_counter + 1) % num_teams

    return teams


'''
gen_node_deltas: True|False, if marked True, will generate nodes from scratch, otherwise read from 'task_deltas' file
update_task_baseline_deltas: True|False, if marked true will update the task_baseline_deltas.npy file with new values
'''
def run(gen_node_deltas, update_task_baseline_deltas):
    
    # Generating all employees
    EMPLOYEES = gen_employees(EMPLOYEE_COUNT)
    
    # Generating teams and dividing employees into teams
    TEAMS = gen_teams(TEAM_COUNT,EMPLOYEES)

    # Generating DAG
    if gen_node_deltas:
        DAG = dgn.gen_DAG(5, TEAMS, EMPLOYEES, task_baseline_times, update_task_baseline_deltas)
    else:
        DAG = dgn.gen_DAG_from_file('task_baseline_deltas.npy', 'task_edges.npy', TEAMS, EMPLOYEES) # TODO: rewrite to include filepath instead of file name
        
    return DAG

if __name__ == "__main__":
    DAG = run(gen_node_deltas=False, update_task_baseline_deltas=False)
    sc = dgn.rcpsp_solver_with_buffer(DAG, 0, 0)
    print(sc)
    dgn.print_DAG(DAG)
    dgn.display_DAG(DAG)
    dgn.visualize_schedule(DAG)


