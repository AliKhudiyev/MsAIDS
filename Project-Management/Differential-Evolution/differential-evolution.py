import os
import matplotlib.pyplot as plt
from tkinter import *


def select_goal(option):
    global entry_approximationPoint

    if option == 'Custom point':
        entry_approximationPoint.grid(row=4, column=2)
    else:
        entry_approximationPoint.grid_forget()

def run():
    print('Running')

    global entry_function
    global entry_populationSize
    global entry_f
    global entry_p
    global entry_generationCount
    global goal_option
    global entry_approximationPoint
    global entry_threshold
    global entry_benchmarkRunCount
    global variable_visual
    global variable_multiThread
    global variable_selfAdaptive

    function = entry_function.get()
    population_size = 10
    f = 0.8
    p = 0.9
    generation_count = 100
    approximation_point = None
    threshold = None
    bencmark_run_count = 1
    visual = variable_visual.get()                      # False
    multi_thread = variable_multiThread.get()           # False
    self_adaptive = variable_selfAdaptive.get()         # False

    goal = goal_option.get()

    if len(function) == 0:
        print('No function has been given!')
        return 1
    
    try:
        variable = int(entry_populationSize.get())
        if variable > 0:
            population_size = variable
    except:
        pass

    try:
        variable = int(entry_f.get())
        if 2 >= variable > 0:
            f = variable
    except:
        pass

    try:
        variable = int(entry_p.get())
        if 1 >= variable >= 0:
            p = variable
    except:
        pass

    try:
        variable = int(entry_generationCount.get())
        goal = goal_option.get()
        if variable > 0:
            generation_count = variable
            threshold = None
    except:
        pass

    try:
        variable = float(entry_threshold.get())
        goal = goal_option.get()
        if goal == 'Custom point' and variable >= 0:
            threshold = variable
            generation_count = None
    except:
        pass

    try:
        variable = int(entry_benchmarkRunCount.get())
        if variable > 0:
            bencmark_run_count = variable
    except:
        pass

    # Printing settings
    # print('Function:', function)
    # print('Population size:', population_size)
    # print('F:', f)
    # print('P:', p)
    # print('Number of generations:', generation_count)
    # print('Appriximation point:', approximation_point)
    # print('Threshold:', threshold)
    # print('Benchmark runs:', bencmark_run_count)
    # print('Optimization MT/SA:', f'{multi_thread}/{self_adaptive}')

    dimension = len(set(re.findall('x\[.*?\]', function)))
    # print('Dimension:', dimension)
    args = f'--population={population_size} -f{f} -p{p} --benchmark-run={bencmark_run_count} -o{2*multi_thread+self_adaptive} '
    if generation_count is None:
        args += f'--threshold={threshold} '
    else:
        args += f'--generation={generation_count} '
    
    if goal == 'Global minimum':
        args += '--global-min '
        approximation_point = 0
    elif goal == 'Global maximum':
        args += '--global-max '
        approximation_point = 0
    else:
        approximation_point = float(entry_approximationPoint.get())
    
    if visual:
        args += '--visual '

    os.system(f'./cli.sh {dimension} "{function}" "{args}" {approximation_point}')

    if visual:
        os.system('python src/visualizer.py')


root = Tk()
root.title('Differential Evolution')

frame_main = LabelFrame(root)
frame_main.pack()

label_function = Label(frame_main, text='Function')
label_populationSize = Label(frame_main, text='Population size(opt)')
label_f = Label(frame_main, text='Scaling factor(opt)')
label_p = Label(frame_main, text='Crossover probability(opt)')
label_generationCount = Label(frame_main, text='Number of generations(opt)')
label_approximationPoint = Label(frame_main, text='Goal')
label_threshold = Label(frame_main, text='Error threshold(opt)')
label_benchmarkRunCount = Label(frame_main, text='Number of benchmark runs(opt)')
label_optimization = Label(frame_main, text='Optimization')

entry_function = Entry(frame_main)
entry_populationSize = Entry(frame_main)
entry_f = Entry(frame_main)
entry_p = Entry(frame_main)
entry_generationCount = Entry(frame_main)
entry_approximationPoint = Entry(frame_main)
entry_threshold = Entry(frame_main)
entry_benchmarkRunCount = Entry(frame_main)

variable_visual = IntVar(frame_main)
variable_multiThread = IntVar(frame_main)
variable_selfAdaptive = IntVar(frame_main)
check_visualize = Checkbutton(frame_main, text='Visualize', variable=variable_visual)
check_optimizeMultiThread = Checkbutton(frame_main, text='Multi thread', variable=variable_multiThread)
check_optimizeSelfAdaptive = Checkbutton(frame_main, text='Self-adaptive', variable=variable_selfAdaptive)

goal_option = StringVar(frame_main)
goal_option.set('Global minimum')
menu_goal = OptionMenu(frame_main, goal_option, 'Global minimum', 'Global maximum', 'Custom point', command=select_goal)

button_run = Button(frame_main, text='Run', command=run)

# = = = = = = = = = = = = = = = = =

label_function.grid(row=0, column=0)
label_populationSize.grid(row=1, column=0)
label_f.grid(row=2, column=0)
label_p.grid(row=2, column=2)
label_generationCount.grid(row=3, column=0)
label_approximationPoint.grid(row=4, column=0)
label_threshold.grid(row=5, column=0)
label_benchmarkRunCount.grid(row=6, column=0)
label_optimization.grid(row=7, column=0)

entry_function.grid(row=0, column=1)
entry_populationSize.grid(row=1, column=1)
entry_f.grid(row=2, column=1)
entry_p.grid(row=2, column=3)
entry_generationCount.grid(row=3, column=1)
# entry_approximationPoint.grid(row=4, column=1)
entry_threshold.grid(row=5, column=1)
entry_benchmarkRunCount.grid(row=6, column=1)

check_optimizeMultiThread.grid(row=7, column=1)
check_optimizeSelfAdaptive.grid(row=7, column=2)
check_visualize.grid(row=8, column=0)

menu_goal.grid(row=4, column=1)

button_run.grid(row=10, column=0, columnspan=4)

# = = = = = = = = = = = = = = = = =

root.mainloop()
