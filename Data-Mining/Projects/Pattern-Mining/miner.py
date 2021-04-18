from tkinter import *
from tkinter import filedialog
import os, sys, time


root = Tk()
root.title('Sequential Frequent Pattern Miner')

Label(root, text='Algorithm').grid(row=0, column=0)
Label(root, text='Minimum support').grid(row=1, column=0)
Label(root, text='Maximum length(optional)').grid(row=2, column=0)
Label(root, text='Input').grid(row=3, column=0)
Label(root, text='Output').grid(row=4, column=0)
label_status = Label(root, text='Status', width=5, bg='gray')
label_status.grid(row=6, column=1)

var_algorithm = StringVar(root, value='GSP')
OptionMenu(root, var_algorithm, 'GSP', 'PrefixSpan').grid(row=0, column=1)

entry_minsup = Entry(root)
entry_maxlength = Entry(root)
entry_input = Entry(root)
entry_output = Entry(root)

entry_minsup.grid(row=1, column=1)
entry_maxlength.grid(row=2, column=1)
entry_input.grid(row=3, column=1)
entry_output.grid(row=4, column=1)

var_includeIntervals = IntVar(root, value=0)
Checkbutton(root, text='Include predictive intervals', variable=var_includeIntervals).grid(row=5, column=0)


def set_io(set_input):
    global entry_input, entry_output

    filepath = filedialog.askopenfilename()
    if set_input:
        entry_input.delete(0, 'end')
        entry_input.insert(0, filepath)
    else:
        entry_output.delete(0, 'end')
        entry_output.insert(0, filepath)

def extract():
    global var_algorithm, var_includeIntervals
    global entry_minsup, entry_maxlength
    global entry_input, entry_output
    global label_status

    label_status.config(bg='gray')
    time.sleep(0.5)

    minsup = 0
    maxlength = -1
    input_filepath = None
    output_filepath = None

    try:
        minsup = float(entry_minsup.get())
        input_filepath = entry_input.get()
        output_filepath = entry_output.get()
    except:
        print('Invalid argument(s)!')
        label_status.config(bg='orange')
        return 1
    
    try:
        maxlength = int(entry_maxlength.get())
    except:
        pass

    if not os.path.isfile(input_filepath):
        print('Invalid input file!')
        label_status.config(bg='purple')
        return 1
    if len(output_filepath) < 1:
        print('Invalid output file!')
        label_status.config(bg='purple')
        return 1
    
    command = f'./extract {input_filepath} {minsup} {maxlength} {var_algorithm.get().lower()} {output_filepath}'
    if var_includeIntervals.get() != '':
        command += f'{var_includeIntervals.get()}'
    print(command)
    # status = os.system(command)
    # if status == 0:
    #     label_status.config(bg='green')
    # else:
    #     label_status.config(bg='red')


Button(root, text='Open', command=lambda: set_io(True)).grid(row=3, column=2)
Button(root, text='Create', command=lambda: set_io(False)).grid(row=4, column=2)
Button(root, text='Extract', command=extract).grid(row=6, column=0)

root.mainloop()