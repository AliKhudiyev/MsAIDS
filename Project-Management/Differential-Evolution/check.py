from numpy import sin, cos, tan, arcsin, arccos, arctan, arcsinh, arccosh, arctanh
from numpy import pi, e
import sys

variables = list(map(str,input().strip().split()))
function = input()

for variable in variables:
    start = 0
    while True:
        index = function.find(variable, start)
        if index == -1:
            break
            
        start = index + 1
        if index > 0 and function[index-1:index].isalpha():
            continue
        if function[index+len(variable):index+len(variable)+1].isalpha():
            continue
        function = function.replace(function[index:], function[index:].replace(variable, '1', 1))

try:
    print(eval(function))
except:
    sys.exit(1)
