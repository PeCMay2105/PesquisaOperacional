from __future__ import print_function
from ortools.linear_solver import pywraplp

solver = pywraplp.Solver('simple_lp_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

lowerBound = [0,0]
upperBound = [4,6]
coe = [3,5]
a = [3,2]
b = [18]

x1 = solver.NumVar(lowerBound[0],upperBound[0],'x1')
x2 = solver.NumVar(lowerBound[1],upperBound[1],'x2')

ct = solver.Constraint(-solver.infinity(),b[0],'ct')


ct.SetCoefficient(x1,a[0])
ct.SetCoefficient(x2,a[1])

objetivo = solver.Objective()

objetivo.SetCoefficient(x1,coe[0])
objetivo.SetCoefficient(x2,coe[1])

objetivo.SetMaximization()

solver.Solve()

print("Solução:")
print(f"Valor objetivo = {objetivo.Value()}")
print(f"x1 = {x1.solution_value()}")
print(f"x2 = {x2.solution_value()}")

