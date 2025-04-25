from __future__ import print_function
from ortools.linear_solver import pywraplp
solver = pywraplp.Solver('simple_lp_program', pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)

N = int(input())
lb = [0,0]
ub = [] #Ei
li = []
ti = []
t = []
for i in range(0,N):
    ub.append(float(input()))
    li.append(float(input()))
    ti.append(float(input()))
    t.append(float(input()))
xns = []
ct = solver.Constraint(-solver.infinity(),t[0],'ct')
objetivo = solver.Objective()
for i in range(0,N):
    xq = solver.NumVar(lb[i],ub[i],f'x{i+1}')
    ct.SetCoefficient(xq,ti[i])
    objetivo.SetCoefficient(xq,li[i])
    xns.append(xq)
objetivo.SetMaximization()

solver.Solve()
print("Solucao:")
print(f"Valor objetivo: {objetivo.Value()}")
for i in xns:
    print(f"x{i+1} = {xns[i].solution_value()}")
