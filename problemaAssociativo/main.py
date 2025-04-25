from __future__ import print_function
from ortools.linear_solver import pywraplp
import math
import random
solver = pywraplp.Solver('simple_lp_program',pywraplp.Solver.GLOP_LINEAR_PROGRAMMING)
objetivo = solver.Objective()


N = int(input("Insira o número de pessoas")) #Número de Pessoas e presentes
preferencias = [[random.random() for _ in range(N)] for _ in range(N)] # Matriz NxN de Pessoas, Preferências
matrizBinaria = [[solver.BoolVar(f'x_{i}_{j}')for j in range(N)] for i in range(N)]

for i in range(N):
  constraint = solver.Constraint(1,1,f'Recebe presente {i}')
  for j in range(N):
    constraint.SetCoefficient(matrizBinaria[i][j],1)
  for j in range(N):
    constraint.SetCoefficient(matrizBinaria[j][i],1)

for i in range(N):
  for j in range(N):
    objetivo.SetCoefficient(matrizBinaria[i][j],preferencias[i][j])
objetivo.SetMaximization()

status = solver.Solve()

if status == pywraplp.Solver.OPTIMAL:
  print("Solução ótima")

  print(f"Valor objetivo: {objetivo.Value()}")
  for i in range(N):
    for j in range(N):
      if matrizBinaria[i][j].solution_value() > 0:
        print(f"Pessoa {i+1} recebeu o presente {j} com preferência igual a {preferencias[i][j]}")
else:
  print("Solução ótima não encontrada")







