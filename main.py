from ortools.sat.python import cp_model
import numpy as np

def gym_optimization():
    # Criar o modelo
    model = cp_model.CpModel()
    
    # =============================================================================
    # PARÂMETROS DO PROBLEMA
    # =============================================================================
    
    # Conjuntos
    grupos_musculares = ['Peito', 'Costas', 'Pernas', 'Ombros']  # G = {0, 1, 2, 3}
    dias_semana = ['Segunda', 'Terça', 'Quarta', 'Quinta', 'Sexta', 'Sábado', 'Domingo']  # D = {0, 1, 2, 3, 4, 5, 6}
    
    num_grupos = len(grupos_musculares)  # 4
    num_dias = len(dias_semana)          # 7
    
    # Parâmetros
    X = 50  # Volume máximo semanal (você pode ajustar)
    M = 20  # Máximo de séries por grupo por dia
    S = 20  # Máximo de séries por dia
    
    print(f"Parâmetros do modelo:")
    print(f"- Volume máximo semanal: {X} séries")
    print(f"- Volume máximo diário: {S} séries")
    print(f"- Máximo por grupo/dia: {M} séries")
    print()
    
    # =============================================================================
    # VARIÁVEIS DE DECISÃO
    # =============================================================================
    
    # x[g,d] = número de séries do grupo g no dia d
    x = {}
    for g in range(num_grupos):
        for d in range(num_dias):
            x[g, d] = model.NewIntVar(0, M, f'x[{grupos_musculares[g]}, {dias_semana[d]}]')
    
    # y[g,d] = 1 se treina grupo g no dia d, 0 caso contrário
    y = {}
    for g in range(num_grupos):
        for d in range(num_dias):
            y[g, d] = model.NewBoolVar(f'y[{grupos_musculares[g]}, {dias_semana[d]}]')
    
    print(f"Variáveis criadas:")
    print(f"- x[g,d]: {num_grupos} × {num_dias} = {num_grupos * num_dias} variáveis inteiras")
    print(f"- y[g,d]: {num_grupos} × {num_dias} = {num_grupos * num_dias} variáveis binárias")
    print()
    
    # =============================================================================
    # RESTRIÇÕES
    # =============================================================================
    
    # 1. Relação entre x e y: x[g,d] <= M * y[g,d]
    for g in range(num_grupos):
        for d in range(num_dias):
            model.Add(x[g, d] <= M * y[g, d])
    
    # 2. Volume máximo semanal: Sum(x[g,d]) <= X
    model.Add(sum(x[g, d] for g in range(num_grupos) for d in range(num_dias)) <= X)
    
    # 3. Volume máximo diário: Sum_g(x[g,d]) <= S para cada dia d
    for d in range(num_dias):
        model.Add(sum(x[g, d] for g in range(num_grupos)) <= S)
    
    # 4. Dias consecutivos: y[g,d] + y[g,d+1] <= 1
    for g in range(num_grupos):
        # Dias 0-5 (segunda a sábado)
        for d in range(num_dias - 1):
            model.Add(y[g, d] + y[g, d + 1] <= 1)
        # Ciclo: domingo (6) + segunda (0)
        model.Add(y[g, 6] + y[g, 0] <= 1)
    
    print(f"Restrições adicionadas:")
    print(f"- Relação x-y: {num_grupos * num_dias} restrições")
    print(f"- Volume semanal: 1 restrição")
    print(f"- Volume diário: {num_dias} restrições") 
    print(f"- Dias consecutivos: {num_grupos * num_dias} restrições")
    print()
    
    # =============================================================================
    # FUNÇÃO OBJETIVO
    # =============================================================================
    
    # Maximizar: Sum(x[g,d]) para todos g,d
    objetivo = sum(x[g, d] for g in range(num_grupos) for d in range(num_dias))
    model.Maximize(objetivo)
    
    print("Função objetivo: Maximizar total de séries na semana")
    print()
    
    # =============================================================================
    # RESOLVER O MODELO
    # =============================================================================
    
    # Criar o solver
    solver = cp_model.CpSolver()
    
    # Configurações opcionais do solver
    solver.parameters.max_time_in_seconds = 60.0  # Limite de 60 segundos
    
    print("Resolvendo o modelo...")
    status = solver.Solve(model)
    
    # =============================================================================
    # RESULTADOS
    # =============================================================================
    
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\n{'='*60}")
        print("SOLUÇÃO ENCONTRADA!")
        print(f"{'='*60}")
        
        print(f"\nValor objetivo: {solver.ObjectiveValue()} séries totais na semana")
        print(f"Status: {'ÓTIMA' if status == cp_model.OPTIMAL else 'FEASÍVEL'}")
        print(f"Tempo de resolução: {solver.WallTime():.2f} segundos")
        
        # Criar matriz de resultados
        print(f"\n{'CRONOGRAMA DE TREINO':^60}")
        print(f"{'='*60}")
        
        # Cabeçalho
        print(f"{'Grupo Muscular':<15}", end="")
        for dia in dias_semana:
            print(f"{dia:<8}", end="")
        print("Total")
        print("-" * 60)
        
        # Resultados por grupo muscular
        total_por_grupo = []
        for g in range(num_grupos):
            print(f"{grupos_musculares[g]:<15}", end="")
            total_grupo = 0
            for d in range(num_dias):
                series = solver.Value(x[g, d])
                total_grupo += series
                if series > 0:
                    print(f"{series:<8}", end="")
                else:
                    print(f"{'─':<8}", end="")
            print(f"{total_grupo}")
            total_por_grupo.append(total_grupo)
        
        # Total por dia
        print("-" * 60)
        print(f"{'Total por dia':<15}", end="")
        for d in range(num_dias):
            total_dia = sum(solver.Value(x[g, d]) for g in range(num_grupos))
            print(f"{total_dia:<8}", end="")
        print(f"{sum(total_por_grupo)}")
        
        # Resumo estatístico
        print(f"\n{'RESUMO ESTATÍSTICO':^60}")
        print(f"{'='*60}")
        for g in range(num_grupos):
            dias_treino = sum(1 for d in range(num_dias) if solver.Value(x[g, d]) > 0)
            print(f"{grupos_musculares[g]}: {total_por_grupo[g]} séries em {dias_treino} dias")
        
    else:
        print("\n❌ PROBLEMA INFEASÍVEL!")
        print("Possíveis causas:")
        print("- Volume semanal muito baixo")
        print("- Volume diário muito restritivo")
        print("- Combinação de restrições impossível de satisfazer")
    
    return model, solver

# =============================================================================
# EXECUTAR O MODELO
# =============================================================================

if __name__ == "__main__":
    print("🏋️  OTIMIZADOR DE TREINO DE FORÇA 🏋️")
    print("="*60)
    print()
    
    modelo, solver = gym_optimization()