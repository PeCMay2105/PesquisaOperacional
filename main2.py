from ortools.sat.python import cp_model
import numpy as np

def gym_optimization_test(S=20, M=20, nome_caso="Teste"):
    """
    Função de otimização parametrizada para testes
    
    Parâmetros:
    S: Volume máximo diário  
    M: Máximo de séries por grupo por dia
    nome_caso: Nome do caso de teste
    """
    
    VOLUME_POR_GRUPO = 20  # Volume obrigatório por grupo (fixo)
    X = 5 * VOLUME_POR_GRUPO  # Volume total semanal (100 séries)
    
    print(f"\n🏋️  CASO: {nome_caso.upper()} 🏋️")
    print("="*70)
    print(f"Parâmetros: Volume/grupo={VOLUME_POR_GRUPO} (fixo), S={S} (diário), M={M} (grupo/dia)")
    print(f"Volume total obrigatório: {X} séries")
    print()
    
    # Criar o modelo
    model = cp_model.CpModel()
    
    # Conjuntos
    grupos_musculares = ['Peito', 'Costas', 'Pernas', 'Ombros','Braço']
    dias_semana = ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom']
    
    num_grupos = len(grupos_musculares)
    num_dias = len(dias_semana)
    
    # Variáveis de decisão
    x = {}
    y = {}
    
    for g in range(num_grupos):
        for d in range(num_dias):
            x[g, d] = model.NewIntVar(0, M, f'x[{g},{d}]')
            y[g, d] = model.NewBoolVar(f'y[{g},{d}]')
    
    # Restrições
    # 1. Relação x-y
    for g in range(num_grupos):
        for d in range(num_dias):
            model.Add(x[g, d] <= M * y[g, d])
    
    # 2. Volume EXATO por grupo: Sum(x[g,d]) = VOLUME_POR_GRUPO para cada grupo g
    for g in range(num_grupos):
        model.Add(sum(x[g, d] for d in range(num_dias)) == VOLUME_POR_GRUPO)
    
    # 3. Volume máximo diário
    for d in range(num_dias):
        model.Add(sum(x[g, d] for g in range(num_grupos)) <= S)
    
    # 4. Dias consecutivos
    for g in range(num_grupos):
        for d in range(num_dias - 1):
            model.Add(y[g, d] + y[g, d + 1] <= 1)
        model.Add(y[g, 6] + y[g, 0] <= 1)  # domingo-segunda
    
    # Função objetivo: Minimizar dias de treino
    z = {}
    for d in range(num_dias):
        z[d] = model.NewBoolVar(f'treino_dia_{d}')
    
    # Se há qualquer série no dia d, então z[d] = 1
    for d in range(num_dias):
        total_series_dia = sum(x[g, d] for g in range(num_grupos))
        model.Add(total_series_dia <= S * z[d])
    
    # Minimizar número de dias de treino
    model.Minimize(sum(z[d] for d in range(num_dias)))
    
    # Resolver
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 10.0
    status = solver.Solve(model)
    
    # Resultados
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        dias_treino = int(solver.ObjectiveValue())
        print(f"✅ SOLUÇÃO: {dias_treino} dias de treino (80 séries totais)")
        print(f"Status: {'ÓTIMA' if status == cp_model.OPTIMAL else 'FEASÍVEL'}")
        
        # Tabela compacta
        print(f"\n{'Grupo':<8}", end="")
        for dia in dias_semana:
            print(f"{dia:<5}", end="")
        print("Total")
        print("-" * 50)
        
        total_por_grupo = []
        for g in range(num_grupos):
            nome_grupo = grupos_musculares[g][:7]  # Truncar nome
            print(f"{nome_grupo:<8}", end="")
            total_grupo = 0
            for d in range(num_dias):
                series = solver.Value(x[g, d])
                total_grupo += series
                print(f"{series if series > 0 else '─':<5}", end="")
            print(f"{total_grupo}")
            total_por_grupo.append(total_grupo)
        
        # Total por dia
        print("-" * 50)
        print(f"{'Total':<8}", end="")
        totais_dia = []
        for d in range(num_dias):
            total_dia = sum(solver.Value(x[g, d]) for g in range(num_grupos))
            totais_dia.append(total_dia)
            print(f"{total_dia:<5}", end="")
        print(f"{sum(total_por_grupo)}")
        
        # Análise rápida
        dias_com_treino_real = sum(1 for t in totais_dia if t > 0)
        media_series_dia = sum(total_por_grupo) / dias_com_treino_real if dias_com_treino_real > 0 else 0
        print(f"\n📊 Análise: {dias_com_treino_real} dias ativos, "
              f"média {media_series_dia:.1f} séries/dia ativo")
        
        # Verificar se todos os grupos têm 20 séries
        todos_corretos = all(total == VOLUME_POR_GRUPO for total in total_por_grupo)
        print(f"✅ Todos os grupos com {VOLUME_POR_GRUPO} séries: {'SIM' if todos_corretos else 'NÃO'}")
        
        return True, dias_treino
    
    else:
        print("❌ INFEASÍVEL!")
        dias_minimos = (X + S - 1) // S  # Ceiling division
        print(f"💡 Dica: Com {X} séries e limite diário {S}, precisamos de pelo menos {dias_minimos} dias")
        return False, 0

def executar_todos_os_testes():
    """Executa uma bateria completa de testes"""
    
    print("🧪 BATERIA DE TESTES DO OTIMIZADOR DE TREINO 🧪")
    print("="*70)
    
    # Lista de casos de teste - Agora focando em S (limite diário) e M (limite por grupo/dia)
    casos_teste = [
        # (S, M, nome)
        
        # CASOS BÁSICOS - Volume fixo 80 séries
        (20, 20, "Básico - Limites Generosos"),
        (15, 20, "Limite Diário Moderado"),
        (12, 20, "Limite Diário Restritivo"),
        
        # TESTANDO LIMITE DIÁRIO CRÍTICO
        (10, 20, "Limite Diário Apertado (80÷10=8 dias)"),
        (8, 20, "Limite Diário Muito Apertado"),
        (25, 20, "Limite Diário Generoso"),
        
        # TESTANDO LIMITE POR GRUPO/DIA
        (20, 15, "Máximo por Grupo Moderado"),
        (20, 10, "Máximo por Grupo Restritivo"),
        (20, 8, "Máximo por Grupo Apertado"),
        (20, 5, "Máximo por Grupo Muito Apertado"),
        
        # COMBINAÇÕES INTERESSANTES
        (15, 15, "Ambos Limites Moderados"),
        (12, 12, "Ambos Limites Restritivos"),
        (10, 10, "Ambos Limites Apertados"),
        
        # CASOS REALISTAS
        (18, 12, "Treino Intermediário"),
        (14, 10, "Treino Avançado Focado"),
        (16, 8, "Alta Frequência"),
        
        # CASOS EXTREMOS/LIMITES
        (11, 20, "Mínimo Diário Teórico (80÷7≈11.4)"),
        (5, 20, "Impossível - Diário Muito Baixo"),
        (20, 4, "Impossível - Grupo Muito Baixo"),
        
        # CASOS EFICIENTES
        (20, 6, "Eficiência 4 dias (20×4=80)"),
        (16, 8, "Eficiência 5 dias"),
        (14, 7, "Eficiência 6 dias"),
    ]
    
    resultados = []
    casos_feasiveis = 0
    casos_infeasiveis = 0
    
    for S, M, nome in casos_teste:
        sucesso, dias_treino = gym_optimization_test(S, M, nome)
        resultados.append((nome, S, M, sucesso, dias_treino))
        
        if sucesso:
            casos_feasiveis += 1
        else:
            casos_infeasiveis += 1
        
        print()  # Linha em branco entre casos
    
    # Resumo final
    print("🏆 RESUMO FINAL DOS TESTES 🏆")
    print("="*70)
    print(f"Total de casos testados: {len(casos_teste)}")
    print(f"Casos feasíveis: {casos_feasiveis}")
    print(f"Casos infeasíveis: {casos_infeasiveis}")
    print()
    
    print("📈 RANKING DOS CASOS FEASÍVEIS (por eficiência - menos dias):")
    print("-" * 70)
    casos_feasiveis_ord = [(nome, S, M, dias) for nome, S, M, sucesso, dias in resultados if sucesso]
    casos_feasiveis_ord.sort(key=lambda x: x[3])  # Ordenar por dias (menor = melhor)
    
    for i, (nome, S, M, dias) in enumerate(casos_feasiveis_ord[:10], 1):
        eficiencia = 80 / dias if dias > 0 else 0
        print(f"{i:2d}. {nome:<35} → {dias} dias ({eficiencia:.1f} séries/dia) [S={S}, M={M}]")
    
    return resultados



if __name__ == "__main__":
    # Executar todos os testes
    resultados = executar_todos_os_testes()

    
    