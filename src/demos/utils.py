import pandas as pd
import re

def procesar_expertos(archivo_entrada, archivo_salida):
    data = []

    # Abrir el archivo de texto
    with open(archivo_entrada, 'r', encoding='utf-8') as f:
        for line in f:
            # Saltamos las líneas de encabezado y separadores
            if '|' not in line or 'ID' in line or '---' in line:
                continue
            
            # 1. Separar por la barra vertical |
            parts = line.split('|')
            file_id = parts[0].strip()
            
            # 2. Limpiar y convertir REAL y PRED a números
            # Usamos regex para quitar checks (✓) o cruces (✗) y espacios
            real_val = int(re.sub(r'[^\d]', '', parts[1]))
            pred_val = int(re.sub(r'[^\d]', '', parts[2]))
            
            # 3. Extraer los valores de activación de los expertos (E0-E9)
            # Buscamos el patrón "E(número):(valor)" y nos quedamos con el valor
            experts_raw = parts[3].strip()
            expert_values = [int(v) for v in re.findall(r'E\d+:(\d+)', experts_raw)]
            
            # 4. LÓGICA DE VALIDACIÓN
            # Encontrar el índice del experto con el valor máximo
            max_val = max(expert_values)
            idx_max_expert = expert_values.index(max_val) # Devuelve 0 para E0, 1 para E1...
            
            # ¿Coincide el experto máximo con el valor REAL?
            acierto_real = 1 if idx_max_expert == real_val else 0
            
            # ¿Coincide el experto máximo con la PREDICCIÓN?
            acierto_pred = 1 if idx_max_expert == pred_val else 0
            
            # Construir la fila: ID, REAL, PRED, E0...E9, ACIERTO_REAL, ACIERTO_PRED
            row = [file_id, real_val, pred_val] + expert_values + [acierto_real, acierto_pred]
            data.append(row)

    # Crear los nombres de las columnas
    column_names = ['ID', 'REAL', 'PRED'] + [f'E{i}' for i in range(10)] + ['EXP_VS_REAL', 'EXP_VS_PRED']
    
    # Crear DataFrame y guardar a Excel
    df = pd.DataFrame(data, columns=column_names)
    df.to_excel(archivo_salida, index=False)
    
    # Mostrar resumen rápido en consola
    print(f"--- Proceso completado ---")
    print(f"Archivo guardado como: {archivo_salida}")
    print(f"Total de líneas procesadas: {len(df)}")
    print(f"Precisión del Experto vs Real: {df['EXP_VS_REAL'].mean()*100:.2f}%")
    print(f"Coherencia Experto vs Predicción: {df['EXP_VS_PRED'].mean()*100:.2f}%")

# Ejecutar la función
# Asegúrate de que 'datos.txt' sea el nombre de tu archivo
procesar_expertos('src/demos/siameseSNN/results/biological_pathway_results.txt', 'analisis_final_expertos.xlsx')