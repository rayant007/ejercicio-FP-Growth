import pandas as pd
from mlxtend.frequent_patterns import fpgrowth, association_rules

# Crear el conjunto de datos
data = [
    ['Bread', 'Peanuts', 'Milk', 'Fruit', 'Jam'],
    ['Bread', 'Jam', 'Soda', 'Chips', 'Milk', 'Fruit'],
    ['Steak', 'Jam', 'Soda', 'Chips', 'Bread'],
    ['Jam', 'Soda', 'Peanuts', 'Milk', 'Fruit'],
    ['Jam', 'Soda', 'Chips', 'Milk', 'Bread'],
    ['Fruit', 'Soda', 'Chips', 'Milk'],
    ['Fruit', 'Soda', 'Peanuts', 'Milk'],
    ['Fruit', 'Peanuts', 'Cheese', 'Yogurt']
]

# Convertir el conjunto de datos a un formato de DataFrame de transacciones
all_items = sorted(set(item for transaction in data for item in transaction))
encoded_data = pd.DataFrame([[1 if item in transaction else 0 for item in all_items] for transaction in data], columns=all_items)

# Aplicar el algoritmo FP-Growth para obtener conjuntos frecuentes
frequent_itemsets = fpgrowth(encoded_data, min_support=0.2, use_colnames=True)

# Generar las reglas de asociación
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)

# Resultados
frequent_itemsets, rules
