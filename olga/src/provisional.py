import pandas as pd
import os
import sys

# -------------------------------------------------------------------------
# 1. CONFIGURACIÓ I IMPORTS
# -------------------------------------------------------------------------
# Assegurem que Python troba els mòduls veïns
sys.path.append(os.path.dirname(__file__))

from operador_ingredients import substituir_ingredients_prohibits, adaptar_plat_a_estil
# Nota: En importar això, es carregarà el model FlavorGraph (trigarà uns segons)

# -------------------------------------------------------------------------
# 2. CARREGAR DADES (Simulació de la Memòria del Sistema)
# -------------------------------------------------------------------------
print("\n--- 1. CARREGANT DADES ---")

BASE_DIR = os.path.join(os.path.dirname(__file__), '..') # Pugem un nivell a l'arrel
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Carreguem la Base d'Ingredients (Ontologia)
csv_ingredients = os.path.join(DATA_DIR, 'ingredients_en.csv')
base_ingredients = []

try:
    df = pd.read_csv(csv_ingredients)
    # Convertim a llista de diccionaris per passar-ho a les funcions
    base_ingredients = df.to_dict(orient='records')
    print(f"✅ Ingredients carregats: {len(base_ingredients)} entrades.")
except FileNotFoundError:
    print(f"❌ ERROR: No s'ha trobat {csv_ingredients}.")
    print("   Assegura't d'executar l'script des de la carpeta correcta.")
    sys.exit()

# Carreguem la Base d'Estils (per provar l'adaptació)
csv_estils = os.path.join(DATA_DIR, 'estils.csv')
base_estils = {}

try:
    df_estils = pd.read_csv(csv_estils)
    # Convertim "ing1|ing2" a llistes reals si cal
    for _, row in df_estils.iterrows():
        nom = row['nom_estil']
        ings = str(row['ingredients']).split('|') if isinstance(row.get('ingredients'), str) else []
        base_estils[nom] = row.to_dict()
        base_estils[nom]['ingredients'] = ings
    print(f"✅ Estils carregats: {len(base_estils)} entrades.")
except Exception as e:
    print(f"⚠️  Avís: No s'ha pogut carregar estils ({e}). Les proves d'estil poden fallar.")


# -------------------------------------------------------------------------
# 3. TESTS UNITARIS
# -------------------------------------------------------------------------

def imprimir_resultat(titol, original, modificat):
    print(f"\n🔹 {titol}")
    print(f"   Plat Original: {original['ingredients']}")
    print(f"   Plat Modificat: {modificat['ingredients']}")
    if 'log_transformacio' in modificat and modificat['log_transformacio']:
        print("   📝 LOG CANVIS:")
        for log in modificat['log_transformacio']:
            print(f"      - {log}")
    else:
        print("   ℹ️  Sense canvis.")

# --- CAS 1: AL·LÈRGIA (Substitució directa) ---
# Objectiu: Substituir 'gambes' per un altre marisc similar.
# Nota: Canvia 'shrimp' pel nom exacte que tinguis al teu ingredients.csv (anglès o català)
plat_marisc = {
    "nom": "Paella de Marisc",
    "ingredients": ["rice", "shrimp", "squid", "peas"], 
    "curs": "segon"
}
prohibits = {"shrimp"} # Usuari al·lèrgic a les gambes

resultat_1 = substituir_ingredients_prohibits(plat_marisc, prohibits, base_ingredients)
imprimir_resultat("TEST 1: Al·lèrgia (Marisc)", plat_marisc, resultat_1)


# --- CAS 2: ADAPTACIÓ A ESTIL (Prohibició per context) ---
# Objectiu: El 'bacon' no és halal. S'hauria de canviar per 'beef' o 'turkey' (si són a la mateixa categoria).
plat_bacon = {
    "nom": "Ous amb Bacon",
    "ingredients": ["egg", "bacon", "toast"],
    "curs": "primer"
}

# Creem un estil 'fake' per testar si no el tenim al CSV
estil_test = "halal_test"
base_estils[estil_test] = {
    "ingredients": ["egg", "toast", "beef", "chicken", "turkey"] # Llista blanca (el bacon no hi és)
}

resultat_2 = adaptar_plat_a_estil(plat_bacon, estil_test, base_estils, base_ingredients)
imprimir_resultat("TEST 2: Adaptació Estil (No Porc)", plat_bacon, resultat_2)


# --- CAS 3: FALLADA CONTROLADA (Sense substitut) ---
# Objectiu: Veure què passa si no hi ha substitut a la categoria.
plat_impossible = {
    "nom": "Plat Estrany",
    "ingredients": ["chocolate"], # Categoria: dolços/altres
    "curs": "postres"
}
# Prohibim xocolata i suposem que no hi ha res més a la seva categoria al CSV
prohibits_xoco = {"chocolate"} 

resultat_3 = substituir_ingredients_prohibits(plat_impossible, prohibits_xoco, base_ingredients)
imprimir_resultat("TEST 3: Sense Substitut (Ha d'avisar)", plat_impossible, resultat_3)

print("\n---------------------------------------")
print("🏁 TESTS FINALITZATS")