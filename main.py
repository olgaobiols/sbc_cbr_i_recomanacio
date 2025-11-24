import json
from estructura_cas import DescripcioProblema
from retriever import Retriever
from operadors_transformacio import *
import pandas as pd
import ast
import re

# Carreguem la base de plats amb totes les columnes com a string
base_plats = pd.read_csv("base_plats_casos.csv", dtype=str)

def parse_ingredients(s):
    if pd.isna(s):
        return []
    # Substituïm cometes tipogràfiques per cometes simples
    s = s.replace("‘", "'").replace("’", "'").replace("“", '"').replace("”", '"')
    # Eliminem espais sobrants a dins de la llista
    s = re.sub(r"\s*,\s*", ",", s.strip())
    try:
        return ast.literal_eval(s)
    except Exception as e:
        print(f"Error parsejant ingredients: {s}\n{e}")
        return []

# Convertim la columna 'ingredients' de string a llista
base_plats['ingredients'] = base_plats['ingredients'].apply(parse_ingredients)

# Convertim també apte_esdeveniment i disponibilitat
for col in ['apte_esdeveniment', 'disponibilitat']:
    base_plats[col] = base_plats[col].apply(parse_ingredients)


# Carreguem la base d'ingredients
with open("base_ingredients.json", "r", encoding="utf-8") as f:
    base_ingredients = json.load(f)
    
# Carreguem la base de tipus de cuina
with open("tipus_cuina.json", "r", encoding="utf-8") as f:
    base_cuina = json.load(f)  
    
    


def imprimir_resultat(candidats):
    print(f"\n--- {len(candidats)} CASOS TROBATS ---")
    for i, c in enumerate(candidats):
        cas = c['cas']
        score = c['score_final']
        detall = c['detall']
        
        print(f"\n#{i+1} [Similitud: {score:.1%}] - ID Cas: {cas['id_cas']}")
        print(f"   Estil Original: {cas['problema']['estil_culinari']} ({cas['problema']['tipus_esdeveniment']})")
        print(f"   Preu: {cas['solucio']['preu_total']}€ (vs. Comensals: {cas['problema']['n_comensals']})")
        print(f"   Menú: {cas['solucio']['primer_plat']['nom']} + {cas['solucio']['segon_plat']['nom']} + {cas['solucio']['postres']['nom']}")
        print(f"   Detall: Semàntica {detall['sim_semantica']} | Numèrica {detall['sim_numerica']}")

def main():
    # 1. Inicialitzem el motor de recuperació
    # Assegura't que existeix 'base_de_casos.json' (executa generador_base_casos.py abans)
    retriever = Retriever("base_de_casos.json")

    # --- ESCENARI DE PROVA 1: Cuina Asiàtica ---
    print("\n\nESCENARI 1: Usuari demana 'Sopar empresa asiàtic'")
    # Nota: Al CSV tenim 'oriental_fusio' i 'japones', però l'usuari escriu 'asiàtic'.
    # El model de llenguatge haurà de fer la connexió.
    peticio_1 = DescripcioProblema(
        tipus_esdeveniment="sopar empresa",
        estil_culinari="japonesa",
        n_comensals=50,
        temporada="primavera",
        pressupost_max=40.0,
        restriccions=["cap"],
        formalitat="informal"
    )
    
    resultats_1 = retriever.recuperar_casos_similars(peticio_1)
    imprimir_resultat(resultats_1)

    # --- ESCENARI DE PROVA 2: Boda Tradicional ---
    print("\n\nESCENARI 2: Usuari demana 'Boda clàssica'")
    # Provem si troba els casos 'mediterrani_fresc' o 'tradicional_espanyol'
    peticio_2 = DescripcioProblema(
        tipus_esdeveniment="boda",
        estil_culinari="clàssic tradicional",
        n_comensals=90,
        temporada="estiu",
        pressupost_max=100.0,
        restriccions=["cap"],
        formalitat="formal"
    )

    resultats_2 = retriever.recuperar_casos_similars(peticio_2)
    imprimir_resultat(resultats_2)
    
    print("\n\nEXTREIENT INGREDIENTS DEL MENÚ...")
    if resultats_1:
        cas_seleccionat = resultats_1[0]['cas']  # Agafem el cas amb més similitud
        plats_cas = [
            cas_seleccionat['solucio']['primer_plat']['nom'],
            cas_seleccionat['solucio']['segon_plat']['nom'],
            cas_seleccionat['solucio']['postres']['nom']
        ]
        
        # Extraiem ingredients de cada plat
        ingredients_per_plat = {}
        for plat in plats_cas:
            fila = base_plats[base_plats['nom_plat'] == plat]
            if not fila.empty:
                ingredients_per_plat[plat] = fila.iloc[0]['ingredients']
            else:
                ingredients_per_plat[plat] = []  # Si no troba el plat
    
        print("Ingredients originals del menú:")
        for plat, ingredients in ingredients_per_plat.items():
            print(f"  {plat}: {ingredients}")
        
    
    print("\n\n\nPASSANT A L'ADAPTACIÓ...")
    # AQUÍ LA CRIDA A L'OPERADOR
    if resultats_1:
        cas_seleccionat = resultats_1[0]['cas'] # Agafem el cas amb més similitud
        plat1 = cas_seleccionat['solucio']['primer_plat']
        plat2 = cas_seleccionat['solucio']['segon_plat']
        
        # Substituïm ingredients per estil i temporada
        plat1_modificat = substituir_ingredient(
            plat1, peticio_1.estil_culinari, base_ingredients, base_cuina, peticio_1.temporada)
        plat2_modificat = substituir_ingredient(
            plat2, peticio_1.estil_culinari, base_ingredients, base_cuina, peticio_1.temporada
        )

        print(f"\nMenú adaptat:")
        print(f"  Primer plat: {plat1_modificat['nom']} amb ingredients {plat1_modificat['ingredients']}")
        print(f"  Segon plat: {plat2_modificat['nom']} amb ingredients {plat2_modificat['ingredients']}")

if __name__ == "__main__":
    main()