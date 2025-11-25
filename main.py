import json
from estructura_cas import DescripcioProblema
from retriever import Retriever
from operadors_transformacio import *
import pandas as pd
import ast
import re
import csv


# Carreguem la base d'ingredients
base_ingredients = []

with open("ingredients.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)  # converteix cada fila en un diccionari
    for row in reader:
        base_ingredients.append(row)
    
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
    
    print("\n\nEXTREIENT INGREDIENTS I CURS DEL MENÚ...")
    if resultats_1:
        cas_seleccionat = resultats_1[0]['cas']  # Agafem el cas amb més similitud
        plats_cas = [
            cas_seleccionat['solucio']['primer_plat'],
            cas_seleccionat['solucio']['segon_plat'],
            cas_seleccionat['solucio']['postres']
        ]
        
        ingredients_i_curs_per_plat = {}
        for plat in plats_cas:
            nom_plat = plat['nom']
            curs_plat = plat['curs']
            ingredients = plat['ingredients']
            
            ingredients_i_curs_per_plat[nom_plat] = {
                'ingredients': ingredients,
                'curs': curs_plat
            }

        print("Ingredients i curs de cada plat del menú:")
        for nom, info in ingredients_i_curs_per_plat.items():
            print(f"  {nom} ({info['curs']}): {info['ingredients']}")
        
    
    print("\n\n\nPASSANT A L'ADAPTACIÓ...")
    # AQUÍ LA CRIDA A L'OPERADOR
    if resultats_1:
        cas_seleccionat = resultats_1[0]['cas'] # Agafem el cas amb més similitud
        plat1 = cas_seleccionat['solucio']['primer_plat']
        plat2 = cas_seleccionat['solucio']['segon_plat']
        postres = cas_seleccionat['solucio']['postres']
        
        # Substituïm ingredients per estil i temporada
        plat1_modificat = substituir_ingredient(
            plat1, peticio_1.estil_culinari, base_ingredients, base_cuina)
        plat2_modificat = substituir_ingredient(
            plat2, peticio_1.estil_culinari, base_ingredients, base_cuina)
        postres_modificat = substituir_ingredient(
            postres, peticio_1.estil_culinari, base_ingredients, base_cuina)


        print(f"\nMenú adaptat:")
        print(f"  Primer plat: {plat1_modificat['nom']} amb ingredients {plat1_modificat['ingredients']}")
        print(f"  Segon plat: {plat2_modificat['nom']} amb ingredients {plat2_modificat['ingredients']}")
        print(f"  Postres: {postres_modificat['nom']} amb ingredients {postres_modificat['ingredients']}")


if __name__ == "__main__":
    main()