import json
from estructura_cas import DescripcioProblema
from retriever import Retriever
from antic.operadors_transformacio import *
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
    
    
# Carreguem la base d'estils culinaris
base_estils = {}
with open("estils.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # indexem per nom_estil per poder fer base_estils['cuina_molecular']
        base_estils[row["nom_estil"]] = row

# Carreguem la base de tècniques
base_tecnniques = {}
with open("tecniques.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        # indexem per nom_tecnica
        base_tecnniques[row["nom_tecnica"]] = row



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
    print("\n\n\nPASSANT A L'ADAPTACIÓ...")
    if resultats_1:
        cas_seleccionat = resultats_1[0]['cas']  # cas amb més similitud
        plat1 = cas_seleccionat['solucio']['primer_plat']
        plat2 = cas_seleccionat['solucio']['segon_plat']
        postres = cas_seleccionat['solucio']['postres']
        
        # 1) Substituïm ingredients per estil 'japonesa' (el que ve de la petició)
        plat1_modificat = substituir_ingredient(
            plat1, peticio_1.estil_culinari, base_ingredients, base_cuina)
        plat2_modificat = substituir_ingredient(
            plat2, peticio_1.estil_culinari, base_ingredients, base_cuina)
        postres_modificat = substituir_ingredient(
            postres, peticio_1.estil_culinari, base_ingredients, base_cuina)

        print(f"\nMenú adaptat d'INGREDIENTS (estil llenguatge: {peticio_1.estil_culinari}):")
        print(f"  Primer plat: {plat1_modificat['nom']} amb ingredients {plat1_modificat['ingredients']}")
        print(f"  Segon plat:  {plat2_modificat['nom']} amb ingredients {plat2_modificat['ingredients']}")
        print(f"  Postres:     {postres_modificat['nom']} amb ingredients {postres_modificat['ingredients']}")

# 2) Ara fem una prova d'adaptació de TÈCNICA a un estil 'conceptual' (p.ex. cuina_molecular)
        estil_objectiu = "cuina_molecular" # de moment ho posem fix per provar

        print(f"\n\n### ADAPTACIÓ DE TÈCNIQUES AL NOU ESTIL: '{estil_objectiu}' ###")
        
        # --- Plat 1 ---
        # Recollim la llista de tuples (nom_tecnica, raó)
        tecniques_1_amb_rao = triar_tecniques_aplicables(
            plat1_modificat, estil_objectiu, base_estils, base_tecnniques, base_ingredients)
        # Extraiem només els noms per a la funció aplica_tecniques
        tecniques_1_noms = [t[0] for t in tecniques_1_amb_rao] 
        plat1_tecnica, log_1 = aplica_tecniques_al_plat(plat1_modificat, tecniques_1_noms)
        justificacio_1 = justifica_tecniques_aplicades(plat1_tecnica, tecniques_1_amb_rao, base_tecnniques) # NOU

        # --- Plat 2 ---
        # Recollim la llista de tuples (nom_tecnica, raó)
        tecniques_2_amb_rao = triar_tecniques_aplicables(
            plat2_modificat, estil_objectiu, base_estils, base_tecnniques, base_ingredients)
        tecniques_2_noms = [t[0] for t in tecniques_2_amb_rao]
        plat2_tecnica, log_2 = aplica_tecniques_al_plat(plat2_modificat, tecniques_2_noms)
        justificacio_2 = justifica_tecniques_aplicades(plat2_tecnica, tecniques_2_amb_rao, base_tecnniques) # NOU
        
        # --- Postres ---
        # Recollim la llista de tuples (nom_tecnica, raó)
        tecniques_postres_amb_rao = triar_tecniques_aplicables(
            postres_modificat, estil_objectiu, base_estils, base_tecnniques, base_ingredients)
        tecniques_postres_noms = [t[0] for t in tecniques_postres_amb_rao]
        postres_tecnica, log_postres = aplica_tecniques_al_plat(postres_modificat, tecniques_postres_noms)
        justificacio_postres = justifica_tecniques_aplicades(postres_tecnica, tecniques_postres_amb_rao, base_tecnniques) # NOU

        # 3) Mostrem la justificació i el resum final de tècniques
        
        print("\n\n--- JUSTIFICACIÓ DELS CANVIS DETALLADA ---")
        print(f"**Primer Plat: {plat1_tecnica['nom']}**")
        print(justificacio_1)
        print("-" * 20)
        
        print(f"**Segon Plat: {plat2_tecnica['nom']}**")
        print(justificacio_2)
        print("-" * 20)

        print(f"**Postres: {postres_tecnica['nom']}**")
        print(justificacio_postres)
        print("-" * 20)

        print("\nMenú amb TÈCNIQUES adaptades:")
        for etiqueta, plat_original, plat_final in [
            ("Primer plat", plat1, plat1_tecnica),
            ("Segon plat", plat2, plat2_tecnica),
            ("Postres", postres, postres_tecnica),
        ]:
            t_old = plat_original.get("tecnica_principal", "cap_definida")
            t_new = plat_final.get("tecnica_principal", "cap_definida")
            t_sec = ", ".join(plat_final.get("tecnniques_secundaries", [])) or "cap"
            print(f"  {etiqueta}: **{plat_original['nom']}**")
            print(f"     - Tècnica original: {t_old}")
            print(f"     - Tècnica nova (Principal): {t_new}")
            print(f"     - Tècniques Secundàries: {t_sec}")


if __name__ == "__main__":
    main()