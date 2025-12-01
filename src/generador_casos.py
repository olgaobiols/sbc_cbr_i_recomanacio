import csv
import json
import ast
from dataclasses import asdict

# Importem les teves classes definides a estructura_cas.py
# Assegura't que el fitxer es diu exactament 'estructura_cas.py'
from estructura_cas import Plat, DescripcioProblema, SolucioMenu, AvaluacioCas, CasMenu

# --- 1. FUNCIONS AUXILIARS ---

def carregar_cataleg_plats(path_cataleg):
    """
    Llegeix base_plats_part_1.csv i retorna un diccionari per cercar ingredients ràpidament.
    """
    cataleg = {}
    try:
        with open(path_cataleg, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                nom_normalitzat = row['nom_plat'].strip().lower()
                
                # Parsejar llista d'ingredients
                try:
                    ingredients = ast.literal_eval(row['ingredients'])
                except:
                    ingredients = []
                
                # Parsejar preu
                try:
                    preu = float(row['preu_cost'])
                except:
                    preu = 0.0

                # Guardem les dades
                cataleg[nom_normalitzat] = {
                    "ingredients": ingredients,
                    "preu": preu,
                    # Aprofitem la formalitat com a tag d'estil inicial si n'hi ha
                    "estil_tags": [row.get('formalitat', 'indiferent')] 
                }
        print(f"✅ Catàleg carregat: {len(cataleg)} plats disponibles.")
        return cataleg
    except FileNotFoundError:
        print(f"❌ Error: No s'ha trobat el fitxer {path_cataleg}")
        return {}

def crear_objecte_plat(nom_plat, curs_assignat, cataleg):
    """
    Crea un objecte Plat buscant els detalls al catàleg.
    """
    nom_clean = nom_plat.strip()
    nom_key = nom_clean.lower()
    
    dades = cataleg.get(nom_key)
    
    if dades:
        return Plat(
            nom=nom_clean,
            ingredients=dades['ingredients'],
            curs=curs_assignat,  # Adaptat al teu nou nom de variable 'curs'
            estil_tags=dades['estil_tags'],
            preu=dades['preu']   # Adaptat al teu nou nom de variable 'preu'
        )
    else:
        print(f"⚠️ Avís: El plat '{nom_plat}' no s'ha trobat al catàleg. Es crea buit.")
        return Plat(
            nom=nom_clean,
            ingredients=[],
            curs=curs_assignat,
            estil_tags=[],
            preu=0.0
        )

# --- 2. MAIN: GENERACIÓ DEL JSON ---

def generar_base_casos():
    fitxer_seed = 'base_menus_part_1.csv'
    fitxer_cataleg = 'base_plats_part_1.csv'
    fitxer_output = 'base_de_casos.json'

    # 1. Carregar coneixement del domini (Part 1)
    cataleg = carregar_cataleg_plats(fitxer_cataleg)

    l_casos = []

    try:
        with open(fitxer_seed, mode='r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for row in reader:
                # --- A. Construir PROBLEMA ---
                restriccions_llista = row['restriccions'].split('|') if row['restriccions'] != 'cap' else []
                
                problema = DescripcioProblema(
                    tipus_esdeveniment=row['tipus_event'],
                    estil_culinari=row['estil_culinari'],
                    n_comensals=int(row['comensals']),
                    temporada=row['temporada'],
                    pressupost_max=float(row['pressupost_max']),
                    restriccions=restriccions_llista,
                    formalitat="inferida" # Valor per defecte ja que no és al CSV seed explícitament
                )

                # --- B. Construir SOLUCIÓ ---
                # Creem els objectes Plat utilitzant les dades del catàleg
                plat_1 = crear_objecte_plat(row['solucio_1r'], 'primer', cataleg)
                plat_2 = crear_objecte_plat(row['solucio_2n'], 'segon', cataleg)
                plat_3 = crear_objecte_plat(row['solucio_postres'], 'postres', cataleg)

                # Gestionem les begudes
                begudes = [
                    row['solucio_beg_1r'],
                    row['solucio_beg_2n'],
                    row['solucio_beg_postres']
                ]
                # Neteja de duplicats si és la mateixa beguda tot l'àpat
                if len(set(begudes)) == 1:
                    begudes = [begudes[0]]

                solucio = SolucioMenu(
                    primer_plat=plat_1,
                    segon_plat=plat_2,
                    postres=plat_3,
                    begudes=begudes,
                    preu_total=float(row['solucio_preu']),
                    descripcio=f"Menú {row['estil_culinari']} generat històricament."
                )

                # --- C. Construir AVALUACIÓ ---
                es_valid = (row['resultat_exit'] == 'True')
                
                avaluacio = AvaluacioCas(
                    derivacio="Cas Seed (Manual)",
                    feedback_textual="Cas inicial provinent del sistema expert Part 1.",
                    validat=es_valid,
                    utilitat=1.0 if es_valid else 0.0 # Assignem utilitat 1.0 si va ser èxit
                )

                # --- D. MUNTAR EL CAS FINAL ---
                nou_cas = CasMenu(
                    id_cas=int(row['id_cas']),
                    problema=problema,
                    solucio=solucio,
                    avaluacio=avaluacio
                )

                # Utilitzem asdict de dataclasses per convertir tota l'estructura a dict recursivament
                l_casos.append(asdict(nou_cas))

        # 3. Guardar a JSON
        with open(fitxer_output, 'w', encoding='utf-8') as json_file:
            json.dump(l_casos, json_file, indent=4, ensure_ascii=False)
        
        print(f"\n🎉 Èxit! S'ha generat '{fitxer_output}' amb {len(l_casos)} casos.")

    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
    except Exception as e:
        print(f"❌ Error inesperat: {e}")

if __name__ == "__main__":
    generar_base_casos()