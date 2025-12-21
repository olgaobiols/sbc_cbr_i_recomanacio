import json
import os
from typing import Dict, List, Set, Any, Tuple
from datetime import datetime

# Configuració de fitxers
PATH_USER_PROFILES = "data/user_profiles.json"
PATH_LEARNED_RULES = "data/learned_rules.json"

# Llindar: Quants usuaris diferents s'han de queixar perquè sigui una regla global?
LLINDAR_GLOBAL_CONFIDENCE = 3 

class MemoriaPersonal:
    """
    CANAL A: Memòria Episòdica/Personal.
    Gestiona les preferències específiques de cada usuari (persistent).
    """
    def __init__(self):
        self.path = PATH_USER_PROFILES
        self.dades = self._carregar()

    def _carregar(self) -> Dict:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _guardar(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.dades, f, indent=4, ensure_ascii=False)

    def get_perfil(self, user_id: str) -> Dict:
        if user_id not in self.dades:
            self.dades[user_id] = {
                "ingredients_rebutjats": [], # Llista negra personal
                "parelles_rebutjades": [],   # Incompatibilitats personals
                "history_sessions": []
            }
        return self.dades[user_id]

    def registrar_rebuig_ingredient(self, user_id: str, ingredient: str):
        perfil = self.get_perfil(user_id)
        if ingredient not in perfil["ingredients_rebutjats"]:
            perfil["ingredients_rebutjats"].append(ingredient)
            self._guardar()
            print(f"[Canal A] Usuari '{user_id}': Afegit '{ingredient}' a la llista negra personal.")

    def registrar_rebuig_parella(self, user_id: str, ing_a: str, ing_b: str):
        perfil = self.get_perfil(user_id)
        parella = sorted([ing_a, ing_b]) # Guardem ordenat per evitar duplicats A-B vs B-A
        str_parella = f"{parella[0]}|{parella[1]}"
        
        if str_parella not in perfil["parelles_rebutjades"]:
            perfil["parelles_rebutjades"].append(str_parella)
            self._guardar()
            print(f"[Canal A] Usuari '{user_id}': Afegida incompatibilitat personal '{str_parella}'.")

class MemoriaGlobal:
    """
    CANAL B: Memòria Semàntica/Global.
    Acumula evidència de múltiples usuaris per crear regles de domini.
    """
    def __init__(self):
        self.path = PATH_LEARNED_RULES
        self.dades = self._carregar()
        self._assegurar_estructura()

    def _carregar(self) -> Dict:
        if not os.path.exists(self.path):
            # Estructura inicial
            return {
                "regles_ingredients": [], # Ingredients prohibits globalment
                "regles_parelles": [],    # Combinacions prohibides globalment
                "comptadors": {           # Evidència acumulada
                    "ingredients": {}, 
                    "parelles": {}
                }
            }
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                return json.load(f)
        except:
            return {}

    def _assegurar_estructura(self):
        """Garanteix que totes les claus necessàries existeixen, fins i tot si el JSON està incomplet."""
        self.dades.setdefault("regles_ingredients", [])
        self.dades.setdefault("regles_parelles", [])
        comptadors = self.dades.setdefault("comptadors", {})
        comptadors.setdefault("ingredients", {})
        comptadors.setdefault("parelles", {})

    def _guardar(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(self.dades, f, indent=4, ensure_ascii=False)

    def _check_promocio_global(self, tipus: str, clau: str, comptador: int):
        """Si superem el llindar, promovem a regla global."""
        if comptador >= LLINDAR_GLOBAL_CONFIDENCE:
            list_key = "regles_ingredients" if tipus == "ingredients" else "regles_parelles"
            
            if clau not in self.dades[list_key]:
                self.dades[list_key].append(clau)
                print(f" [Canal B] NOVA REGLA GLOBAL APRESA: {clau} és universalment dolent!")
                # Aquí podries netejar el comptador o deixar-lo créixer
            self._guardar()

    def acumular_evidencia_ingredient(self, ingredient: str):
        counts = self.dades["comptadors"]["ingredients"]
        counts[ingredient] = counts.get(ingredient, 0) + 1
        print(f"[Canal B] Evidència acumulada per '{ingredient}': {counts[ingredient]}/{LLINDAR_GLOBAL_CONFIDENCE}")
        self._guardar()
        self._check_promocio_global("ingredients", ingredient, counts[ingredient])

    def acumular_evidencia_parella(self, ing_a: str, ing_b: str):
        counts = self.dades["comptadors"]["parelles"]
        parella = sorted([ing_a, ing_b])
        clau = f"{parella[0]}|{parella[1]}"
        
        counts[clau] = counts.get(clau, 0) + 1
        print(f"[Canal B] Evidència acumulada per combinació '{clau}': {counts[clau]}/{LLINDAR_GLOBAL_CONFIDENCE}")
        self._guardar()
        self._check_promocio_global("parelles", clau, counts[clau])

class GestorRevise:
    """
    Controlador de la fase REVISE.
    Interacciona amb l'usuari i actualitza les memòries.
    """
    def __init__(self):
        self.mem_personal = MemoriaPersonal()
        self.mem_global = MemoriaGlobal()

    def input_nota(self, prompt: str) -> int:
        while True:
            try:
                val = int(input(prompt))
                if 1 <= val <= 5: return val
            except: pass
            print("  Si us plau, introdueix un número de 1 a 5.")

    def avaluar_proposta(self, cas_proposat: Dict, user_id: str = "guest"):
        print("\n🧐 --- FASE REVISE: AVALUACIÓ ---")
        
        # 1. N1: Feedback Global
        nota_global = self.input_nota("Puntua el menú globalment (1-5): ")
        
        resultat = {
            "puntuacio_global": nota_global,
            "ingredients_rebutjats": [],
            "parelles_rebutjades": [],
            "tipus_resultat": "exit" if nota_global >= 4 else "fracas_suau"
        }

        if nota_global == 5:
            print(" Fantàstic! No calen més detalls.")
            return resultat

        # 2. N2: Aspectes (Opcional, només si la nota no és perfecta)
        print("Pots detallar una mica més? (Prem Enter per saltar)")
        nota_gust = input("  Nota Gust (1-5): ")
        nota_orig = input("  Nota Originalitat (1-5): ")
        
        # 3. N3: Feedback Granular (Crític per aprendre)
        print("\nHi ha algun ingredient o combinació que vulguis vetar?")
        print("Escriu 'NO ingredient' (ex: 'NO api') o 'NO A+B' (ex: 'NO maduixa+all').")
        print("Escriu 'FI' per acabar.")
        
        while True:
            cmd = input("> ").strip()
            if cmd.upper() == "FI" or cmd == "": break
            
            if cmd.upper().startswith("NO "):
                target = cmd[3:].strip().lower()
                
                # Cas Parella (A+B)
                if "+" in target:
                    parts = target.split("+")
                    if len(parts) == 2:
                        ing_a, ing_b = parts[0].strip(), parts[1].strip()
                        self.mem_personal.registrar_rebuig_parella(user_id, ing_a, ing_b)
                        self.mem_global.acumular_evidencia_parella(ing_a, ing_b)
                        resultat["parelles_rebutjades"].append(f"{ing_a}|{ing_b}")
                
                # Cas Ingredient Únic
                else:
                    self.mem_personal.registrar_rebuig_ingredient(user_id, target)
                    self.mem_global.acumular_evidencia_ingredient(target)
                    resultat["ingredients_rebutjats"].append(target)
        
        # Classificació final del fracàs
        if resultat["ingredients_rebutjats"] or resultat["parelles_rebutjades"]:
            # Si l'usuari ha vetat coses específiques, considerem que hi ha hagut un problema de contingut
            # Si la nota era molt baixa (1-2), és crític. Si és 3, és suau.
            if nota_global <= 2:
                resultat["tipus_resultat"] = "fracas_critic"
            else:
                resultat["tipus_resultat"] = "fracas_suau" # Acceptable però millorable

        return resultat
