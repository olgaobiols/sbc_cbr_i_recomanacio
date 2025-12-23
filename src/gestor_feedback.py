import json
import os
from typing import Dict, Any
from Revise import GestorRevise as CoreGestorRevise

"""
GESTOR DE FEEDBACK I APRENENTATGE (Memòria Dual)
------------------------------------------------
Implementació de l'arquitectura de Canal A i B.
Gestiona la persistència de preferències d'usuari i la inferència de regles globals
basant-se en la recurrència dels rebuigs (aprenentatge semàntic).
"""

PATH_USER = "data/user_profiles.json"
PATH_RULES = "data/learned_rules.json"
LLINDAR_GLOBAL = 3  # Tau_global (eq 8): Consens necessari per promoure un rebuig a regla de domini

def _json_rw(path: str, data: Dict = None) -> Dict:
    """Helper unificat per lectura/escriptura segura de JSON."""
    if data is None: # Mode Lectura
        if not os.path.exists(path): return {}
        try:
            with open(path, "r", encoding="utf-8") as f: return json.load(f)
        except: return {}
    
    # Mode Escriptura (Atomic)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f: 
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp, path)
    return data

class MemoriaPersonal:
    """Canal A: Memòria Episòdica (Preferències personals de l'usuari)."""
    def __init__(self):
        self.data = _json_rw(PATH_USER)

    def _add(self, uid: str, list_key: str, val: str):
        if uid not in self.data or not isinstance(self.data.get(uid), dict):
            self.data[uid] = {}
        self.data[uid].setdefault("rejected_ingredients", [])
        self.data[uid].setdefault("rejected_pairs", [])
        self.data[uid].setdefault(list_key, [])
        
        target_list = self.data[uid][list_key]
        if val not in target_list:
            target_list.append(val)
            _json_rw(PATH_USER, self.data)

    def registrar_rebuig_ingredient(self, uid: str, ing: str):
        if ing: self._add(str(uid), "rejected_ingredients", ing.strip().lower())

    def registrar_rebuig_parella(self, uid: str, a: str, b: str):
        if a and b: 
            key = "|".join(sorted([a.strip().lower(), b.strip().lower()]))
            self._add(str(uid), "rejected_pairs", key)

class MemoriaGlobal:
    """Canal B: Memòria Semàntica (Regles del Domini i Comptadors)."""
    def __init__(self):
        self.data = _json_rw(PATH_RULES)
        # Inicialització d'estructura mínima
        for k in ["counters", "global_rules"]:
            if k not in self.data: 
                self.data[k] = {"ingredients": {} if k=="counters" else [], "pairs": {} if k=="counters" else []}

    def _process(self, category: str, key: str):
        cnt = self.data["counters"][category]
        cnt[key] = cnt.get(key, 0) + 1
        
        # Promoció a regla global si supera el llindar de consens (Tau_global)
        if cnt[key] >= LLINDAR_GLOBAL:
            rules = self.data["global_rules"][category]
            if key not in rules:
                rules.append(key)
                if category == "pairs":
                    pretty = key.replace("|", " + ")
                    print(
                        "[Memòria Global] Parella vetada promoguda a regla global: "
                        f"{pretty} (evidència: {cnt[key]})"
                    )
                else:
                    print(
                        "[Memòria Global] Ingredient vetat promogut a regla global: "
                        f"{key} (evidència: {cnt[key]})"
                    )
        
        _json_rw(PATH_RULES, self.data)

    def acumular_evidencia_ingredient(self, ing: str):
        if ing: self._process("ingredients", ing.strip().lower())

    def acumular_evidencia_parella(self, a: str, b: str):
        if a and b: 
            self._process("pairs", "|".join(sorted([a.strip().lower(), b.strip().lower()])))

class GestorRevise(CoreGestorRevise):
    """Wrapper que injecta les memòries persistents al controlador de la fase Revise."""
    def __init__(self):
        super().__init__(MemoriaPersonal(), MemoriaGlobal())
