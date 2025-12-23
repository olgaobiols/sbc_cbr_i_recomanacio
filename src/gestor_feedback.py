import json
import os
from typing import Dict, Any, List, Optional
from Revise import GestorRevise as CoreGestorRevise

"""
GESTOR DE FEEDBACK I APRENENTATGE (Memòria Dual)
------------------------------------------------
Implementació de l'arquitectura de Canal A i B.
Gestiona la persistència de preferències d'usuari (episòdica) i la inferència 
de regles globals (semàntica) basant-se en la recurrència dels rebuigs.
"""

# --- CONFIGURACIÓ DE PERSISTÈNCIA ---
PATH_USER = "data/user_profiles.json"
PATH_RULES = "data/learned_rules.json"

# Llindar de consens (Tau_global): Mínim de rebuigs per considerar una regla de domini
LLINDAR_GLOBAL = 3 


def _json_rw(path: str, data: Optional[Dict] = None) -> Dict:
    """
    Helper unificat per a lectura i escriptura segura de JSON.
    Utilitza escriptura atòmica per evitar la corrupció de dades.
    """
    if data is None:  # MODE LECTURA
        if not os.path.exists(path):
            return {}
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return {}

    # MODE ESCRIPTURA (Atòmica)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)
        os.replace(tmp_path, path)
    except IOError as e:
        print(f"[Error] No s'ha pogut escriure a {path}: {e}")
    return data


class MemoriaPersonal:
    """
    CANAL A: Memòria Episòdica.
    Emmagatzema les preferències i rebuigs específics de cada usuari.
    """
    def __init__(self):
        self.data = _json_rw(PATH_USER)

    def _update_user_list(self, uid: str, list_key: str, val: str):
        """Mètode intern per afegir elements de forma única al perfil de l'usuari."""
        uid_str = str(uid)
        if uid_str not in self.data:
            self.data[uid_str] = {
                "rejected_ingredients": [],
                "rejected_pairs": []
            }
        
        target_list = self.data[uid_str].setdefault(list_key, [])
        if val not in target_list:
            target_list.append(val)
            _json_rw(PATH_USER, self.data)

    def registrar_rebuig_ingredient(self, uid: str, ing: str):
        """Registra un ingredient que l'usuari no vol tornar a veure."""
        if ing:
            self._update_user_list(uid, "rejected_ingredients", ing.strip().lower())

    def registrar_rebuig_parella(self, uid: str, a: str, b: str):
        """Registra una combinació de dos ingredients rebutjada per l'usuari."""
        if a and b:
            key = "|".join(sorted([a.strip().lower(), b.strip().lower()]))
            self._update_user_list(uid, "rejected_pairs", key)


class MemoriaGlobal:
    """
    CANAL B: Memòria Semàntica.
    Gestiona el coneixement compartit i promou rebuigs recurrents a regles del domini.
    """
    def __init__(self):
        self.data = _json_rw(PATH_RULES)
        self._assegurar_estructura()

    def _assegurar_estructura(self):
        """Garanteix que el fitxer de regles tingui el format correcte."""
        for section in ["counters", "global_rules"]:
            if section not in self.data:
                self.data[section] = {"ingredients": {}, "pairs": {}}
            # Converteix llistes antigues a diccionaris si cal per als comptadors
            if section == "global_rules":
                for cat in ["ingredients", "pairs"]:
                    if not isinstance(self.data[section].get(cat), list):
                        self.data[section][cat] = []

    def _processar_evidencia(self, category: str, key: str):
        """
        Incrementa el comptador d'evidència i avalua si s'ha de promoure 
        a regla global segons el llindar $\tau_{global}$.
        """
        # 1. Incrementar comptador
        counters = self.data["counters"][category]
        counters[key] = counters.get(key, 0) + 1
        
        # 2. Avaluar promoció
        if counters[key] >= LLINDAR_GLOBAL:
            rules = self.data["global_rules"][category]
            if key not in rules:
                rules.append(key)
                self._notificar_promocio(category, key, counters[key])
        
        _json_rw(PATH_RULES, self.data)

    def _notificar_promocio(self, category: str, key: str, count: int):
        """Log visual quan una preferència passa a ser coneixement del sistema."""
        pretty_key = key.replace("|", " + ") if category == "pairs" else key
        label = "Parella vetada" if category == "pairs" else "Ingredient vetat"
        print(f"[Memòria Global] {label} promogut a regla global: {pretty_key} (Evidència: {count})")

    def acumular_evidencia_ingredient(self, ing: str):
        if ing:
            self._processar_evidencia("ingredients", ing.strip().lower())

    def acumular_evidencia_parella(self, a: str, b: str):
        if a and b:
            key = "|".join(sorted([a.strip().lower(), b.strip().lower()]))
            self._processar_evidencia("pairs", key)


class GestorRevise(CoreGestorRevise):
    """
    Injecció de dependències: Connecta el controlador de la fase REVISE
    amb els sistemes de memòria persistents.
    """
    def __init__(self):
        # Injectem les instàncies de memòria personal i global al Core
        super().__init__(
            mem_personal=MemoriaPersonal(), 
            mem_global=MemoriaGlobal()
        )