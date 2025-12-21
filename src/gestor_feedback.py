import json
import os
from typing import Dict, Any

from Revise import GestorRevise as CoreGestorRevise

# Configuracio de fitxers
PATH_USER_PROFILES = "data/user_profiles.json"
PATH_LEARNED_RULES = "data/learned_rules.json"

# Llindar: si el comptador supera aquest valor, es promociona a regla global
GLOBAL_PROMOTION_THRESHOLD = 3


def _safe_write_json(path: str, data: Dict) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)


class MemoriaPersonal:
    def __init__(self) -> None:
        self.path = PATH_USER_PROFILES
        self.data = self._load()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {}
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _save(self) -> None:
        _safe_write_json(self.path, self.data)

    def _ensure_profile(self, user_id: str) -> Dict[str, Any]:
        uid = str(user_id)
        if uid not in self.data:
            self.data[uid] = {
                "rejected_ingredients": [],
                "rejected_pairs": [],
            }
        return self.data[uid]

    def registrar_rebuig_ingredient(self, user_id: str, ingredient: str) -> None:
        perfil = self._ensure_profile(user_id)
        ing = str(ingredient).strip().lower()
        if ing and ing not in perfil["rejected_ingredients"]:
            perfil["rejected_ingredients"].append(ing)
            self._save()

    def registrar_rebuig_parella(self, user_id: str, ing_a: str, ing_b: str) -> None:
        perfil = self._ensure_profile(user_id)
        a = str(ing_a).strip().lower()
        b = str(ing_b).strip().lower()
        if not a or not b:
            return
        key = "|".join(sorted([a, b]))
        if key not in perfil["rejected_pairs"]:
            perfil["rejected_pairs"].append(key)
            self._save()


class MemoriaGlobal:
    def __init__(self) -> None:
        self.path = PATH_LEARNED_RULES
        self.data = self._load()
        self._ensure_structure()

    def _load(self) -> Dict[str, Any]:
        if not os.path.exists(self.path):
            return {
                "counters": {"ingredients": {}, "pairs": {}},
                "global_rules": {"ingredients": [], "pairs": []},
            }
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _ensure_structure(self) -> None:
        self.data.setdefault("counters", {})
        self.data["counters"].setdefault("ingredients", {})
        self.data["counters"].setdefault("pairs", {})
        self.data.setdefault("global_rules", {})
        self.data["global_rules"].setdefault("ingredients", [])
        self.data["global_rules"].setdefault("pairs", [])

    def _save(self) -> None:
        _safe_write_json(self.path, self.data)

    def _promote_if_needed(self, category: str, key: str, count: int) -> None:
        if count > GLOBAL_PROMOTION_THRESHOLD:
            rules = self.data["global_rules"][category]
            if key not in rules:
                rules.append(key)
        self._save()

    def acumular_evidencia_ingredient(self, ingredient: str) -> None:
        ing = str(ingredient).strip().lower()
        if not ing:
            return
        counts = self.data["counters"]["ingredients"]
        counts[ing] = counts.get(ing, 0) + 1
        self._promote_if_needed("ingredients", ing, counts[ing])

    def acumular_evidencia_parella(self, ing_a: str, ing_b: str) -> None:
        a = str(ing_a).strip().lower()
        b = str(ing_b).strip().lower()
        if not a or not b:
            return
        key = "|".join(sorted([a, b]))
        counts = self.data["counters"]["pairs"]
        counts[key] = counts.get(key, 0) + 1
        self._promote_if_needed("pairs", key, counts[key])


class GestorRevise(CoreGestorRevise):
    """
    Wrapper per centralitzar la lògica a Revise.py i mantenir compatibilitat.
    """
    def __init__(self) -> None:
        super().__init__(MemoriaPersonal(), MemoriaGlobal())
