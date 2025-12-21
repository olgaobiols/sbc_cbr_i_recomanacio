import json
import os
from typing import Dict, List, Optional, Tuple, Any

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


class GestorRevise:
    """
    Controlador de la fase REVISE amb doble memòria (personal + global).
    """
    def __init__(self) -> None:
        self.user_profiles = self._load_user_profiles()
        self.learned_rules = self._load_learned_rules()

    def _load_user_profiles(self) -> Dict[str, Dict[str, List[str]]]:
        if not os.path.exists(PATH_USER_PROFILES):
            return {}
        try:
            with open(PATH_USER_PROFILES, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _load_learned_rules(self) -> Dict[str, Dict[str, Any]]:
        if not os.path.exists(PATH_LEARNED_RULES):
            return {
                "counters": {"ingredients": {}, "pairs": {}},
                "global_rules": {"ingredients": [], "pairs": []},
            }
        try:
            with open(PATH_LEARNED_RULES, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception:
            data = {}
        data.setdefault("counters", {})
        data["counters"].setdefault("ingredients", {})
        data["counters"].setdefault("pairs", {})
        data.setdefault("global_rules", {})
        data["global_rules"].setdefault("ingredients", [])
        data["global_rules"].setdefault("pairs", [])
        return data

    def _save_user_profiles(self) -> None:
        _safe_write_json(PATH_USER_PROFILES, self.user_profiles)

    def _save_learned_rules(self) -> None:
        _safe_write_json(PATH_LEARNED_RULES, self.learned_rules)

    def _ensure_profile(self, user_id: str) -> Dict[str, List[str]]:
        uid = str(user_id)
        if uid not in self.user_profiles:
            self.user_profiles[uid] = {
                "rejected_ingredients": [],
                "rejected_pairs": [],
            }
        return self.user_profiles[uid]

    def _normalize_ingredient(self, item: str) -> str:
        return str(item).strip().lower()

    def _normalize_pair(self, item: str) -> Optional[str]:
        raw = str(item).strip().lower()
        if "+" in raw:
            parts = [p.strip() for p in raw.split("+") if p.strip()]
        elif "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
        else:
            return None
        if len(parts) != 2:
            return None
        a, b = sorted(parts)
        return f"{a}|{b}"

    def _increment_global_counter(self, category: str, key: str) -> None:
        counters = self.learned_rules["counters"][category]
        counters[key] = counters.get(key, 0) + 1
        if counters[key] > GLOBAL_PROMOTION_THRESHOLD:
            global_list = self.learned_rules["global_rules"][category]
            if key not in global_list:
                global_list.append(key)
        self._save_learned_rules()

    def process_rejection(self, user_id: str, item: str, type: str) -> Optional[str]:
        perfil = self._ensure_profile(user_id)
        if type == "ingredient":
            ing = self._normalize_ingredient(item)
            if not ing:
                return None
            if ing not in perfil["rejected_ingredients"]:
                perfil["rejected_ingredients"].append(ing)
                self._save_user_profiles()
            self._increment_global_counter("ingredients", ing)
            return ing

        if type == "pair":
            pair_key = self._normalize_pair(item)
            if not pair_key:
                return None
            if pair_key not in perfil["rejected_pairs"]:
                perfil["rejected_pairs"].append(pair_key)
                self._save_user_profiles()
            self._increment_global_counter("pairs", pair_key)
            return pair_key

        return None

    def input_nota(self, prompt: str) -> int:
        while True:
            try:
                val = int(input(prompt))
                if 1 <= val <= 5:
                    return val
            except Exception:
                pass
            print("  Si us plau, introdueix un numero de 1 a 5.")

    def input_nota_opcional(self, prompt: str) -> Optional[int]:
        while True:
            raw = input(prompt).strip()
            if raw == "":
                return None
            try:
                val = int(raw)
                if 1 <= val <= 5:
                    return val
            except Exception:
                pass
            print("  Introdueix un numero de 1 a 5 o prem Enter per saltar.")

    def collect_feedback(self, case: Dict, user_id: str) -> Dict[str, Any]:
        print("\n🧐 --- FASE REVISE: AVALUACIO ---")
        n1 = self.input_nota("Puntua el menu globalment (1-5): ")

        print("Pots detallar una mica mes? (Prem Enter per saltar)")
        n2_taste = self.input_nota_opcional("  Nota Gust (1-5): ")
        n2_originality = self.input_nota_opcional("  Nota Originalitat (1-5): ")

        rejected_ingredients: List[str] = []
        rejected_pairs: List[str] = []

        print("\nHi ha algun ingredient o combinacio que vulguis vetar?")
        print("Escriu 'NO ingredient' (ex: 'NO api') o 'NO A+B' (ex: 'NO maduixa+all').")
        print("Escriu 'FI' per acabar.")

        while True:
            cmd = input("> ").strip()
            if cmd == "" or cmd.upper() == "FI":
                break
            if cmd.upper().startswith("NO "):
                target = cmd[3:].strip()
            else:
                target = cmd

            if "+" in target or "|" in target:
                normalized = self.process_rejection(user_id, target, "pair")
                if normalized:
                    rejected_pairs.append(normalized)
                else:
                    print("  Format de parella invalid. Usa 'A+B'.")
            else:
                normalized = self.process_rejection(user_id, target, "ingredient")
                if normalized:
                    rejected_ingredients.append(normalized)
                else:
                    print("  Ingredient invalid.")

        return {
            "puntuacio_global": n1,
            "aspectes": {"gust": n2_taste, "originalitat": n2_originality},
            "ingredients_rebutjats": rejected_ingredients,
            "parelles_rebutjades": rejected_pairs,
        }

    def evaluate_result(
        self,
        puntuacio_global: int,
        n2_taste: Optional[int],
        n2_originality: Optional[int],
        rejected_ingredients: List[str],
        rejected_pairs: List[str],
    ) -> str:
        if rejected_ingredients or rejected_pairs or puntuacio_global <= 2:
            return "CRITICAL_FAILURE"
        if puntuacio_global == 3:
            return "SOFT_FAILURE"

        low_aspect = False
        if n2_taste is not None and n2_taste <= 2:
            low_aspect = True
        if n2_originality is not None and n2_originality <= 2:
            low_aspect = True
        if low_aspect:
            return "SOFT_FAILURE"

        if puntuacio_global >= 4:
            return "SUCCESS"

        return "SOFT_FAILURE"

    def avaluar_proposta(self, cas_proposat: Dict, user_id: str = "guest") -> Dict[str, Any]:
        feedback = self.collect_feedback(cas_proposat, str(user_id))
        status = self.evaluate_result(
            feedback["puntuacio_global"],
            feedback["aspectes"]["gust"],
            feedback["aspectes"]["originalitat"],
            feedback["ingredients_rebutjats"],
            feedback["parelles_rebutjades"],
        )
        feedback["tipus_resultat"] = status
        return feedback
