import os
import re
import unicodedata
from typing import Dict, Optional, Set, Tuple
import pandas as pd

"""
OPERADOR DE MARIDATGE DE BEGUDES (FASE REUSE)

Selecciona la beguda òptima per a cada plat del menú, combinant seguretat i afinitat gastronòmica.
El procés aplica filtres estrictes (ordre del curs, alcohol, al·lèrgens i dietes) i, un cop
garantida la compatibilitat, puntua les begudes segons la seva coherència amb els ingredients
del plat (família, categoria macro i sabors).

L’ingredient principal té pes doble en el càlcul per reflectir la seva importància.
El mòdul retorna tant la beguda recomanada com una justificació detallada del maridatge.
"""


_ALLERGEN_COLUMN_CANDIDATES = ("alergen", "alergens", "allergen", "allergens")
_BEGUDES_ALLERGENS_CACHE: Dict[str, Optional[object]] = {
    "path": None,
    "column": None,
    "by_id": None,
}

def _normalize_text(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.replace("-", " ").replace("_", " ").lower().split())

def _parse_allergens(value: str) -> Set[str]:
    if not value:
        return set()
    parts = re.split(r"[|,;/]+", str(value))
    normalized = {_normalize_text(p) for p in parts if p and _normalize_text(p)}
    return {p for p in normalized if p not in {"none", "cap", "no", "null", "nan"}}

def _detect_allergen_column(columns) -> Optional[str]:
    for col in columns:
        norm = _normalize_text(col).replace(" ", "_")
        if norm in _ALLERGEN_COLUMN_CANDIDATES:
            return col
    return None

def _load_begudes_allergens(path: str = "data/begudes.csv") -> Tuple[Optional[str], Dict[str, Set[str]]]:
    cached_path = _BEGUDES_ALLERGENS_CACHE.get("path")
    cached_col = _BEGUDES_ALLERGENS_CACHE.get("column")
    cached_map = _BEGUDES_ALLERGENS_CACHE.get("by_id")
    if cached_path == path and cached_map is not None:
        return cached_col, cached_map

    if not os.path.exists(path):
        _BEGUDES_ALLERGENS_CACHE.update({"path": path, "column": None, "by_id": {}})
        return None, {}

    df = pd.read_csv(path)
    col = _detect_allergen_column(df.columns)
    by_id: Dict[str, Set[str]] = {}
    if col and "id" in df.columns:
        for _, row in df.iterrows():
            drink_id = row.get("id")
            if pd.isna(drink_id):
                continue
            drink_id = str(drink_id).strip()
            by_id[drink_id] = _parse_allergens(row.get(col))

    _BEGUDES_ALLERGENS_CACHE.update({"path": path, "column": col, "by_id": by_id})
    return col, by_id

def _row_has_prohibited_allergens(
    beguda_row: Dict,
    prohibited_allergens: Set[str],
    allergen_col: Optional[str],
    allergens_by_id: Dict[str, Set[str]],
) -> bool:
    if not prohibited_allergens:
        return False
    drink_id = str(beguda_row.get("id") or "").strip()
    allergens = set()
    mapped = allergens_by_id.get(drink_id)
    if mapped:
        allergens.update(mapped)
    if allergen_col and allergen_col in beguda_row:
        allergens.update(_parse_allergens(beguda_row.get(allergen_col)))
    else:
        for key in _ALLERGEN_COLUMN_CANDIDATES:
            if key in beguda_row:
                allergens.update(_parse_allergens(beguda_row.get(key)))
                break
    return bool(allergens & prohibited_allergens)

def _first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""

def _normalize_key(value):
    return str(value or "").strip().lower()

def _normalize_ingredient_row(ing_row):
    out = dict(ing_row)
    out["nom_ingredient"] = _first_present(out, ("nom_ingredient", "ingredient_name", "name"))
    out["rol_tipic"] = _first_present(out, ("rol_tipic", "typical_role", "role"))
    out["familia"] = _first_present(out, ("familia", "family"))
    out["categoria_macro"] = _first_present(out, ("categoria_macro", "macro_category"))
    out["sabors_base"] = _first_present(out, ("sabors_base", "base_flavors"))
    return out

def get_ingredient_principal(plat, base_ingredients):
    """Retorna l'ingredient del plat amb typical_role = main."""
    ingredient_principal = None
    llista_ingredients = []
    
    for ing in plat.get("ingredients", []):
        ing_key = _normalize_key(ing)
        for ing_row in base_ingredients:
            row_name = _normalize_key(
                _first_present(ing_row, ("nom_ingredient", "ingredient_name", "name"))
            )
            if row_name == ing_key:
                norm_row = _normalize_ingredient_row(ing_row)
                llista_ingredients.append(norm_row)
                if norm_row.get('rol_tipic') == "main":
                    ingredient_principal = norm_row
    
    # Fallback: si no hi ha ingredient principal, escollim el primer ingredient reconegut
    if ingredient_principal is None and llista_ingredients:
        ingredient_principal = llista_ingredients[0]
        
    # Treu l'ingredient principal de la llista d'altres ingredients
    llista_ingredients_filtrada = [
        ing for ing in llista_ingredients
        if ing != ingredient_principal
    ]

    return ingredient_principal, llista_ingredients_filtrada

def passa_filtre_dur(plat, beguda_row, begudes_usades):
    curs = plat.get("curs", "")
    ordre = beguda_row.get("maridatge_ordre", "").strip()
    id = beguda_row.get("id", "").strip()
    
    # Si la beguda ja ha estat usada no passa:
    if id in begudes_usades:
        return False
    
    # Si la beguda és general passa directament
    if beguda_row.get("es_general", "").strip().lower() == "si":
        return True
        
    # ORDRE obligatori
    if curs == "primer":
        return ordre == "ordre-primer"
    elif curs == "segon":
        return ordre == "ordre-segon"
    elif curs == "postres":
        return ordre == "ordre-postres"
    else:
        return False

def passa_restriccions(beguda_row, restriccions, alcohol):
    alcohol_beguda = str(beguda_row.get("alcohol", "")).strip()
    alergens_beguda = set()
    for key in _ALLERGEN_COLUMN_CANDIDATES:
        if key in beguda_row:
            alergens_beguda = _parse_allergens(beguda_row.get(key))
            break
    dietes_beguda = {
        p.strip().lower()
        for p in str(beguda_row.get("dietes", "")).split("|")
        if p and p.strip()
    }
    
     # 1. Comprovació alcohol
    if alcohol.lower() == "no" and alcohol_beguda == "si":
        return False

    # 2. Al·lèrgens: cap restricció d’al·lèrgens ha d’estar present
    for restriccio in restriccions:
        restriccio_norm = _normalize_text(restriccio)
        if restriccio_norm and restriccio_norm in alergens_beguda:
            return False

    # 3. Dietes: si alguna restricció és tipus dieta, ha d’estar present a la beguda
    restriccions_dieta = {'vegan', 'vegetarian', 'kosher_friendly', 'halal_friendly'}
    has_halal = False
    for restriccio in restriccions:
        restriccio_norm = _normalize_text(restriccio)
        if restriccio_norm in {'halal', 'halal friendly', 'halal_friendly'}:
            has_halal = True
            continue
        if restriccio_norm in {'kosher', 'kosher friendly', 'kosher_friendly'}:
            restriccio_norm = 'kosher_friendly'
        if restriccio_norm in restriccions_dieta and restriccio_norm not in dietes_beguda:
            return False
    if has_halal and alcohol_beguda == "si":
        return False

    return True
    
    

def score_beguda_per_plat(beguda_row, ingredient_principal, llista_ingredients):
    total_score = 0
    breakdown = {
        "ingredient_principal": None,
        "ingredients_secundaris": []
    }
    
    # ------------------------------
    # Funció interna per puntuar 1 ingredient
    # ------------------------------
    def score_per_ingredient(ingredient):
        if not ingredient:
            return 0, {}

        score = 0
        detalls = {
            "nom": ingredient.get("nom_catala"),
            "familia": None,
            "categoria_macro": None,
            "sabors_match": [],
            "sabors_conflicte": []
        }

        # --- Famílies ---
        fam_beguda = set(beguda_row["va_be_amb_familia"].split("|"))
        if ingredient["familia"] in fam_beguda:
            score += 2
            detalls["familia"] = ingredient["familia"]

        # --- Macro categories ---
        macro_beguda = set(beguda_row["va_be_amb_categoria_macro"].split("|"))
        if ingredient["categoria_macro"] in macro_beguda:
            score += 2
            detalls["categoria_macro"] = ingredient["categoria_macro"]

        # --- Sabors ---
        sabors_beguda = set(beguda_row["va_be_amb_sabors"].split("|"))
        evita_sabors = set(beguda_row["evita_sabors"].split("|"))
        sabors_ing = set(ingredient["sabors_base"].split("|"))

        match = sabors_ing & sabors_beguda
        conflict = sabors_ing & evita_sabors

        score += len(match)
        score -= len(conflict)

        detalls["sabors_match"] = sorted(match)
        detalls["sabors_conflicte"] = sorted(conflict)

        return score, detalls


    # ---------------------------------------------------------
    # 1) Puntuar ingredients normals
    # ---------------------------------------------------------
    for ing in llista_ingredients:
        sc, det = score_per_ingredient(ing)
        total_score += sc
        breakdown["ingredients_secundaris"].append(det)

    # ---------------------------------------------------------
    # 2) Puntuar ingredient principal (DOBLE)
    # ---------------------------------------------------------
    if ingredient_principal:
        sc, det = score_per_ingredient(ingredient_principal)
        det["pes_principal"] = "x2"
        det["subtotal"] = sc * 2
        total_score += sc * 2
        breakdown["ingredient_principal"] = det

    breakdown["total_score"] = total_score

    return total_score, breakdown

def recomana_beguda_per_plat(
    plat,
    base_begudes,
    base_ingredients,
    restriccions,
    alcohol,
    begudes_usades,
    prohibited_allergens: Optional[Set[str]] = None,
):
    candidates = []
    
    ing_main, llista_ing = get_ingredient_principal(plat, base_ingredients)
    allergen_col, allergens_by_id = _load_begudes_allergens()
    prohibited_norm = {_normalize_text(a) for a in (prohibited_allergens or []) if a}

    for row in base_begudes:
        if not passa_filtre_dur(plat, row, begudes_usades):
            continue
        if _row_has_prohibited_allergens(row, prohibited_norm, allergen_col, allergens_by_id):
            continue
        if not passa_restriccions(row, restriccions, alcohol):
            continue
        candidates.append(row)

    if not candidates:
        return None, None

    millor = None
    millor_score = float("-inf")

    for row in candidates:
        sc, breakdown = score_beguda_per_plat(row, ing_main, llista_ing)
        if sc > millor_score:
            millor = row
            millor_score = sc
            millor_breakdown = breakdown

    
    if millor is not None:
        begudes_usades.add(millor.get("id"))

    return millor, millor_score, millor_breakdown