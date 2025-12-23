import random
import unicodedata
from typing import List, Dict, Set, Any, Optional
import numpy as np
from flavorgraph_embeddings import FlavorGraphWrapper

"""
OPERADOR D'ADAPTACIÓ D'INGREDIENTS (FASE REUSE)
Implementa la lògica de transformació híbrida descrita a la secció 2.2 de l'informe.

Aquest mòdul gestiona dues estratègies d'adaptació:
1. Correcció (Seguretat): Substitueix ingredients que violen restriccions (al·lèrgies/dietes)
   utilitzant un filtratge ontològic estricte i selecció per pairing amb la resta d'ingredients del plat.
2. Adaptació d'Estil (Creativitat): Transforma plats existent cap a nous estils
   (Latent Style Adaptation) utilitzant vectors d'embeddings i injecció de soroll.

També gestiona la coherència global (evitar parelles prohibides detectades pel Feedback N3).
"""

FG_WRAPPER = FlavorGraphWrapper()

# ---------------------------------------------------------------------
# FUNCIONS AUXILIARS DE GESTIÓ DE DADES I VECTORS
# ---------------------------------------------------------------------
def _normalize_text(value: str) -> str:
    """Normalitza cadenes de text (ASCII, minúscules) per a comparacions robustes."""
    if not value: return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    return " ".join(text.replace("-", " ").replace("_", " ").lower().split())

def _is_empty_tag(value: str) -> bool:
    if not value:
        return True
    return _normalize_text(value) in {"none", "no", "null", "nan", "n/a", "na"}

def _normalize_category(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value)).encode("ascii", "ignore").decode("ascii")
    text = text.strip().lower().replace("-", "_").replace(" ", "_")
    return "_".join(part for part in text.split("_") if part)

def _normalize_diet_tag(value: str) -> str:
    norm = _normalize_text(value)
    if norm in {"vega", "vegan"}:
        return "vegan"
    if norm in {"vegetaria", "vegetarian"}:
        return "vegetarian"
    if norm in {"halal", "halal friendly"}:
        return "halal_friendly"
    if norm in {"kosher", "kosher friendly"}:
        return "kosher_friendly"
    return norm

def _is_truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return _normalize_text(value) in {"true", "si", "yes", "1"}

def _ingredient_name_blob(info: Dict, fallback: str = "") -> str:
    parts = [
        fallback,
        info.get("ingredient_name"),
        info.get("nom_ingredient"),
        info.get("nom_catala"),
        info.get("name"),
    ]
    return _normalize_text(" ".join(p for p in parts if p))

def _is_meat_or_fish(info: Dict, ingredient_name: str = "") -> bool:
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    fam = _normalize_text(info.get("family") or info.get("familia"))
    blob = _ingredient_name_blob(info, ingredient_name)
    if cat in _MEAT_FISH_CATEGORIES:
        return True
    if any(token in fam for token in _MEAT_FISH_FAMILY_MARKERS):
        return True
    if any(token in blob for token in _MEAT_FISH_KEYWORDS):
        return True
    if cat in _PROTEIN_ANIMAL_CATEGORIES:
        if "egg" in fam or "egg" in blob or "dairy" in fam:
            return False
        return True
    return False

def _is_fungi(info: Dict, ingredient_name: str = "") -> bool:
    fam = _normalize_text(info.get("family") or info.get("familia"))
    blob = _ingredient_name_blob(info, ingredient_name)
    return any(token in fam or token in blob for token in _FUNGI_MARKERS)

def _is_vegetable(info: Dict, ingredient_name: str = "") -> bool:
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    fam = _normalize_text(info.get("family") or info.get("familia"))
    blob = _ingredient_name_blob(info, ingredient_name)
    if cat in _VEGETABLE_CATEGORIES:
        return True
    return "vegetable" in fam or "vegetable" in blob

def _is_baking_ingredient(info: Dict, ingredient_name: str = "") -> bool:
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    blob = _ingredient_name_blob(info, ingredient_name)
    if cat in _BAKING_CATEGORIES:
        return True
    if any(token in blob for token in _BAKING_KEYWORDS):
        return True
    return False

def _is_halal_haram(info: Dict, ingredient_name: str = "") -> bool:
    fam = _normalize_text(info.get("family") or info.get("familia"))
    blob = _ingredient_name_blob(info, ingredient_name)
    if any(token in fam or token in blob for token in _HALAL_HARAM_MARKERS):
        return True
    return False

def _es_candidat_postres_segura(info: Dict, ingredient_name: str = "") -> bool:
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    fam = _normalize_text(info.get("family") or info.get("familia"))
    blob = _ingredient_name_blob(info, ingredient_name)

    if _is_meat_or_fish(info, ingredient_name):
        return False
    if _is_fungi(info, ingredient_name):
        return False
    if _is_vegetable(info, ingredient_name):
        if not any(token in blob for token in _DESSERT_VEG_EXCEPTIONS):
            return False

    if cat in _DESSERT_SAFE_CATEGORIES:
        return True
    if cat == "other" and "dairy_substitute" in fam:
        return True
    if any(token in blob for token in _BAKING_HELPER_KEYWORDS):
        return True
    return False

def _es_substitucio_semanticament_coherent(
    info_orig: Dict,
    info_cand: Dict,
    orig_name: str = "",
    cand_name: str = "",
) -> bool:
    if _is_baking_ingredient(info_orig, orig_name) and _is_meat_or_fish(info_cand, cand_name):
        return False
    if _is_meat_or_fish(info_orig, orig_name) and _is_baking_ingredient(info_cand, cand_name):
        return False
    return True

def _es_no_vega(info: Dict, ingredient_name: str = "") -> bool:
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    fam = _normalize_text(info.get("family") or info.get("familia"))
    tipus = _normalize_text(info.get("type") or info.get("tipus"))
    blob = _ingredient_name_blob(info, ingredient_name)

    if _is_truthy(info.get("es_animal")):
        return True
    if cat in _NON_VEGAN_CATEGORIES:
        return True
    if any(token in fam for token in _NON_VEGAN_FAMILY_MARKERS):
        return True
    if any(token in tipus for token in _NON_VEGAN_TYPE_MARKERS):
        return True
    if any(token in blob for token in _NON_VEGAN_KEYWORDS):
        return True
    return False

def _es_no_vegetaria(info: Dict, ingredient_name: str = "") -> bool:
    tipus = _normalize_text(info.get("type") or info.get("tipus"))
    if any(token in tipus for token in _MEAT_FISH_TYPE_MARKERS):
        return True
    if _is_meat_or_fish(info, ingredient_name):
        return True
    return False

def _normalize_vector(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None: return None
    norm = np.linalg.norm(vec)
    return vec / norm if norm != 0 else None

def _calcular_vector_context(ingredients: List[str], exclude_index: int = -1) -> Optional[np.ndarray]:
    """
    Calcula el vector mitjà de la resta d'ingredients (Context).
    Clau per avaluar el 'Pairing': com de bé encaixa un ingredient amb els seus veïns.
    """
    vectors = [
        FG_WRAPPER.get_vector(ing) for i, ing in enumerate(ingredients)
        if i != exclude_index and FG_WRAPPER.get_vector(ing) is not None
    ]
    return np.mean(vectors, axis=0) if vectors else None

def _vector_promig_plat(ingredients: List[str]) -> Optional[np.ndarray]:
    """Calcula el centroide del plat complet (per comparar amb l'Estil objectiu)."""
    vectors = [v for ing in ingredients if (v := FG_WRAPPER.get_vector(ing)) is not None]
    return _normalize_vector(np.mean(vectors, axis=0)) if vectors else None

def _check_compatibilitat(ingredient_info: Dict, perfil_usuari: Optional[Dict]) -> bool:
    """Verifica restriccions dures (Seguretat): al·lèrgies i dietes explícites."""
    if not ingredient_info: return False
    if not perfil_usuari: return True

    # Validació d'al·lèrgies
    alergies_usuari = {
        _normalize_text(a)
        for a in perfil_usuari.get('alergies', [])
        if a and not _is_empty_tag(a)
    }
    if alergies_usuari:
        alergens_ing = {
            _normalize_text(p)
            for p in str(
                ingredient_info.get('allergens')
                or ingredient_info.get('alergens')
                or ''
            ).split('|')
            if p and not _is_empty_tag(p)
        }
        familia_ing = _normalize_text(ingredient_info.get('family') or ingredient_info.get('familia'))
        if alergies_usuari.intersection(alergens_ing) or (familia_ing and familia_ing in alergies_usuari):
            return False

    # Validació de dieta
    dieta_usuari = _normalize_diet_tag(perfil_usuari.get('dieta'))
    if dieta_usuari:
        if dieta_usuari == "halal_friendly":
            if _is_halal_haram(ingredient_info):
                return False
            return True
        dietes_raw = (
            ingredient_info.get('allowed_diets')
            or ingredient_info.get('dietes_permeses')
            or ingredient_info.get('dietes')
            or ''
        )
        dietes_ing = {
            _normalize_diet_tag(p)
            for p in str(dietes_raw).split('|')
            if p and not _is_empty_tag(p)
        }
        if dieta_usuari == "vegan":
            if _es_no_vega(ingredient_info):
                return False
        elif dieta_usuari == "vegetarian":
            if _es_no_vegetaria(ingredient_info):
                return False
        elif dietes_ing and dieta_usuari not in dietes_ing:
            return False

    return True

def _check_parelles_prohibides(candidat: str, context_ingredients: List[str], parelles_prohibides: Set[str]) -> bool:
    """
    Verifica si afegir 'candidat' genera una combinació prohibida (feedback aprenentatge).
    Consulta el conjunt de regles negatives (Canal A/B).
    """
    if not parelles_prohibides: return False
    cand_norm = _normalize_text(candidat)
    
    for other in context_ingredients:
        other_norm = _normalize_text(other)
        if cand_norm == other_norm: continue 
        
        # Clau ordenada per garantir unicitat de la parella (A|B == B|A)
        pair = sorted([cand_norm, other_norm])
        if f"{pair[0]}|{pair[1]}" in parelles_prohibides:
            return True
    return False

def _build_perfil_context(perfil_base: Optional[Dict], info_prohibit: Dict) -> Dict:
    """Crea un perfil temporal afegint les restriccions de l'ingredient que eliminem (per seguretat)."""
    perfil = perfil_base.copy() if perfil_base else {}
    alergies = {_normalize_text(a) for a in perfil.get('alergies', []) if a}
    
    # Afegim al·lèrgens de l'ingredient prohibit al context de cerca
    for tag in str(info_prohibit.get('allergens') or info_prohibit.get('alergens') or '').split('|'):
        if t := _normalize_text(tag):
            if not _is_empty_tag(t):
                alergies.add(t)
    if fam := _normalize_text(info_prohibit.get('family')): alergies.add(fam)
    
    perfil['alergies'] = alergies
    return perfil

def ingredients_incompatibles(ingredients: List[str], kb: Any, perfil_usuari: Optional[Dict]) -> Set[str]:
    """Identifica quins ingredients del plat violen el perfil de l'usuari."""
    prohibits = set()
    if not perfil_usuari: return prohibits
    
    for ing in ingredients:
        if info := kb.get_info_ingredient(ing):
            if not _check_compatibilitat(info, perfil_usuari):
                prohibits.add(ing)
        else:
            dieta = _normalize_diet_tag(perfil_usuari.get("dieta"))
            ing_norm = _normalize_text(ing)
            if dieta == "vegan":
                if any(token in ing_norm for token in _NON_VEGAN_KEYWORDS):
                    prohibits.add(ing)
            elif dieta == "vegetarian":
                if any(token in ing_norm for token in _MEAT_FISH_KEYWORDS):
                    prohibits.add(ing)
            elif dieta == "halal_friendly":
                if any(token in ing_norm for token in _HALAL_HARAM_MARKERS):
                    prohibits.add(ing)
    return prohibits

def _get_candidats_per_categoria(categoria: str, kb: Any) -> List[str]:
    cat_norm = _normalize_category(categoria)
    candidats = []
    for nom, info in kb.ingredients.items():
        c_macro = _normalize_category(info.get('macro_category') or info.get('categoria_macro'))
        if c_macro == cat_norm:
            candidats.append(info['ingredient_name'])
    return candidats

def _categoria_fallbacks(categoria_norm: str, perfil_usuari: Optional[Dict]) -> List[str]:
    if not perfil_usuari: return []
    dieta = _normalize_text(perfil_usuari.get('dieta'))
    categoria_norm = _normalize_category(categoria_norm)
    fallbacks = []
    if dieta in {"vegan", "vegetarian"}:
        if categoria_norm in _PROTEINA_CATEGORIES:
            fallbacks.extend(["protein_vegetal", "plant_vegetal", "grain"])
        if categoria_norm in _DAIRY_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "fat", "other"])
        if categoria_norm in _EGG_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "grain"])
    return [_normalize_category(c) for c in fallbacks]

def _es_candidat_coherent(info_orig: Dict, info_cand: Dict, cat_orig_norm: str) -> bool:
    cat_orig_norm = _normalize_category(cat_orig_norm)
    r_orig = _normalize_text(info_orig.get('typical_role'))
    r_cand = _normalize_text(info_cand.get('typical_role'))
    if cat_orig_norm in _PROTEINA_CATEGORIES or r_orig in _MAIN_ROLES:
        return r_cand in _MAIN_ROLES
    if r_orig in _SIDE_ROLES and r_cand:
        return r_cand in _SIDE_ROLES or r_cand in _MAIN_ROLES
    return True 

def _ordenar_candidats_per_afinitat(candidats: List[str], kb: Any, info_orig: Dict) -> List[str]:
    pref_fam = _normalize_text(info_orig.get('family'))
    pref_role = _normalize_text(info_orig.get('typical_role'))
    ranking = []
    for cand in candidats:
        info = kb.get_info_ingredient(cand)
        score = 0
        if info:
            if _normalize_text(info.get('family')) == pref_fam: score += 2
            if _normalize_text(info.get('typical_role')) == pref_role: score += 1
        ranking.append((score, cand))
    ranking.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in ranking]

# ---------------------------------------------------------------------
# LÒGICA ONTOLÒGICA (Simbolisme)
# ---------------------------------------------------------------------
_PROTEINA_CATEGORIES = {"protein_animal", "protein_vegetal", "fish_white", "fish_oily", "proteina_animal"}
_DAIRY_CATEGORIES = {"dairy", "dairy_cheese", "dairy_cream", "lacti"}
_GRAIN_CATEGORIES = {"grain", "processed_cereal", "cereal_feculent"}
_EGG_CATEGORIES = {"egg"}
_MAIN_ROLES = {"main", "principal", "main course"}
_SIDE_ROLES = {"side", "base"}
_MEAT_FISH_CATEGORIES = {"fish_white", "fish_oily", "fish", "seafood"}
_PROTEIN_ANIMAL_CATEGORIES = {"protein_animal", "proteina_animal"}
_NON_VEGAN_CATEGORIES = _MEAT_FISH_CATEGORIES | _PROTEIN_ANIMAL_CATEGORIES | _DAIRY_CATEGORIES | _EGG_CATEGORIES
_MEAT_FISH_FAMILY_MARKERS = {
    "meat",
    "poultry",
    "beef",
    "pork",
    "fish",
    "seafood",
    "shellfish",
    "crustacean",
    "mollusc",
}
_NON_VEGAN_FAMILY_MARKERS = _MEAT_FISH_FAMILY_MARKERS | {"dairy", "cheese", "milk", "egg", "honey"}
_MEAT_FISH_TYPE_MARKERS = {"animal_product", "meat", "fish", "seafood"}
_NON_VEGAN_TYPE_MARKERS = _MEAT_FISH_TYPE_MARKERS | {"dairy", "egg", "eggs"}
_MEAT_FISH_KEYWORDS = {
    "chicken",
    "pollastre",
    "beef",
    "vedella",
    "veal",
    "pork",
    "porc",
    "ham",
    "pernil",
    "bacon",
    "fish",
    "peix",
    "seafood",
    "marisc",
    "shellfish",
    "crustacean",
    "mollusc",
    "shrimp",
    "prawn",
    "salmon",
    "tuna",
    "cod",
    "sardine",
}
_NON_VEGAN_KEYWORDS = _MEAT_FISH_KEYWORDS | {
    "milk",
    "llet",
    "cheese",
    "formatge",
    "butter",
    "mantega",
    "cream",
    "nata",
    "yogurt",
    "iogurt",
    "egg",
    "ou",
    "ous",
    "honey",
    "mel",
}
_VEGETABLE_CATEGORIES = {"vegetable", "plant_vegetal"}
_FUNGI_MARKERS = {"mushroom", "fungi", "fungus"}
_BAKING_CATEGORIES = {"grain", "sweet", "sweetener"}
_BAKING_KEYWORDS = {
    "flour",
    "farina",
    "sugar",
    "sucre",
    "baking",
    "yeast",
    "powder",
    "baking_powder",
}
_BAKING_HELPER_KEYWORDS = {"baking", "yeast", "powder"}
_DESSERT_SAFE_CATEGORIES = {
    "fruit",
    "nuts",
    "nut",
    "grain",
    "dairy",
    "sweetener",
    "sweet",
    "spice",
    "fat",
}
_DESSERT_VEG_EXCEPTIONS = {
    "carrot",
    "carrot_puree",
    "carrot_slices",
}
_HALAL_HARAM_MARKERS = {
    "pork",
    "porc",
    "ham",
    "pernil",
    "bacon",
    "lard",
    "prosciutto",
    "pancetta",
    "salami",
    "chorizo",
    "alcohol",
    "wine",
    "beer",
    "rum",
    "gin",
    "vodka",
    "whisky",
    "whiskey",
    "brandy",
    "liqueur",
    "liquor",
}

LATENT_CONDIMENT_SETS = {
    "citric": {
        "starter": [
            "lemon-mustard vinaigrette (mild)",
            "lemon zest + extra virgin olive oil",
            "yogurt-lime sauce",
        ],
        "main": [
            "light lemon-butter sauce",
            "orange-citrus herb glaze",
            "soft citrus mojo",
        ],
        "dessert": [
            "lemon coulis",
            "lime cream",
            "citrus syrup",
        ],
    },
    "fumat": {
        "starter": [
            "smoked oil drizzle",
            "smoked paprika oil",
            "light smoked-vegetable cream",
        ],
        "main": [
            "mild smoky barbecue sauce",
            "smoky soy-honey glaze",
            "dark jus with smoky note",
        ],
        "dessert": [
            "smoked caramel",
            "dark chocolate + smoked salt",
            "vanilla cream with a smoky touch",
        ],
    },
    "italia": {
        "starter": [
            "classic pesto (Genovese-style)",
            "basil tomato sauce",
            "light parmesan cream",
        ],
        "main": [
            "herbed tomato sauce",
            "wine + rosemary reduction",
            "sage-butter sauce",
        ],
        "dessert": [
            "mascarpone cream",
            "coffee sauce",
            "sweet Marsala reduction",
        ],
    },
    "mexica": {
        "starter": [
            "mild pico de gallo",
            "tomatillo salsa verde",
            "sour cream + lime",
        ],
        "main": [
            "roasted red salsa",
            "mild mole sauce",
            "spiced adobo",
        ],
        "dessert": [
            "chocolate + cinnamon",
            "piloncillo caramel",
            "vanilla-cinnamon cream",
        ],
    },
    "picant": {
        "starter": [
            "infused chili oil (controlled heat)",
            "yogurt + mild chili sauce",
            "spicy vinaigrette (mild)",
        ],
        "main": [
            "fermented chili sauce (small dose)",
            "sweet-spicy glaze",
            "spiced cream with chili",
        ],
        "dessert": [
            "chocolate + chili",
            "hot honey",
            "berry coulis with chili",
        ],
    },
    "tropical": {
        "starter": [
            "mango vinaigrette",
            "coconut-lime sauce",
            "tropical fruit cold cream",
        ],
        "main": [
            "pineapple glaze",
            "spiced coconut sauce",
            "tropical chutney",
        ],
        "dessert": [
            "passionfruit coulis",
            "coconut cream",
            "mango-lime sauce",
        ],
    },
    "umami": {
        "starter": [
            "concentrated mushroom broth",
            "mild miso cream",
            "mushroom oil",
        ],
        "main": [
            "reduced soy sauce",
            "vegetable demi-glace",
            "mushroom-wine sauce",
        ],
        "dessert": [
            "salted caramel",
            "intense dark chocolate",
            "coffee cream",
        ],
    },
}
_RECENT_CONDIMENTS = {
    style: {course: [] for course in courses} for style, courses in LATENT_CONDIMENT_SETS.items()
}

def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))

def _map_course_to_condiment_key(plat: Dict[str, Any]) -> Optional[str]:
    curs = _normalize_text(plat.get("curs", ""))
    if "postres" in curs:
        return "dessert"
    if curs in {"segon", "main"}:
        return "main"
    if curs in {"primer", "starter"}:
        return "starter"
    return None

def _weighted_choice(candidates: List[str], recent: List[str], rng: random.Random) -> Optional[str]:
    if not candidates:
        return None
    recent_set = set(recent)
    fresh = [c for c in candidates if c not in recent_set]
    if fresh:
        return rng.choice(fresh)

    weights = []
    for cand in candidates:
        weight = 1.0
        if recent:
            if cand == recent[-1]:
                weight = 0.2
            elif len(recent) > 1 and cand == recent[-2]:
                weight = 0.5
        weights.append(weight)
    total = sum(weights)
    if total <= 0:
        return rng.choice(candidates)
    roll = rng.random() * total
    acc = 0.0
    for cand, weight in zip(candidates, weights):
        acc += weight
        if roll <= acc:
            return cand
    return candidates[-1]

def pick_latent_condiment(
    style: str,
    course: str,
    temperature: float,
    fallback_mode: bool = False,
    random_mode: bool = False,
    rng: Optional[random.Random] = None,
) -> Optional[str]:
    if not style or not course:
        return None
    if style not in LATENT_CONDIMENT_SETS:
        return None
    if course not in LATENT_CONDIMENT_SETS[style]:
        return None

    rng = rng or random
    t = _clamp(float(temperature), 0.1, 0.9)
    base_prob = _clamp((t - 0.2) / 0.7, 0.0, 1.0)
    if fallback_mode:
        prob = max(base_prob, 0.6)
    else:
        prob = base_prob if random_mode else 0.0

    if rng.random() >= prob:
        return None

    candidates = list(LATENT_CONDIMENT_SETS[style][course])
    recent = _RECENT_CONDIMENTS.get(style, {}).get(course, [])
    choice = _weighted_choice(candidates, recent, rng)
    if choice is None:
        return None

    if style in _RECENT_CONDIMENTS and course in _RECENT_CONDIMENTS[style]:
        _RECENT_CONDIMENTS[style][course].append(choice)
        if len(_RECENT_CONDIMENTS[style][course]) > 2:
            _RECENT_CONDIMENTS[style][course] = _RECENT_CONDIMENTS[style][course][-2:]
    return choice

def _condiment_random_mode(temperature: float, rng: random.Random) -> bool:
    t = _clamp(float(temperature), 0.1, 0.9)
    chance = _clamp(0.05 + 0.5 * ((t - 0.1) / 0.8), 0.0, 0.6)
    return rng.random() < chance


# CONSTANTS ONTOLÒGIQUES 
# MILLORRRRRRRRRRRRRRRRRRRRAAAAAAAAAAAAAAAAAAR



def _get_candidats_per_categoria(categoria: str, kb: Any) -> List[str]:
    cat_norm = _normalize_category(categoria)
    candidats = []
    for info in kb.ingredients.values():
        if _normalize_category(info.get('macro_category') or info.get('categoria_macro')) != cat_norm:
            continue
        nom = info.get("ingredient_name") or info.get("nom_ingredient") or info.get("name")
        if nom:
            candidats.append(nom)
    return candidats

def _categoria_fallbacks(categoria_norm: str, perfil_usuari: Optional[Dict]) -> List[str]:
    """Defineix substitucions ontològiques segures quan la categoria original està prohibida."""
    if not perfil_usuari: return []
    dieta = _normalize_text(perfil_usuari.get('dieta'))
    categoria_norm = _normalize_category(categoria_norm)
    fallbacks = []
    
    if dieta in {"vegan", "vegetarian"}:
        if categoria_norm in _PROTEINA_CATEGORIES:
            fallbacks.extend(["protein_vegetal", "plant_vegetal", "grain"])
        elif categoria_norm in _DAIRY_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "fat", "other"])
        elif categoria_norm in _EGG_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "grain"])
            
    return [_normalize_category(c) for c in fallbacks]

def _role_tokens(value: Any) -> Set[str]:
    if not value:
        return set()
    tokens = set()
    for part in str(value).split("|"):
        part_norm = _normalize_text(part)
        if part_norm:
            tokens.add(part_norm)
    return tokens

def _es_candidat_coherent(info_orig: Dict, info_cand: Dict, cat_orig_norm: str) -> bool:
    """Verifica que el substitut mantingui el rol estructural al plat (Coherència)."""
    cat_orig_norm = _normalize_category(cat_orig_norm)
    r_orig_tokens = _role_tokens(info_orig.get('typical_role'))
    r_cand_tokens = _role_tokens(info_cand.get('typical_role'))
    
    if cat_orig_norm in _PROTEINA_CATEGORIES or (r_orig_tokens & _MAIN_ROLES):
        return bool(r_cand_tokens & _MAIN_ROLES)
    if r_orig_tokens & _SIDE_ROLES and r_cand_tokens:
        return bool(r_cand_tokens & _SIDE_ROLES) or bool(r_cand_tokens & _MAIN_ROLES)
    return True 

def _ordenar_candidats_per_afinitat(candidats: List[str], kb: Any, info_orig: Dict) -> List[str]:
    """Ordena candidats ontològics per proximitat taxonòmica (família > rol)."""
    pref_fam = _normalize_text(info_orig.get('family'))
    pref_role = _normalize_text(info_orig.get('typical_role'))
    ranking = []
    
    for cand in candidats:
        info = kb.get_info_ingredient(cand)
        score = 0
        if info:
            if _normalize_text(info.get('family')) == pref_fam: score += 2
            if _normalize_text(info.get('typical_role')) == pref_role: score += 1
        ranking.append((score, cand))
        
    ranking.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in ranking]

def _check_role_compatibility(cat_orig: str, cat_cand: str) -> bool:
    """Validació estricta de compatibilitat entre categories (Llei de ferro)."""
    cat_orig = _normalize_category(cat_orig)
    cat_cand = _normalize_category(cat_cand)
    if not cat_orig or not cat_cand or "unknown" in (cat_orig, cat_cand):
        return False

    proteins = {"meat", "fish", "protein", "protein_animal", "protein_vegetal", "proteina_animal"}
    seasonings = {"herb", "spice", "condiment"}
    structural = {"grain", "pasta", "potato", "processed_cereal", "cereal_feculent", "vegetable"}
    dessert_ok = {"fruit", "sweet", "nut"}

    if cat_orig in proteins: return cat_cand in proteins
    if cat_orig in seasonings: return cat_cand in seasonings
    if cat_orig in structural: return cat_cand in structural
    if cat_orig == "fruit": return cat_cand in dessert_ok

    return cat_cand == cat_orig

_DESSERT_ALLOWED_CATEGORIES = {
    "fruit",
    "sweet",
    "sweetener",
    "dairy",
    "nuts",
    "nut",
}
_DESSERT_EXTRA_CATEGORIES = {
    "fat",
    "spice",
    "seasoning",
    "sauce",
    "other",
    "alcohol",
}
_DESSERT_SAVORY_FLAVORS = {"salty", "umami", "savory", "meaty", "fishy", "smoky"}
_DESSERT_SOFT_FLAVORS = {"sweet", "mild_sweet", "creamy", "mild", "tropical"}
_DESSERT_GOOD_FLAVORS = {
    "sweet",
    "mild_sweet",
    "creamy",
    "mild",
    "fruity",
    "citrus",
    "citrusy",
    "tropical",
    "vanilla",
    "caramel",
    "chocolate",
    "cocoa",
    "berry",
}

def _es_apte_postres(info: Dict, intensitat: float) -> bool:
    """Filtre per evitar ingredients salats/umami en postres."""
    if not info:
        return False
    cat = _normalize_category(info.get("macro_category") or info.get("categoria_macro"))
    if cat not in _DESSERT_ALLOWED_CATEGORIES and cat not in _DESSERT_EXTRA_CATEGORIES:
        return False

    fam = _normalize_text(info.get("family") or info.get("familia"))
    flavors_raw = info.get("base_flavors") or info.get("sabors_base") or ""
    flavors = {_normalize_text(f) for f in str(flavors_raw).split("|") if f}

    if flavors & _DESSERT_SAVORY_FLAVORS:
        return False

    # Evita formatges salats en postres, a no ser que siguin suaus/dolcos.
    if "cheese" in fam and not (flavors & _DESSERT_SOFT_FLAVORS):
        return False

    # Categories extra: exigeix senyals dolcos/fruit/citrus.
    if cat in _DESSERT_EXTRA_CATEGORIES:
        if not (flavors & _DESSERT_GOOD_FLAVORS):
            return False

    return True

# ---------------------------------------------------------------------
# OPERADORS PRINCIPALS (INTERFÍCIE)
# ---------------------------------------------------------------------
def substituir_ingredient(plat: Dict[str, Any], target: str, kb: Any, estils_latents: Dict = None,
                          mode: str = "restriccio", intensitat: float = 0.5,
                          perfil_usuari: Optional[Dict] = None, llista_blanca: Optional[Set[str]] = None,
                          parelles_prohibides: Optional[Set[str]] = None, ingredients_estil_usats: Optional[Set[str]] = None) -> Dict[str, Any]:
    """Punt d'entrada principal per a substitucions."""
    if mode == "latent":
        return _adaptar_latent_core(plat, target, kb, estils_latents, intensitat,
                                    parelles_prohibides, perfil_usuari, ingredients_estil_usats)
    return plat

def adaptar_plat_a_estil_latent(plat: Dict[str, Any], nom_estil: str, kb: Any, base_estils_latents: Dict,
                                intensitat: float = 0.5, parelles_prohibides: Optional[Set[str]] = None,
                                ingredients_estil_usats: Optional[Set[str]] = None, perfil_usuari: Optional[Dict] = None) -> Dict[str, Any]:
    """Wrapper específic per a l'adaptació creativa d'estils."""
    return _adaptar_latent_core(plat, nom_estil, kb, base_estils_latents, intensitat,
                                parelles_prohibides, perfil_usuari, ingredients_estil_usats)

def substituir_ingredients_prohibits(plat: Dict[str, Any], ingredients_prohibits: Set[str], kb: Any,
                                     perfil_usuari: Optional[Dict] = None, llista_blanca: Optional[Set[str]] = None,
                                     ingredients_usats: Optional[Set[str]] = None, parelles_prohibides: Optional[Set[str]] = None,
                                     preferits: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Substitueix ingredients que violen restriccions dures.
    Utilitza una estratègia híbrida: cerca candidats via ontologia i selecciona el millor via FlavorGraph.
    """
    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients'])
    log_canvis = list(plat.get('log_transformacio', []))
    curs_norm = _normalize_text(plat.get("curs"))
    es_postres = "postres" in curs_norm or "postre" in curs_norm or "dessert" in curs_norm
    
    prohibits_norm = {_normalize_text(i) for i in ingredients_prohibits}
    whitelist_norm = ({_normalize_text(i) for i in llista_blanca} if llista_blanca else None)
    preferits = list(preferits or [])

    for i, ing_nom in enumerate(nou_plat['ingredients']):
        ing_norm = _normalize_text(ing_nom)
        
        if ing_norm in prohibits_norm:
            info_orig = kb.get_info_ingredient(ing_nom)
            if not info_orig: continue 
            
            # Context i candidats potencials
            perfil_context = _build_perfil_context(perfil_usuari, info_orig)
            cat_macro = info_orig.get('macro_category') or info_orig.get('categoria_macro')
            if not cat_macro: continue
            
            cats_candidats = [_normalize_category(cat_macro)]
            cats_candidats.extend(_categoria_fallbacks(_normalize_category(cat_macro), perfil_usuari))
            
            candidats_map = {}
            candidats_dup_map = {}
            context_ingredients = [ing for k, ing in enumerate(nou_plat['ingredients']) if k != i]
            context_norm = {_normalize_text(ing) for ing in context_ingredients}

            # Filtratge de candidats
            for cat in cats_candidats:
                for cand_nom in _get_candidats_per_categoria(cat, kb):
                    c_norm = _normalize_text(cand_nom)
                    
                    if c_norm == ing_norm or c_norm in prohibits_norm: continue
                    if whitelist_norm and c_norm not in whitelist_norm: continue
                    if parelles_prohibides and _check_parelles_prohibides(cand_nom, context_ingredients, parelles_prohibides): continue

                    info_cand = kb.get_info_ingredient(cand_nom)
                    if not info_cand:
                        continue
                    if es_postres and not _es_candidat_postres_segura(info_cand, cand_nom):
                        continue
                    if not _es_substitucio_semanticament_coherent(info_orig, info_cand, ing_nom, cand_nom):
                        continue
                    if _check_compatibilitat(info_cand, perfil_context) and _es_candidat_coherent(info_orig, info_cand, _normalize_text(cat_macro)):
                        if c_norm in context_norm:
                            candidats_dup_map[c_norm] = cand_nom
                        else:
                            candidats_map[c_norm] = cand_nom

            if candidats_map:
                candidats_finals = list(candidats_map.values())
            else:
                candidats_finals = list(candidats_dup_map.values())
            if not candidats_finals:
                candidats_relax = []
                for cat in cats_candidats:
                    for cand_nom in _get_candidats_per_categoria(cat, kb):
                        c_norm = _normalize_text(cand_nom)
                        if c_norm == ing_norm or c_norm in prohibits_norm:
                            continue
                        if whitelist_norm and c_norm not in whitelist_norm:
                            continue
                        if parelles_prohibides and _check_parelles_prohibides(cand_nom, context_ingredients, parelles_prohibides):
                            continue
                        info_cand = kb.get_info_ingredient(cand_nom)
                        if not info_cand:
                            continue
                        if es_postres and not _es_candidat_postres_segura(info_cand, cand_nom):
                            continue
                        if not _es_substitucio_semanticament_coherent(info_orig, info_cand, ing_nom, cand_nom):
                            continue
                        if _check_compatibilitat(info_cand, perfil_context):
                            candidats_relax.append(cand_nom)

                if candidats_relax:
                    vistos = set()
                    candidats_finals = []
                    for cand in candidats_relax:
                        c_norm = _normalize_text(cand)
                        if c_norm in vistos:
                            continue
                        vistos.add(c_norm)
                        candidats_finals.append(cand)

            # Selecció Híbrida: Vectors (FlavorGraph) > Ontologia
            millor_substitut = None
            millor_score = -99.0
            justificacio = ""

            vec_orig = FG_WRAPPER.get_vector(ing_nom)
            vec_context = _calcular_vector_context(context_ingredients)
            use_vectors = vec_orig is not None or vec_context is not None

            # Preferències d'usuari (prioritat si encaixa)
            best_pref = None
            best_pref_score = -99.0
            pref_threshold = 0.08 if use_vectors else 0.0
            pref_candidates = []
            pref_dup_candidates = []
            for pref in preferits:
                info_pref = kb.get_info_ingredient(pref)
                if not info_pref:
                    continue
                pref_name = info_pref.get("ingredient_name") or pref
                pref_norm = _normalize_text(pref_name)
                if pref_norm == ing_norm or pref_norm in prohibits_norm:
                    continue
                if whitelist_norm and pref_norm not in whitelist_norm:
                    continue
                if parelles_prohibides and _check_parelles_prohibides(pref_name, context_ingredients, parelles_prohibides):
                    continue
                if es_postres and not _es_candidat_postres_segura(info_pref, pref_name):
                    continue
                if not _es_substitucio_semanticament_coherent(info_orig, info_pref, ing_nom, pref_name):
                    continue
                if not _check_compatibilitat(info_pref, perfil_context):
                    continue
                if not _es_candidat_coherent(info_orig, info_pref, _normalize_category(cat_macro)):
                    continue
                if pref_norm in context_norm:
                    pref_dup_candidates.append(pref_name)
                else:
                    pref_candidates.append(pref_name)

            pref_pool = pref_candidates or pref_dup_candidates
            if pref_pool:
                for cand in pref_pool:
                    sim_self = FG_WRAPPER.similarity_with_vector(cand, vec_orig) if vec_orig is not None else None
                    sim_pair = FG_WRAPPER.similarity_with_vector(cand, vec_context) if vec_context is not None else None
                    score = (0.65 * (sim_self or 0)) + (0.35 * (sim_pair or 0))
                    if score > best_pref_score:
                        best_pref_score = score
                        best_pref = cand
                if best_pref is not None:
                    effective_threshold = pref_threshold if candidats_finals else 0.0
                    if best_pref_score >= effective_threshold:
                        millor_substitut = best_pref
                        millor_score = best_pref_score
                        justificacio = "Preferència d'usuari"

            if millor_substitut is None and not candidats_finals:
                log_canvis.append(f"Avís: No s'ha trobat substitut segur per {ing_nom}")
                continue

            if use_vectors:
                if millor_substitut is None:
                    for cand in candidats_finals:
                        sim_self = FG_WRAPPER.similarity_with_vector(cand, vec_orig) if vec_orig is not None else None
                        sim_pair = FG_WRAPPER.similarity_with_vector(cand, vec_context) if vec_context is not None else None

                        score = (0.65 * (sim_self or 0)) + (0.35 * (sim_pair or 0))
                        if score > millor_score:
                            millor_score = score
                            millor_substitut = cand

            if millor_substitut:
                if not justificacio:
                    justificacio = f"FlavorGraph (similitud+pairing {millor_score:.2f})"
            else:
                if ordenats := _ordenar_candidats_per_afinitat(candidats_finals, kb, info_orig):
                    millor_substitut = ordenats[0]
                    justificacio = "Ontologia (família/rol)"
                else:
                    millor_substitut = random.choice(candidats_finals)
                    justificacio = "Aleatori"

            nou_plat['ingredients'][i] = millor_substitut
            log_canvis.append(f"Substitució: {ing_nom} -> {millor_substitut} [{justificacio}]")
            if ingredients_usats: ingredients_usats.add(_normalize_text(millor_substitut))

    nou_plat['log_transformacio'] = log_canvis
    return nou_plat

# ---------------------------------------------------------------------
# ADAPTACIÓ LATENT AGRESSIVA (Core Logic)
# ---------------------------------------------------------------------
def _adaptar_latent_core(plat: Dict, nom_estil: str, kb: Any, base_estils_latents: Dict, intensitat: float,
                         parelles_prohibides: Optional[Set[str]] = None, perfil_usuari: Optional[Dict] = None,
                         ingredients_estil_usats: Optional[Set[str]] = None):
    """
    Motor de creativitat: Modifica el plat per apropar-lo a un 'Estil Latent' utilitzant vectors.
    Aplica 3 fases: Substitució (A), Inserció (B) i Fallback Simbòlic (C).
    """
    if not base_estils_latents: return plat
    
    # Configuració inicial de l'Estil
    estil_data = base_estils_latents.get(nom_estil, {})
    vector_estil = FG_WRAPPER.compute_concept_vector(estil_data.get('ingredients', []))
    if vector_estil is None: return plat

    if ingredients_estil_usats is None: ingredients_estil_usats = set()

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients']) 
    log = []
    es_postres = "postres" in str(plat.get('curs')).lower()
    
    # Paràmetres d'exploració
    temperatura = min(0.99, 0.4 + intensitat)
    n_search = 30 + (len(ingredients_estil_usats) * 5)
    canvis_fets = 0

    # FASE A: SUBSTITUCIÓ (Prioritat: Transformació estructural)
    # Busca substituir ingredients existents per opcions més properes a l'estil objectiu.    
    for i, ing_original in enumerate(nou_plat['ingredients']):
        vec_orig = FG_WRAPPER.get_vector(ing_original)
        if vec_orig is None: continue
        
        sim_style_orig = FG_WRAPPER.similarity_with_vector(ing_original, vector_estil) or 0.0
        if sim_style_orig > 0.85: continue # L'ingredient ja és idoni

        # Cerca creativa (temperatura alta)
        candidats = FG_WRAPPER.get_creative_candidates(ing_original, n=n_search, temperature=temperatura, style_vector=vector_estil)
        
        info_orig = kb.get_info_ingredient(ing_original)
        if not info_orig:
            continue
        cat_orig = _normalize_category(info_orig.get('macro_category') or "unknown")
        
        millor_cand = None
        millor_score_hibrid = -99.0
        
        # Pesos agressius: Prioritzem Estil (60%) sobre Pairing (35%) i Fidelitat Original (5%)
        W_STYLE, W_PAIRING, W_SELF = 0.60, 0.35, 0.05
        vec_context = _calcular_vector_context(nou_plat['ingredients'], exclude_index=i)

        for cand, _ in candidats:
            cand_norm = _normalize_text(cand)
            if cand_norm == _normalize_text(ing_original) or cand_norm in ingredients_estil_usats: continue

            info_cand = kb.get_info_ingredient(cand)
            if not info_cand: continue
            
            # Validacions de seguretat i ontològiques
            context_noms = [nou_plat['ingredients'][k] for k in range(len(nou_plat['ingredients'])) if k != i]
            if parelles_prohibides and _check_parelles_prohibides(cand, context_noms, parelles_prohibides): continue
            if not _check_compatibilitat(info_cand, perfil_usuari): continue
            
            cat_cand = _normalize_category(info_cand.get("macro_category") or "unknown")
            if es_postres:
                if not _es_apte_postres(info_cand, intensitat):
                    continue
                # Relaxacio per postres (permet canviar fruita per dolc si intensitat es alta)
                if cat_orig == "fruit" and cat_cand not in {"fruit", "nuts"} and not (intensitat > 0.6 and cat_cand == "sweetener"):
                    continue
            else:
                if not _check_role_compatibility(cat_orig, cat_cand): continue

            # Càlcul de Puntuació Híbrida
            sim_style = FG_WRAPPER.similarity_with_vector(cand, vector_estil) or 0.0
            sim_self = FG_WRAPPER.similarity_with_vector(cand, vec_orig) or 0.0
            
            sim_pairing = 0.0
            if vec_context is not None:
                sim_pair_raw = FG_WRAPPER.similarity_with_vector(cand, vec_context)
                if sim_pair_raw: sim_pairing = sim_pair_raw
            
            score = (W_STYLE * sim_style) + (W_PAIRING * sim_pairing) + (W_SELF * sim_self)
            score_current = (W_STYLE * sim_style_orig) + (W_PAIRING * sim_pairing) + (W_SELF * 1.0)
            
            if score > score_current or (sim_style > sim_style_orig + 0.1):
                if score > millor_score_hibrid:
                    millor_score_hibrid = score
                    millor_cand = cand

        if millor_cand:
            nou_plat['ingredients'][i] = millor_cand
            log.append(f"Estil {nom_estil}: Substituït {ing_original} per {millor_cand}")
            ingredients_estil_usats.add(_normalize_text(millor_cand))
            canvis_fets += 1

    # FASE B: INSERCIÓ (Enriquiment)
    # Si el plat encara és lluny de l'estil, s'afegeixen ingredients representatius (tocs).
    sim_global = 0.0
    if (vp := _vector_promig_plat(nou_plat['ingredients'])) is not None:
        if (ve := _normalize_vector(vector_estil)) is not None: sim_global = float(np.dot(vp, ve))
    
    TARGET_SIM = 0.88 if es_postres else 0.82

    if ((canvis_fets == 0) or (sim_global < TARGET_SIM)) and (len(nou_plat['ingredients']) < 9) and (intensitat >= 0.3):
        rescue_mode = n_search >= 50
        representants = FG_WRAPPER.get_style_representatives(vector_estil, n=n_search, exclude_names=nou_plat['ingredients'])
        
        millor_toc = None
        millor_val = -1.0
        vec_context_final = _calcular_vector_context(nou_plat['ingredients']) 

        # Cerca del millor complement
        for cand, score_style in representants:
            if _normalize_text(cand) in ingredients_estil_usats: continue

            if info_cand := kb.get_info_ingredient(cand):
                if parelles_prohibides and _check_parelles_prohibides(cand, nou_plat['ingredients'], parelles_prohibides): continue
                if not _check_compatibilitat(info_cand, perfil_usuari): continue
                
                if es_postres and not _es_apte_postres(info_cand, intensitat):
                    continue

                pairing = 0.0
                if vec_context_final is not None:
                    if p := FG_WRAPPER.similarity_with_vector(cand, vec_context_final): pairing = p
                
                # En mode rescat donem tot el pes a l'estil, sinó balancegem amb pairing
                score_final = score_style if rescue_mode else (0.6 * score_style) + (0.4 * pairing)
                
                # Llindar mínim per acceptar inserció
                threshold = 0.4 if rescue_mode else -1.0
                if score_final > threshold and score_final > millor_val:
                    millor_val = score_final
                    millor_toc = cand

        if millor_toc:
            nou_plat['ingredients'].append(millor_toc)
            log.append(f"Estil {nom_estil}: Afegit {millor_toc} com a toc final.")
            ingredients_estil_usats.add(_normalize_text(millor_toc))

    # FASE C: FALLBACK SIMBÒLIC (Últim recurs)
    # Si no s'han fet canvis vectorials, s'intenten substitucions clàssiques (ex: oli -> oli de sèsam).
    if not log:
        candidats_estil = [
            c for c in (base_estils_latents.get(nom_estil) or {}).get("ingredients", [])
            if _normalize_text(c) not in ingredients_estil_usats
        ]
        
        # Pre-càlcul de pairing per candidats simbòlics
        vec_ctx = _calcular_vector_context(nou_plat.get("ingredients", []))
        puntuats = []
        for cand in candidats_estil:
            if info := kb.get_info_ingredient(cand):
                if not _check_compatibilitat(info, perfil_usuari): continue
                cat = _normalize_category(info.get("macro_category") or "unknown")
                if es_postres and not _es_apte_postres(info, intensitat):
                    continue
                
                sim = 0.0
                if vec_ctx is not None:
                     if s := FG_WRAPPER.similarity_with_vector(cand, vec_ctx): sim = s
                puntuats.append((cand, cat, sim))
        
        puntuats.sort(key=lambda x: x[2], reverse=True)
        
        accio = None
        boring_ingredients = {"oil", "sunflower oil", "sugar", "water", "vinegar", "cream", "milk"}

        # Intent de substitució d'ingredients "avorrits"
        if puntuats:
            for i, ing in enumerate(nou_plat.get("ingredients", [])):
                if _normalize_text(ing) in boring_ingredients:
                    info_ing = kb.get_info_ingredient(ing)
                    cat_ing = _normalize_category(info_ing.get("macro_category") or "unknown") if info_ing else ""
                    
                    for cand, cat_cand, sim_pair in puntuats:
                        if cat_cand == cat_ing: # Match categòric
                            nou_plat["ingredients"][i] = cand
                            accio = f"Substituït {ing} per {cand} (Fallback Simbòlic)"
                            ingredients_estil_usats.add(_normalize_text(cand))
                            break
                if accio: break

        # Inserció directa si falla la substitució
        if not accio and puntuats:
            existing = {_normalize_text(x) for x in nou_plat.get("ingredients", [])}
            cand = None
            for cand_opt, _, _ in puntuats:
                if _normalize_text(cand_opt) not in existing:
                    cand = cand_opt
                    break
            if cand:
                nou_plat["ingredients"].append(cand)
                accio = f"Afegit {cand} (Fallback Simbòlic)"
                ingredients_estil_usats.add(_normalize_text(cand))

        if accio: log.append(f"Estil {nom_estil}: {accio}")

    # FASE D: CONDIMENT LATENT (extra opcional)
    course_key = _map_course_to_condiment_key(plat)
    if course_key and not nou_plat.get("condiment"):
        rng = random
        temp_condiment = _clamp(float(intensitat), 0.1, 0.9)
        fallback_mode = canvis_fets == 0
        random_mode = _condiment_random_mode(temp_condiment, rng)
        condiment = pick_latent_condiment(
            nom_estil,
            course_key,
            temp_condiment,
            fallback_mode=fallback_mode,
            random_mode=random_mode,
            rng=rng,
        )
        if condiment:
            nou_plat["condiment"] = condiment
            log.append(f"Condiment latent: {condiment}")

    nou_plat['log_transformacio'] = log
    return nou_plat
