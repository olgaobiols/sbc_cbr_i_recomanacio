import random
import unicodedata
from typing import List, Dict, Set, Any, Optional
import numpy as np
from flavorgraph_embeddings import FlavorGraphWrapper

# Inicialitzem el motor d'IA un sol cop (o es pot passar com argument)
# Assegura't que els paths siguin correctes respecte on executes el main
FG_WRAPPER = FlavorGraphWrapper()

# ---------------------------------------------------------------------
# 1. FUNCIONS AUXILIARS (Gestió de Dades)
# ---------------------------------------------------------------------

_ING_LOOKUP_CACHE: Dict[int, Dict[str, Dict]] = {}

def _normalize_text(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("-", " ").replace("_", " ").lower().strip()
    text = " ".join(text.split())
    return text

def _check_compatibilitat(ingredient_info: Dict, perfil_usuari: Optional[Dict]) -> bool:
    """
    Retorna False si l'ingredient xoca amb alguna restricció de l'usuari.
    perfil_usuari ex: {'alergies': {'crustaceans', 'nuts'}, 'dieta': 'vegan'}
    """
    if not ingredient_info:
        return False
    if not perfil_usuari:
        return True

    # 1. Al·lèrgies
    alergies_usuari = {
        _normalize_text(a) for a in perfil_usuari.get('alergies', []) if a
    }
    if alergies_usuari:
        alergens_ing = {
            _normalize_text(part)
            for part in str(ingredient_info.get('allergens', '')).split('|')
            if part
        }
        familia_ing = _normalize_text(ingredient_info.get('family'))
        if alergies_usuari.intersection(alergens_ing):
            return False
        if familia_ing and familia_ing in alergies_usuari:
            return False

    # 2. Dietes (ex: vegan, halal)
    dieta_usuari = perfil_usuari.get('dieta')
    if dieta_usuari:
        dieta_norm = _normalize_text(dieta_usuari)
        dietes_ing = {
            _normalize_text(part)
            for part in str(ingredient_info.get('allowed_diets', '')).split('|')
            if part
        }
        if dieta_norm and dieta_norm not in dietes_ing:
            return False

    return True

def _get_lookup_table(base_ingredients: List[Dict]) -> Dict[str, Dict]:
    """
    Construeix (i cacheja) un índex {nom_normalitzat: info}.
    """
    key = id(base_ingredients)
    lookup = _ING_LOOKUP_CACHE.get(key)
    if lookup is not None:
        return lookup

    lookup = {}
    for ing in base_ingredients:
        nom = _normalize_text(ing.get('ingredient_name'))
        if nom:
            lookup[nom] = ing
    _ING_LOOKUP_CACHE[key] = lookup
    return lookup

def _build_perfil_context(perfil_base: Optional[Dict], info_prohibit: Dict) -> Dict:
    """
    A partir de la informació del client i de l'ingredient prohibit,
    ampliem el conjunt d'al·lèrgies (ex: 'shrimp' -> 'crustacean').
    """
    perfil = {}
    if perfil_base:
        perfil.update(perfil_base)

    alergies_actuals = {
        _normalize_text(a) for a in perfil.get('alergies', []) if a
    }

    alergens = str(info_prohibit.get('allergens', '')).split('|')
    for tag in alergens:
        tag_norm = _normalize_text(tag)
        if tag_norm:
            alergies_actuals.add(tag_norm)

    familia = _normalize_text(info_prohibit.get('family'))
    if familia:
        alergies_actuals.add(familia)

    perfil['alergies'] = alergies_actuals
    return perfil

def _get_info_ingredient(nom_ingredient: str, base_ingredients: List[Dict]) -> Dict:
    """
    Busca la informació completa d'un ingredient (categoria, família, etc.)
    dins de la llista de diccionaris carregada del CSV.
    """
    lookup = _get_lookup_table(base_ingredients)
    return lookup.get(_normalize_text(nom_ingredient), {})

def _get_candidats_per_categoria(categoria: str, base_ingredients: List[Dict]) -> List[str]:
    """
    Retorna tots els noms d'ingredients que pertanyen a una mateixa categoria macro.
    Ex: Si categoria='proteina_animal', retorna ['pollastre', 'vedella', 'porc'...]
    """
    categoria_norm = _normalize_text(categoria)
    return [
        ing.get('ingredient_name')
        for ing in base_ingredients
        if _normalize_text(ing.get('macro_category') or ing.get('categoria_macro')) == categoria_norm
        and ing.get('ingredient_name')
    ]


_PROTEINA_CATEGORIES = {
    _normalize_text("protein_animal"),
    _normalize_text("protein vegetal"),
    _normalize_text("protein_vegetal"),
    _normalize_text("fish_white"),
    _normalize_text("fish_oily"),
    _normalize_text("proteina_animal"),
    _normalize_text("proteina vegetal"),
}
_DAIRY_CATEGORIES = {
    _normalize_text("dairy"),
    _normalize_text("lacti"),
    _normalize_text("dairy_cheese"),
    _normalize_text("dairy_cream"),
}
_EGG_CATEGORIES = {_normalize_text("egg")}
_GRAIN_CATEGORIES = {
    _normalize_text("grain"),
    _normalize_text("processed_cereal"),
    _normalize_text("cereal_feculent"),
}
_PLANT_PROTEIN_FALLBACKS = [
    _normalize_text("plant_vegetal"),
    _normalize_text("protein_vegetal"),
    _normalize_text("grain"),
]
_DAIRY_FALLBACKS = [
    _normalize_text("plant_vegetal"),
    _normalize_text("other"),
    _normalize_text("fat"),
]
_EGG_FALLBACKS = [
    _normalize_text("plant_vegetal"),
    _normalize_text("grain"),
]

_MAIN_ROLES = {
    _normalize_text("main"),
    _normalize_text("principal"),
    _normalize_text("main course"),
}
_SIDE_ROLES = {
    _normalize_text("side"),
    _normalize_text("base"),
}

_FAMILY_SOY = {
    _normalize_text("soy"),
    _normalize_text("soy_derivative"),
    _normalize_text("tempeh"),
    _normalize_text("tofu"),
}
_FAMILY_LEGUME = {
    _normalize_text("legume"),
    _normalize_text("chickpea"),
    _normalize_text("lentil"),
}
_FAMILY_DAIRY_ALT = {
    _normalize_text("dairy_substitute"),
    _normalize_text("coconut"),
    _normalize_text("oat"),
    _normalize_text("rice"),
    _normalize_text("soy"),
    _normalize_text("soy_derivative"),
}
_FAMILY_NUT_ALTERNATIVE = {
    _normalize_text("almond_nut"),
    _normalize_text("cashew"),
    _normalize_text("hazelnut_nut"),
    _normalize_text("macadamia"),
}


def _categoria_fallbacks(categoria_norm: str, perfil_usuari: Optional[Dict]) -> List[str]:
    if not perfil_usuari:
        return []
    dieta = _normalize_text(perfil_usuari.get('dieta'))
    alergies = {
        _normalize_text(a) for a in perfil_usuari.get('alergies', []) if a
    }
    fallbacks: List[str] = []

    if dieta in {"vegan", "vegetarian"}:
        if categoria_norm in _PROTEINA_CATEGORIES:
            fallbacks.extend(_PLANT_PROTEIN_FALLBACKS)
        if categoria_norm in _DAIRY_CATEGORIES:
            fallbacks.extend(_DAIRY_FALLBACKS)
        if categoria_norm in _EGG_CATEGORIES:
            fallbacks.extend(_EGG_FALLBACKS)
    if "gluten" in alergies and categoria_norm in _GRAIN_CATEGORIES:
        fallbacks.extend([
            _normalize_text("plant_vegetal"),
            _normalize_text("vegetable"),
        ])

    result = []
    seen = set()
    for cat in fallbacks:
        if cat and cat not in seen and cat != categoria_norm:
            seen.add(cat)
            result.append(cat)
    return result


def _rol_principal(role: str) -> bool:
    return _normalize_text(role) in _MAIN_ROLES


def _rol_acceptable(role: str) -> bool:
    role_norm = _normalize_text(role)
    return role_norm in _MAIN_ROLES or role_norm in _SIDE_ROLES


def _es_candidat_coherent(info_original: Dict, info_candidat: Dict, categoria_original_norm: str) -> bool:
    role_orig = _normalize_text(
        info_original.get('typical_role') or info_original.get('rol_tipic') or ""
    )
    role_cand = _normalize_text(
        info_candidat.get('typical_role') or info_candidat.get('rol_tipic') or ""
    )

    if categoria_original_norm in _PROTEINA_CATEGORIES or role_orig in _MAIN_ROLES:
        return role_cand in _MAIN_ROLES

    if role_orig in _SIDE_ROLES and role_cand:
        return role_cand in _SIDE_ROLES or role_cand in _MAIN_ROLES

    return True


def _prioritat_reempla(info_original: Dict, info_candidat: Dict, categoria_original_norm: str) -> int:
    cat_cand = _normalize_text(
        info_candidat.get('macro_category') or info_candidat.get('categoria_macro') or ""
    )
    if cat_cand == categoria_original_norm:
        return 100

    family = _normalize_text(info_candidat.get('family'))

    if categoria_original_norm in _PROTEINA_CATEGORIES:
        if family in _FAMILY_SOY:
            return 5
        if family in _FAMILY_LEGUME:
            return 3
        return 1

    if categoria_original_norm in _DAIRY_CATEGORIES:
        if family in _FAMILY_DAIRY_ALT:
            return 5
        if family in _FAMILY_NUT_ALTERNATIVE:
            return 3
        return 1

    if categoria_original_norm in _EGG_CATEGORIES:
        if family in _FAMILY_SOY:
            return 4
        if family in _FAMILY_LEGUME:
            return 2
        return 1

    return 1

def _ordenar_candidats_per_afinitat(
    candidats: List[str],
    base_ingredients: List[Dict],
    info_original: Dict
) -> List[str]:
    """
    Donada una llista de candidats vàlids, retorna una versió ordenada
    prioritzant família i rol típic semblant a l'ingredient original.
    """
    prefer_family = _normalize_text(info_original.get('family'))
    prefer_role = _normalize_text(info_original.get('typical_role'))

    ranking = []
    for candidat in candidats:
        info_cand = _get_info_ingredient(candidat, base_ingredients)
        score = 0
        if info_cand:
            if prefer_family and _normalize_text(info_cand.get('family')) == prefer_family:
                score += 2
            if prefer_role and _normalize_text(info_cand.get('typical_role')) == prefer_role:
                score += 1
        ranking.append((score, candidat))

    ranking.sort(key=lambda x: (-x[0], x[1]))
    return [c for _, c in ranking]

def _normalize_vector(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None:
        return None
    norm = np.linalg.norm(vec)
    if norm == 0:
        return None
    return vec / norm

def _vector_promig_plat(ingredients: List[str]) -> Optional[np.ndarray]:
    vectors = []
    for ing in ingredients:
        vec = FG_WRAPPER.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None
    mitjana = np.mean(vectors, axis=0)
    return _normalize_vector(mitjana)

def _similitud_plat_estil(ingredients: List[str], vector_estil: Optional[np.ndarray]) -> Optional[float]:
    if vector_estil is None:
        return None
    vector_plat = _vector_promig_plat(ingredients)
    vector_estil_norm = _normalize_vector(vector_estil)
    if vector_plat is None or vector_estil_norm is None:
        return None
    return float(np.dot(vector_plat, vector_estil_norm))

# ---------------------------------------------------------------------
# 2. NUCLI DE L'OPERADOR (Lògica Híbrida)
# ---------------------------------------------------------------------

def substituir_ingredients_prohibits(
    plat: Dict[str, Any], 
    ingredients_prohibits: Set[str], 
    base_ingredients: List[Dict],
    perfil_usuari: Optional[Dict] = None,
    llista_blanca: Optional[Set[str]] = None,
    ingredients_usats: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """
    Revisa els ingredients del plat. Si en troba un de prohibit, el substitueix
    fent servir:
    1. ONTOLOGIA: Per mantenir la coherència (Carn per Carn, Fruita per Fruita).
    2. FLAVORGRAPH: Per trobar el més similar sensorialment dins la categoria.
    """
    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients']) # Copia per no modificar l'original
    log_canvis = []
    prohibits_norm = {_normalize_text(ing) for ing in ingredients_prohibits}
    whitelist_norm = (
        {_normalize_text(ing) for ing in llista_blanca} if llista_blanca else None
    )

    for i, ingredient_actual in enumerate(nou_plat['ingredients']):
        ing_actual_norm = _normalize_text(ingredient_actual)
        
        # Si l'ingredient està a la llista negra
        if ing_actual_norm in prohibits_norm:
            
            # 1. Recuperem metadades (què és aquest ingredient?)
            info = _get_info_ingredient(ingredient_actual, base_ingredients)
            if not info:
                print(f"[AVÍS] No tenim informació per substituir '{ingredient_actual}'. Es manté.")
                continue
            perfil_context = _build_perfil_context(perfil_usuari, info)

            categoria = info.get('macro_category') or info.get('categoria_macro')

            if not categoria:
                # Si no sabem què és, no ens arrisquem a canviar-lo malament
                print(f"[AVÍS] No tenim informació per substituir '{ingredient_actual}'. Es manté.")
                continue

            categoria_norm = _normalize_text(categoria)
            categories_considerar = []
            cats_seen = set()

            def _afegeix_categoria(cat_txt: str):
                cat_norm = _normalize_text(cat_txt)
                if cat_norm and cat_norm not in cats_seen:
                    cats_seen.add(cat_norm)
                    categories_considerar.append(cat_norm)

            _afegeix_categoria(categoria_norm)
            for extra_cat in _categoria_fallbacks(categoria_norm, perfil_usuari):
                _afegeix_categoria(extra_cat)

            candidats_map = {}
            ingredients_actuals_norm = {
                _normalize_text(nou_plat['ingredients'][idx])
                for idx in range(len(nou_plat['ingredients']))
                if idx != i
            }

            for categoria_target in categories_considerar:
                possibles_candidats = _get_candidats_per_categoria(categoria_target, base_ingredients)
                for candidat in possibles_candidats:
                    cand_norm = _normalize_text(candidat)
                    if not cand_norm:
                        continue
                    if cand_norm == ing_actual_norm:
                        continue
                    if cand_norm in prohibits_norm:
                        continue
                    if cand_norm in ingredients_actuals_norm:
                        continue
                    if ingredients_usats and cand_norm in ingredients_usats:
                        continue
                    if whitelist_norm and cand_norm not in whitelist_norm:
                        continue
                    info_cand = _get_info_ingredient(candidat, base_ingredients)
                    if not info_cand:
                        continue
                    if not _check_compatibilitat(info_cand, perfil_context):
                        continue
                    if not _es_candidat_coherent(info, info_cand, categoria_norm):
                        continue
                    if cand_norm not in candidats_map:
                        candidats_map[cand_norm] = candidat

            candidats_filtrats = list(candidats_map.values())

            if not candidats_filtrats:
                print(f"[AVÍS] No hi ha substituts disponibles per '{ingredient_actual}' a la categoria '{categoria}'.")
                continue

            # 3. Rànquing intel·ligent (FlavorGraph)
            # Busquem quin dels candidats s'assembla més a l'original
            ranking = []
            vector = FG_WRAPPER.get_vector(ingredient_actual)

            if vector is not None:
                ranking = FG_WRAPPER._find_nearest_to_vector(
                    target_vector=vector,
                    n=8,  # busquem una mica més de marge
                    exclude_names=list(prohibits_norm | {ing_actual_norm})  # Seguretat extra
                )
            
            # Intersecció: Volem el millor del rànquing que TAMBÉ sigui de la categoria correcta
            millor_substitut = None
            justificacio = None
            millor_tuple = (-1, -999.0)

            # Primer intent: El millor candidat segons FlavorGraph que respecti l'ontologia
            for nom_candidat, score in ranking:
                cand_norm = _normalize_text(nom_candidat)
                candidat = candidats_map.get(cand_norm)
                if not candidat:
                    continue
                info_cand = _get_info_ingredient(candidat, base_ingredients)
                priority = _prioritat_reempla(info, info_cand, categoria_norm)
                tup = (priority, score)
                if tup > millor_tuple:
                    millor_tuple = tup
                    millor_substitut = candidat
                    justificacio = f"FlavorGraph (score {score:.2f})"

            # Segon intent (Fallback): Si FlavorGraph no troba res proper a la categoria,
            # ordenem per família/rol per mantenir coherència gastronòmica.
            if not millor_substitut:
                ordenats = _ordenar_candidats_per_afinitat(candidats_filtrats, base_ingredients, info)
                for cand in ordenats:
                    cand_norm = _normalize_text(cand)
                    if cand_norm in candidats_map:
                        millor_substitut = cand
                        justificacio = "Ontologia (família/rol similar)"
                        break

            # Últim recurs: escollim un aleatori per no deixar el plat sense substitut
            if not millor_substitut and candidats_filtrats:
                millor_substitut = random.choice(candidats_filtrats)
                justificacio = "Ontologia (aleatori)"

            if not millor_substitut:
                print(f"[AVÍS] Cap substitut compatible per '{ingredient_actual}'.")
                continue

            # 4. Aplicar canvi
            nou_plat['ingredients'][i] = millor_substitut
            log_canvis.append(
                f"Substitució: {ingredient_actual} -> {millor_substitut} (Categoria: {categoria}, {justificacio})"
            )
            if ingredients_usats is not None:
                ingredients_usats.add(_normalize_text(millor_substitut))

    # Guardem el registre de canvis al plat per explicar-ho a l'usuari després
    nou_plat['log_transformacio'] = log_canvis
    return nou_plat

# ---------------------------------------------------------------------
# 3. CAS D'ÚS ESPECÍFIC: ADAPTACIÓ A ESTIL
# ---------------------------------------------------------------------

def adaptar_plat_a_estil(
    plat: Dict,
    nom_estil: str,
    base_estils: Dict,
    base_ingredients: List[Dict],
    perfil_usuari: Optional[Dict] = None
):
    """
    Wrapper que utilitza la funció genèrica de dalt.
    Determina quins ingredients sobren segons l'estil i els substitueix.
    """
    info_estil = base_estils.get(nom_estil)
    if not info_estil:
        return plat

    # Obtenim ingredients típics de l'estil
    ingredients_permesos = set(info_estil.get('ingredients', []))
    ingredients_permesos_norm = {_normalize_text(ing) for ing in ingredients_permesos}
    
    # Identifiquem els del plat que NO encaixen (ingredients prohibits en aquest context)
    # NOTA: Això és estricte. Si un ingredient no està a la definició de l'estil, es canvia.
    # Podries relaxar-ho comprovant només ingredients molt clau.
    prohibits = {
        ing for ing in plat['ingredients'] 
        if _normalize_text(ing) not in ingredients_permesos_norm
    }
    
    if not prohibits:
        return plat # El plat ja encaixa

    perfil = perfil_usuari or info_estil.get('perfil_usuari')

    print(f"--- Adaptant '{plat['nom']}' a l'estil {nom_estil} ---")
    return substituir_ingredients_prohibits(
        plat,
        prohibits,
        base_ingredients,
        perfil_usuari=perfil,
        llista_blanca=ingredients_permesos
    )

def adaptar_plat_a_estil_latent(
    plat: Dict,
    nom_estil: str,
    base_estils: Dict,
    base_ingredients: List[Dict],
    intensitat: float = 0.4,
    perfil_usuari: Optional[Dict] = None,
    ingredients_usats_latent: Optional[Set[str]] = None,
):
    """
    Adapta un plat movent els ingredients cap a la direcció vectorial de l'estil,
    sense necessitat de llistes blanques estrictes.
    """
    info_estil = base_estils.get(nom_estil)
    if not info_estil: return plat

    # 1. Construïm el vector de l'estil (Dimensions Latents)
    # L'estil al CSV té una llista d'ingredients representatius. Els usem per crear el concepte.
    ingredients_estil = info_estil.get('ingredients', [])
    style_normals = {_normalize_text(ing) for ing in ingredients_estil if ing}
    vector_estil = FG_WRAPPER.compute_concept_vector(ingredients_estil)
    
    if vector_estil is None:
        print(f"[AVÍS] No s'ha pogut calcular el vector per '{nom_estil}'")
        return plat

    perfil_global = perfil_usuari or info_estil.get('perfil_usuari') or {}
    curs = (plat.get("curs") or "").lower()
    es_postres = "postres" in curs or "dessert" in curs
    global_latent = ingredients_usats_latent
    dessert_categories = {"sweet", "fruit", "dairy", "sweetener"}
    dessert_candidates_style: list[tuple[str, Dict, float]] = []
    if es_postres:
        for ing_style in ingredients_estil:
            info_cand = _get_info_ingredient(ing_style, base_ingredients)
            if not info_cand:
                continue
            macro = (info_cand.get('macro_category') or "").lower()
            if macro not in dessert_categories:
                continue
            sim = FG_WRAPPER.similarity_with_vector(ing_style, vector_estil) or 0.0
            dessert_candidates_style.append((ing_style, info_cand, sim))
        if not dessert_candidates_style:
            candidate_pool = [
                ing.get('ingredient_name') for ing in base_ingredients
                if (ing.get('macro_category') or "").lower() in dessert_categories
            ]
            representants = FG_WRAPPER.get_style_representatives(
                vector_estil,
                n=6,
                exclude_names=plat.get('ingredients', []),
                candidate_pool=candidate_pool
            )
            for cand, score in representants:
                info_cand = _get_info_ingredient(cand, base_ingredients)
                if not info_cand:
                    continue
                dessert_candidates_style.append((cand, info_cand, score))
        dessert_candidates_style.sort(key=lambda x: x[2], reverse=True)

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients'])
    log = []
    ingredients_normals = {_normalize_text(ing) for ing in nou_plat['ingredients']}
    max_substitucions = max(1, int(len(nou_plat['ingredients']) * (0.25 + intensitat * 0.4)))
    if es_postres:
        max_substitucions = min(max_substitucions, 1)
    substitucions_realitzades = 0

    min_millora = 0.05 + intensitat * 0.3
    temperatura = min(0.9, 0.2 + intensitat)
    similitud_inicial = _similitud_plat_estil(nou_plat['ingredients'], vector_estil)
    if similitud_inicial is not None:
        log.append(f"Similitud inicial amb '{nom_estil}': {similitud_inicial:.2f}")

    for i, ing_original in enumerate(nou_plat['ingredients']):
        if substitucions_realitzades >= max_substitucions:
            break
        # Mirem si l'ingredient original ja és "prou proper" a l'estil
        vec_original = FG_WRAPPER.get_vector(ing_original)
        if vec_original is None: continue

        # Distància Cosinus entre ingredient i estil
        similitud_estil = FG_WRAPPER.similarity_with_vector(ing_original, vector_estil)
        if similitud_estil is None:
            continue
        llindar_sim = 0.8 if es_postres else 0.7
        if similitud_estil > llindar_sim:
            continue

        # Si no, el "modifiquem"
        print(f"🔄 Adaptant '{ing_original}' cap a '{nom_estil}'...")
        
        # AQUI ESTÀ LA MÀGIA: Vector Original + Vector Estil
        candidats = FG_WRAPPER.get_creative_candidates(
            ingredient_name=ing_original,
            n=8,
            temperature=temperatura,
            style_vector=vector_estil
        )

        # Filtrem el millor candidat que sigui vàlid per restriccions i millori la similitud
        info_orig = _get_info_ingredient(ing_original, base_ingredients)
        cat_orig = info_orig.get('macro_category') if info_orig else None
        ing_norm = _normalize_text(ing_original)

        millor = None
        millor_gain = min_millora
        for cand, _ in candidats:
            info_cand = _get_info_ingredient(cand, base_ingredients)
            if not info_cand:
                continue
            if not _check_compatibilitat(info_cand, perfil_global):
                continue

            macro_cand = info_cand.get('macro_category')
            cand_norm = _normalize_text(info_cand.get('ingredient_name', cand))
            if not cand_norm:
                continue
            if cand_norm != ing_norm and cand_norm in ingredients_normals:
                continue
            if (
                global_latent is not None
                and cand_norm in global_latent
                and cand_norm in style_normals
            ):
                continue

            if es_postres:
                if (macro_cand or "").lower() not in {"sweet", "fruit", "dairy", "sweetener"}:
                    continue
            elif cat_orig and macro_cand and macro_cand != cat_orig and intensitat < 0.7:
                # En plats salats mantenim la categoria llevat d'intensitats altes
                continue

            sim_cand = FG_WRAPPER.similarity_with_vector(cand, vector_estil)
            if sim_cand is None:
                continue
            gain = sim_cand - similitud_estil
            if cat_orig and info_cand.get('macro_category') != cat_orig:
                gain -= 0.05  # penalitza però permet

            if gain >= millor_gain:
                millor = cand
                millor_gain = gain

        if millor and millor != ing_original:
            info_cand = _get_info_ingredient(millor, base_ingredients)
            nom_final = info_cand.get('ingredient_name', millor) if info_cand else millor
            nou_plat['ingredients'][i] = nom_final
            ingredients_normals.discard(ing_norm)
            ingredients_normals.add(_normalize_text(nom_final))
            log.append(f"Estil {nom_estil}: {ing_original} -> {nom_final} (+{millor_gain:.2f} similitud)")
            substitucions_realitzades += 1
            cand_norm = _normalize_text(nom_final)
            if global_latent is not None and cand_norm in style_normals:
                global_latent.add(cand_norm)

    if es_postres and dessert_candidates_style:
        top = dessert_candidates_style[:4]
        random.shuffle(top)
        ordered = top + dessert_candidates_style[len(top):]
        for cand_name, info_cand, sim in ordered:
            nom_cand = info_cand.get('ingredient_name', cand_name)
            cand_norm = _normalize_text(nom_cand)
            if cand_norm in ingredients_normals:
                continue
            if not _check_compatibilitat(info_cand, perfil_global):
                continue
            if global_latent is not None and cand_norm in global_latent:
                continue
            nou_plat['ingredients'].append(nom_cand)
            ingredients_normals.add(cand_norm)
            if global_latent is not None:
                global_latent.add(cand_norm)
            log.append(f"Estil {nom_estil}: afegit {nom_cand} com a toc dolç ({sim:.2f})")
            break

    similitud_post_subs = _similitud_plat_estil(nou_plat['ingredients'], vector_estil) or 0.0
    target_sim = min(0.85, 0.6 + intensitat * 0.25)

    if similitud_post_subs < target_sim:
        deficit = target_sim - similitud_post_subs
        max_afegits = 0 if es_postres else 1
        ingredients_normals = {_normalize_text(ing) for ing in nou_plat['ingredients']}
        candidate_pool = [ing.get('ingredient_name') for ing in base_ingredients if ing.get('ingredient_name')]
        representants = FG_WRAPPER.get_style_representatives(
            vector_estil,
            n=6,
            exclude_names=nou_plat['ingredients'],
            candidate_pool=candidate_pool
        )

        afegits = 0
        random.shuffle(representants)
        for cand, score in representants:
            if afegits >= max_afegits:
                break
            info_cand = _get_info_ingredient(cand, base_ingredients)
            if not info_cand:
                continue
            nom_cand = info_cand.get('ingredient_name', cand)
            cand_norm = _normalize_text(nom_cand)
            if cand_norm in ingredients_normals:
                continue
            if (
                global_latent is not None
                and cand_norm in global_latent
                and cand_norm in style_normals
            ):
                continue
            if es_postres and (info_cand.get('macro_category') or "").lower() not in {"sweet", "fruit", "dairy", "sweetener"}:
                continue
            if not _check_compatibilitat(info_cand, perfil_global):
                continue
            nou_plat['ingredients'].append(nom_cand)
            ingredients_normals.add(cand_norm)
            if global_latent is not None and cand_norm in style_normals:
                global_latent.add(cand_norm)
            afegits += 1
            log.append(f"Estil {nom_estil}: afegit {nom_cand} (representant, score {score:.2f})")

    similitud_final = _similitud_plat_estil(nou_plat['ingredients'], vector_estil)
    if similitud_final is not None:
        log.append(f"Similitud final amb '{nom_estil}': {similitud_final:.2f}")

    nou_plat['log_transformacio'] = log
    return nou_plat
