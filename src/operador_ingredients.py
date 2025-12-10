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
    llista_blanca: Optional[Set[str]] = None
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

            # 2. Generem candidats vàlids (Ontologia)
            possibles_candidats = _get_candidats_per_categoria(categoria, base_ingredients)
            candidats_map = {}
            ingredients_actuals_norm = {
                _normalize_text(nou_plat['ingredients'][idx])
                for idx in range(len(nou_plat['ingredients']))
                if idx != i
            }

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
                if whitelist_norm and cand_norm not in whitelist_norm:
                    continue
                info_cand = _get_info_ingredient(candidat, base_ingredients)
                if not info_cand:
                    continue
                if not _check_compatibilitat(info_cand, perfil_context):
                    continue
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
            
            # Primer intent: El millor candidat segons FlavorGraph que respecti l'ontologia
            for nom_candidat, score in ranking:
                cand_norm = _normalize_text(nom_candidat)
                if cand_norm in candidats_map:
                    millor_substitut = candidats_map[cand_norm]
                    justificacio = f"FlavorGraph (score {score:.2f})"
                    break
            
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
    perfil_usuari: Optional[Dict] = None
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
    vector_estil = FG_WRAPPER.compute_concept_vector(ingredients_estil)
    
    if vector_estil is None:
        print(f"[AVÍS] No s'ha pogut calcular el vector per '{nom_estil}'")
        return plat

    perfil_global = perfil_usuari or info_estil.get('perfil_usuari') or {}

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients'])
    log = []

    min_millora = 0.05 + intensitat * 0.3
    temperatura = min(0.9, 0.2 + intensitat)
    similitud_inicial = _similitud_plat_estil(nou_plat['ingredients'], vector_estil)
    if similitud_inicial is not None:
        log.append(f"Similitud inicial amb '{nom_estil}': {similitud_inicial:.2f}")

    for i, ing_original in enumerate(nou_plat['ingredients']):
        # Mirem si l'ingredient original ja és "prou proper" a l'estil
        vec_original = FG_WRAPPER.get_vector(ing_original)
        if vec_original is None: continue

        # Distància Cosinus entre ingredient i estil
        similitud_estil = FG_WRAPPER.similarity_with_vector(ing_original, vector_estil)
        if similitud_estil is None:
            continue
        if similitud_estil > 0.75:
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

        millor = None
        millor_gain = min_millora
        for cand, _ in candidats:
            info_cand = _get_info_ingredient(cand, base_ingredients)
            if not info_cand:
                continue
            if not _check_compatibilitat(info_cand, perfil_global):
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
            log.append(f"Estil {nom_estil}: {ing_original} -> {nom_final} (+{millor_gain:.2f} similitud)")

    similitud_post_subs = _similitud_plat_estil(nou_plat['ingredients'], vector_estil) or 0.0
    target_sim = min(0.85, 0.6 + intensitat * 0.25)

    if similitud_post_subs < target_sim:
        deficit = target_sim - similitud_post_subs
        max_afegits = 1
        ingredients_normals = {_normalize_text(ing) for ing in nou_plat['ingredients']}
        candidate_pool = [ing.get('ingredient_name') for ing in base_ingredients if ing.get('ingredient_name')]
        representants = FG_WRAPPER.get_style_representatives(
            vector_estil,
            n=6,
            exclude_names=nou_plat['ingredients'],
            candidate_pool=candidate_pool
        )

        afegits = 0
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
            if not _check_compatibilitat(info_cand, perfil_global):
                continue
            nou_plat['ingredients'].append(nom_cand)
            ingredients_normals.add(cand_norm)
            afegits += 1
            log.append(f"Estil {nom_estil}: afegit {nom_cand} (representant, score {score:.2f})")

    similitud_final = _similitud_plat_estil(nou_plat['ingredients'], vector_estil)
    if similitud_final is not None:
        log.append(f"Similitud final amb '{nom_estil}': {similitud_final:.2f}")

    nou_plat['log_transformacio'] = log
    return nou_plat
