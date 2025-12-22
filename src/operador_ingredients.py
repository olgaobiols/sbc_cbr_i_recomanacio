import random
import unicodedata
from typing import List, Dict, Set, Any, Optional
import numpy as np

from flavorgraph_embeddings import FlavorGraphWrapper

FG_WRAPPER = FlavorGraphWrapper()
DEBUG_LATENT = False

def _debug_latent(msg: str) -> None:
    if DEBUG_LATENT:
        print(msg)

# ---------------------------------------------------------------------
# 1. FUNCIONS AUXILIARS (Gestió de Dades i Compatibilitat)
# ---------------------------------------------------------------------
def _calcular_vector_context(ingredients: List[str], exclude_index: int = -1) -> Optional[np.ndarray]:
    """
    Calcula el vector mitjà de la resta d'ingredients del plat (Context).
    Serveix per mesurar el 'Pairing': com de bé queda un ingredient amb els seus veïns.
    """
    vectors = []
    for i, ing in enumerate(ingredients):
        if i == exclude_index: continue
        v = FG_WRAPPER.get_vector(ing)
        if v is not None:
            vectors.append(v)
    
    if not vectors: return None
    return np.mean(vectors, axis=0)

def _normalize_vector(vec: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if vec is None: return None
    norm = np.linalg.norm(vec)
    if norm == 0: return None
    return vec / norm
    
def _vector_promig_plat(ingredients: List[str]) -> Optional[np.ndarray]:
    """Calcula el vector mitjà (centroide) de tots els ingredients del plat."""
    vectors = []
    for ing in ingredients:
        vec = FG_WRAPPER.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
    if not vectors: return None
    mitjana = np.mean(vectors, axis=0)
    return _normalize_vector(mitjana)

def _normalize_text(value: str) -> str:
    if not value: return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.replace("-", " ").replace("_", " ").lower().strip()
    return " ".join(text.split())

def _check_compatibilitat(ingredient_info: Dict, perfil_usuari: Optional[Dict]) -> bool:
    """Verifica restriccions dures (al·lèrgies i dietes)."""
    if not ingredient_info: return False
    if not perfil_usuari: return True

    alergies_usuari = {_normalize_text(a) for a in perfil_usuari.get('alergies', []) if a}
    if alergies_usuari:
        alergens_ing = {
            _normalize_text(part)
            for part in str(ingredient_info.get('allergens', '')).split('|') if part
        }
        familia_ing = _normalize_text(ingredient_info.get('family'))
        if alergies_usuari.intersection(alergens_ing): return False
        if familia_ing and familia_ing in alergies_usuari: return False

    dieta_usuari = perfil_usuari.get('dieta')
    if dieta_usuari:
        dieta_norm = _normalize_text(dieta_usuari)
        dietes_ing = {
            _normalize_text(part)
            for part in str(ingredient_info.get('allowed_diets', '')).split('|') if part
        }
        if dieta_norm and dieta_norm not in dietes_ing: return False

    return True

def _check_parelles_prohibides(
    candidat: str, 
    context_ingredients: List[str], 
    parelles_prohibides: Set[str]
) -> bool:
    """
    Retorna True si afegir 'candidat' al plat genera una combinació prohibida
    (segons Canal A o B) amb algun dels ingredients que ja hi són.
    """
    if not parelles_prohibides: return False
    
    cand_norm = _normalize_text(candidat)
    for other in context_ingredients:
        other_norm = _normalize_text(other)
        if cand_norm == other_norm: continue 
        
        # Construïm la clau ordenada (ex: "all|maduixa")
        pair = sorted([cand_norm, other_norm])
        clau = f"{pair[0]}|{pair[1]}"
        
        if clau in parelles_prohibides:
            return True # Conflicte detectat!
            
    return False

def _build_perfil_context(perfil_base: Optional[Dict], info_prohibit: Dict) -> Dict:
    perfil = {}
    if perfil_base: perfil.update(perfil_base)
    alergies = {_normalize_text(a) for a in perfil.get('alergies', []) if a}
    for tag in str(info_prohibit.get('allergens', '')).split('|'):
        t = _normalize_text(tag)
        if t: alergies.add(t)
    fam = _normalize_text(info_prohibit.get('family'))
    if fam: alergies.add(fam)
    perfil['alergies'] = alergies
    return perfil

def ingredients_incompatibles(
    ingredients: List[str],
    kb: Any,
    perfil_usuari: Optional[Dict],
) -> Set[str]:
    prohibits = set()
    if not perfil_usuari:
        return prohibits
    for ing in ingredients:
        info = kb.get_info_ingredient(ing)
        if not info:
            continue
        if not _check_compatibilitat(info, perfil_usuari):
            prohibits.add(ing)
    return prohibits

# --- Lògica Ontològica ---

_PROTEINA_CATEGORIES = {"protein_animal", "protein_vegetal", "fish_white", "fish_oily", "proteina_animal"}
_DAIRY_CATEGORIES = {"dairy", "dairy_cheese", "dairy_cream", "lacti"}
_GRAIN_CATEGORIES = {"grain", "processed_cereal", "cereal_feculent"}
_EGG_CATEGORIES = {"egg"}
_MAIN_ROLES = {"main", "principal", "main course"}
_SIDE_ROLES = {"side", "base"}

def _get_candidats_per_categoria(categoria: str, kb: Any) -> List[str]:
    cat_norm = _normalize_text(categoria)
    candidats = []
    for nom, info in kb.ingredients.items():
        c_macro = _normalize_text(info.get('macro_category') or info.get('categoria_macro'))
        if c_macro == cat_norm:
            candidats.append(info['ingredient_name'])
    return candidats

def _categoria_fallbacks(categoria_norm: str, perfil_usuari: Optional[Dict]) -> List[str]:
    if not perfil_usuari: return []
    dieta = _normalize_text(perfil_usuari.get('dieta'))
    fallbacks = []
    if dieta in {"vegan", "vegetarian"}:
        if categoria_norm in _PROTEINA_CATEGORIES:
            fallbacks.extend(["protein_vegetal", "plant_vegetal", "grain"])
        if categoria_norm in _DAIRY_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "fat", "other"])
        if categoria_norm in _EGG_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "grain"])
    return [_normalize_text(c) for c in fallbacks]

def _es_candidat_coherent(info_orig: Dict, info_cand: Dict, cat_orig_norm: str) -> bool:
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
# 2. OPERADOR PRINCIPAL
# ---------------------------------------------------------------------

def substituir_ingredient(
    plat: Dict[str, Any], 
    target: str, 
    kb: Any,     
    estils_latents: Dict = None,
    mode: str = "restriccio", 
    intensitat: float = 0.5,
    perfil_usuari: Optional[Dict] = None,
    llista_blanca: Optional[Set[str]] = None,
    parelles_prohibides: Optional[Set[str]] = None, # <--- NOU
    ingredients_estil_usats: Optional[Set[str]] = None
) -> Dict[str, Any]:
    
    if mode == "latent":
        return _adaptar_latent_core(
            plat=plat,
            nom_estil=target,
            kb=kb,
            base_estils_latents=estils_latents,
            intensitat=intensitat,
            parelles_prohibides=parelles_prohibides,
            perfil_usuari=perfil_usuari,
            ingredients_estil_usats=ingredients_estil_usats,
        )
    else:
        # Wrapper per compatibilitat si algú crida amb mode='restriccio' directament
        pass
    return plat

def adaptar_plat_a_estil_latent(
    plat: Dict[str, Any],
    nom_estil: str,
    kb: Any,
    base_estils_latents: Dict,
    intensitat: float = 0.5,
    parelles_prohibides: Optional[Set[str]] = None,
    ingredients_estil_usats: Optional[Set[str]] = None,
    perfil_usuari: Optional[Dict] = None,
) -> Dict[str, Any]:
    """Wrapper de compatibilitat per adaptar un plat a un estil latent."""
    return _adaptar_latent_core(
        plat=plat,
        nom_estil=nom_estil,
        kb=kb,
        base_estils_latents=base_estils_latents,
        intensitat=intensitat,
        parelles_prohibides=parelles_prohibides,
        ingredients_estil_usats=ingredients_estil_usats,
        perfil_usuari=perfil_usuari,
    )

def substituir_ingredients_prohibits(
    plat: Dict[str, Any], 
    ingredients_prohibits: Set[str], 
    kb: Any,
    perfil_usuari: Optional[Dict] = None,
    llista_blanca: Optional[Set[str]] = None,
    ingredients_usats: Optional[Set[str]] = None,
    parelles_prohibides: Optional[Set[str]] = None # <--- NOU
) -> Dict[str, Any]:

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients'])
    log_canvis = []
    
    prohibits_norm = {_normalize_text(i) for i in ingredients_prohibits}
    whitelist_norm = ({_normalize_text(i) for i in llista_blanca} if llista_blanca else None)

    for i, ing_nom in enumerate(nou_plat['ingredients']):
        ing_norm = _normalize_text(ing_nom)
        
        if ing_norm in prohibits_norm:
            info_orig = kb.get_info_ingredient(ing_nom)
            if not info_orig: continue 
            
            perfil_context = _build_perfil_context(perfil_usuari, info_orig)
            cat_macro = info_orig.get('macro_category') or info_orig.get('categoria_macro')
            if not cat_macro: continue
            
            cats_candidats = [_normalize_text(cat_macro)]
            cats_candidats.extend(_categoria_fallbacks(_normalize_text(cat_macro), perfil_usuari))
            
            candidats_map = {}
            # Context actual del plat (tots menys el que estem traient)
            context_ingredients = [
                nou_plat['ingredients'][k] for k in range(len(nou_plat['ingredients'])) if k != i
            ]

            for cat in cats_candidats:
                possibles = _get_candidats_per_categoria(cat, kb)
                for cand_nom in possibles:
                    c_norm = _normalize_text(cand_nom)
                    
                    if c_norm == ing_norm: continue
                    if c_norm in prohibits_norm: continue
                    if whitelist_norm and c_norm not in whitelist_norm: continue
                    
                    # 1. Check Parelles Prohibides (Canal A/B)
                    if parelles_prohibides and _check_parelles_prohibides(cand_nom, context_ingredients, parelles_prohibides):
                        continue

                    info_cand = kb.get_info_ingredient(cand_nom)
                    if not info_cand: continue
                    if not _check_compatibilitat(info_cand, perfil_context): continue
                    if not _es_candidat_coherent(info_orig, info_cand, _normalize_text(cat_macro)): continue
                    
                    candidats_map[c_norm] = cand_nom

            candidats_finals = list(candidats_map.values())
            if not candidats_finals:
                log_canvis.append(f"Avís: No s'ha trobat substitut segur per {ing_nom}")
                continue

            # Selecció Híbrida (similaritat + pairing)
            millor_substitut = None
            millor_score = -99.0
            justificacio = ""

            vec_orig = FG_WRAPPER.get_vector(ing_nom)
            vec_context = _calcular_vector_context(context_ingredients)

            use_vectors = vec_orig is not None or vec_context is not None

            if use_vectors:
                for cand in candidats_finals:
                    sim_self = FG_WRAPPER.similarity_with_vector(cand, vec_orig) if vec_orig is not None else None
                    sim_pair = FG_WRAPPER.similarity_with_vector(cand, vec_context) if vec_context is not None else None

                    score = 0.0
                    if sim_self is not None:
                        score += 0.65 * sim_self
                    if sim_pair is not None:
                        score += 0.35 * sim_pair

                    if score > millor_score:
                        millor_score = score
                        millor_substitut = cand

            if millor_substitut:
                justificacio = f"FlavorGraph (similitud+pairing {millor_score:.2f})"
            else:
                ordenats = _ordenar_candidats_per_afinitat(candidats_finals, kb, info_orig)
                if ordenats:
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
# 3. ADAPTACIÓ LATENT (Amb Check de Parelles)
# ---------------------------------------------------------------------

def _vector_mitja_plat(ingredients: List[str]) -> Optional[np.ndarray]:
    """Calcula el vector mitjà del plat per comparar-lo amb l'estil."""
    vectors = []
    for ing in ingredients:
        vec = FG_WRAPPER.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def _similitud_cosinus(vec_a: Optional[np.ndarray], vec_b: Optional[np.ndarray]) -> float:
    """Calcula la similitud cosinus entre dos vectors."""
    if vec_a is None or vec_b is None:
        return 0.0
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_a, vec_b) / (norm_a * norm_b))

def _check_role_compatibility(cat_orig: str, cat_cand: str) -> bool:
    """
    Llei de ferro: compatibilitat ontològica estricta entre categories.
    """
    if not cat_orig or not cat_cand:
        return False
    if cat_orig == "unknown" or cat_cand == "unknown":
        return False

    proteins = {"meat", "fish", "protein", "protein_animal", "protein_vegetal", "proteina_animal"}
    seasonings = {"herb", "spice", "condiment"}
    structural = {"grain", "pasta", "potato", "processed_cereal", "cereal_feculent", "vegetable"}
    dessert_ok = {"fruit", "sweet", "nut"}

    if cat_orig in proteins:
        return cat_cand in proteins
    if cat_orig in seasonings:
        return cat_cand in seasonings
    if cat_orig in structural:
        return cat_cand in structural
    if cat_orig == "fruit":
        return cat_cand in dessert_ok

    return cat_cand == cat_orig

# ---------------------------------------------------------------------
# 5. ADAPTACIÓ LATENT AGRESSIVA (Substitució Real)
# ---------------------------------------------------------------------

def _adaptar_latent_core(
    plat: Dict, 
    nom_estil: str, 
    kb: Any, 
    base_estils_latents: Dict, 
    intensitat: float,
    parelles_prohibides: Optional[Set[str]] = None,
    perfil_usuari: Optional[Dict] = None,
    ingredients_estil_usats: Optional[Set[str]] = None
):
    if not base_estils_latents: return plat
    
    # SETUP
    estil_data = base_estils_latents.get(nom_estil, {})
    ings_estil = estil_data.get('ingredients', [])
    vector_estil = FG_WRAPPER.compute_concept_vector(ings_estil)
    
    if vector_estil is None: return plat

    if ingredients_estil_usats is None:
        ingredients_estil_usats = set()

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients']) 
    log = []
    
    es_postres = "postres" in str(plat.get('curs')).lower()
    
    # AUGMENTEM LA TEMPERATURA per trobar coses diferents
    temperatura = min(0.99, 0.4 + intensitat)
    n_search = 30 + (len(ingredients_estil_usats) * 5)
    canvis_fets = 0

    # =================================================================
    # FASE A: SUBSTITUCIÓ (Prioritat: Canviar l'estructura)
    # =================================================================
    _debug_latent(f"\n[DEBUG LATENT] --- FASE A: SUBSTITUCIÓ ({nom_estil}) ---")
    
    for i, ing_original in enumerate(nou_plat['ingredients']):
        
        vec_orig = FG_WRAPPER.get_vector(ing_original)
        if vec_orig is None: continue
        
        vec_context = _calcular_vector_context(nou_plat['ingredients'], exclude_index=i)
        sim_style_orig = FG_WRAPPER.similarity_with_vector(ing_original, vector_estil) or 0.0
        
        # Si l'ingredient ja és perfecte (ex: Pinya en estil Tropical), no el toquem
        if sim_style_orig > 0.85: continue

        # Busquem MOLTS candidats per tenir varietat
        candidats = FG_WRAPPER.get_creative_candidates(
            ing_original, n=n_search, temperature=temperatura, style_vector=vector_estil
        )
        total_candidats = len(candidats)
        valids_memoria = sum(
            1 for cand, _ in candidats if _normalize_text(cand) not in ingredients_estil_usats
        )
        _debug_latent(f"  [DEBUG] Candidats Fase A: {total_candidats}, vàlids després de memòria: {valids_memoria}")
        
        info_orig = kb.get_info_ingredient(ing_original)
        cat_orig = str(info_orig.get('macro_category') or "unknown").lower()
        
        millor_cand = None
        millor_score_hibrid = -99.0
        
        # --- PESOS AGRESSIUS ---
        # Volem canviar! No ens importa tant que s'assembli a l'original (W_SELF baix)
        W_STYLE = 0.60 
        W_PAIRING = 0.35 
        W_SELF = 0.05    

        for cand, _ in candidats:
            cand_norm = _normalize_text(cand)
            if cand_norm == _normalize_text(ing_original): continue
            
            # CONTROL DE REPETICIÓ GLOBAL
            if cand_norm in ingredients_estil_usats: continue

            info_cand = kb.get_info_ingredient(cand)
            if not info_cand: continue

            context_noms = [nou_plat['ingredients'][k] for k in range(len(nou_plat['ingredients'])) if k != i]
            if parelles_prohibides and _check_parelles_prohibides(cand, context_noms, parelles_prohibides): continue
            if not _check_compatibilitat(info_cand, perfil_usuari): continue
            
            # FILTRES ONTOLÒGICS
            cat_cand = str(info_cand.get('macro_category') or "unknown").lower()
            
            if es_postres:
                if cat_cand not in {"sweet", "fruit", "dairy", "sweetener", "nut", "alcohol", "spice"}: continue
                # En postres relaxem lleugerament: fruita -> fruita, o bé fruita -> fruit sec/dolç si intensitat alta
                if cat_orig == "fruit" and cat_cand not in {"fruit", "nut"}:
                    if not (intensitat > 0.6 and cat_cand == "sweetener"):
                        continue
            else:
                if not _check_role_compatibility(cat_orig, cat_cand):
                    continue

            # PUNTUACIÓ
            sim_style = FG_WRAPPER.similarity_with_vector(cand, vector_estil) or 0.0
            sim_self = FG_WRAPPER.similarity_with_vector(cand, vec_orig) or 0.0
            
            sim_pairing = 0.0
            if vec_context is not None:
                vec_cand = FG_WRAPPER.get_vector(cand)
                if vec_cand is not None:
                    norm_c, norm_ctx = np.linalg.norm(vec_cand), np.linalg.norm(vec_context)
                    if norm_c > 0 and norm_ctx > 0:
                        sim_pairing = np.dot(vec_cand, vec_context) / (norm_c * norm_ctx)
            
            score = (W_STYLE * sim_style) + (W_PAIRING * sim_pairing) + (W_SELF * sim_self)
            
            # Score de l'ingredient actual
            score_current = (W_STYLE * sim_style_orig) + (W_PAIRING * sim_pairing) + (W_SELF * 1.0)
            
            # Si millora encara que sigui mínimament, o si la similitud amb l'estil és molt alta
            if score > score_current or (sim_style > sim_style_orig + 0.1):
                if score > millor_score_hibrid:
                    millor_score_hibrid = score
                    millor_cand = cand

        if millor_cand:
            _debug_latent(f"  ✅ SUBSTITUCIÓ: {ing_original} -> {millor_cand} (Score: {millor_score_hibrid:.2f})")
            nou_plat['ingredients'][i] = millor_cand
            msg_extra = " (evitant repeticions al menú)" if ingredients_estil_usats else ""
            log.append(f"Estil {nom_estil}: Substituït {ing_original} per {millor_cand}{msg_extra}")
            ingredients_estil_usats.add(_normalize_text(millor_cand))
            canvis_fets += 1

    # =================================================================
    # FASE B: INSERCIÓ (Només si el plat ha quedat pobre)
    # =================================================================
    
    vec_plat_nou = _vector_promig_plat(nou_plat['ingredients'])
    sim_global = 0.0
    if vec_plat_nou is not None and vector_estil is not None:
        v_p = _normalize_vector(vec_plat_nou)
        v_e = _normalize_vector(vector_estil)
        if v_p is not None and v_e is not None: sim_global = float(np.dot(v_p, v_e))
    
    # Si ja hem fet substitucions bones a la Fase A, sim_global haurà pujat i potser saltem la B
    TARGET_SIM = 0.82 if not es_postres else 0.88
    MAX_INGS = 9 

    _debug_latent(f"\n[DEBUG LATENT] --- FASE B: INSERCIÓ (Similitud: {sim_global:.2f} vs {TARGET_SIM}) ---")

    força_insercio = (canvis_fets == 0) or (sim_global < TARGET_SIM)

    if força_insercio and (len(nou_plat['ingredients']) < MAX_INGS) and (intensitat >= 0.3):
        rescue_mode = n_search >= 50
        # Passada 1: cerca dinàmica segons memòria
        representants = FG_WRAPPER.get_style_representatives(
            vector_estil, n=n_search, exclude_names=nou_plat['ingredients']
        )
        total_rep = len(representants)
        valids_memoria = sum(
            1 for cand, _ in representants if _normalize_text(cand) not in ingredients_estil_usats
        )
        _debug_latent(f"  [DEBUG] Candidats Fase B: {total_rep}, vàlids després de memòria: {valids_memoria}")

        millor_toc = None
        millor_pairing = -1.0
        millor_style = 0.0
        vec_context_final = _calcular_vector_context(nou_plat['ingredients']) 

        def _score_candidat(score_style, pairing, pes_style, pes_pairing):
            return (pes_style * score_style) + (pes_pairing * pairing)

        for cand, score_style in representants:
            cand_norm = _normalize_text(cand)
            if cand_norm in ingredients_estil_usats: continue # NO REPETIR

            info_cand = kb.get_info_ingredient(cand)
            if not info_cand: continue
            
            if parelles_prohibides and _check_parelles_prohibides(cand, nou_plat['ingredients'], parelles_prohibides): continue
            if not _check_compatibilitat(info_cand, perfil_usuari): continue
            
            cat_cand = str(info_cand.get('macro_category')).lower()
            if es_postres:
                if cat_cand not in {"sweet", "fruit", "dairy", "sweetener", "nut", "alcohol"}: continue

            pairing = 0.0
            if vec_context_final is not None:
                v_c = FG_WRAPPER.get_vector(cand)
                if v_c is not None:
                    norm_c, norm_ctx = np.linalg.norm(v_c), np.linalg.norm(vec_context_final)
                    if norm_c > 0 and norm_ctx > 0:
                        pairing = np.dot(v_c, vec_context_final) / (norm_c * norm_ctx)
            
            if rescue_mode:
                # Rescat: prioritzem només l'estil i ignorem el pairing
                score_final = score_style
                if score_final > 0.4 and score_final > millor_style:
                    millor_style = score_final
                    millor_toc = cand
            else:
                score_final = _score_candidat(score_style, pairing, 0.6, 0.4)
                if score_final > millor_pairing:
                    millor_pairing = score_final
                    millor_toc = cand

        # Passada 2 (rescat): si no hem trobat res, mirem molt més profund i prioritzem l'estil
        if millor_toc is None:
            representants = FG_WRAPPER.get_style_representatives(
                vector_estil, n=100, exclude_names=nou_plat['ingredients']
            )
            total_rep = len(representants)
            valids_memoria = sum(
                1 for cand, _ in representants if _normalize_text(cand) not in ingredients_estil_usats
            )
            _debug_latent(f"  [DEBUG] Rescat Fase B: {total_rep}, vàlids després de memòria: {valids_memoria}")

            millor_pairing = -1.0
            millor_style = 0.0
            for cand, score_style in representants:
                cand_norm = _normalize_text(cand)
                if cand_norm in ingredients_estil_usats: continue

                info_cand = kb.get_info_ingredient(cand)
                if not info_cand: continue
                
                if parelles_prohibides and _check_parelles_prohibides(cand, nou_plat['ingredients'], parelles_prohibides): continue
                if not _check_compatibilitat(info_cand, perfil_usuari): continue
                
                cat_cand = str(info_cand.get('macro_category')).lower()
                if es_postres:
                    if cat_cand not in {"sweet", "fruit", "dairy", "sweetener", "nut", "alcohol"}: continue

                pairing = 0.0
                if vec_context_final is not None:
                    v_c = FG_WRAPPER.get_vector(cand)
                    if v_c is not None:
                        norm_c, norm_ctx = np.linalg.norm(v_c), np.linalg.norm(vec_context_final)
                        if norm_c > 0 and norm_ctx > 0:
                            pairing = np.dot(v_c, vec_context_final) / (norm_c * norm_ctx)

                # Rescat fort: només estil, ignorem pairing i acceptem si supera llindar
                score_final = score_style
                if score_final > 0.4 and score_final > millor_style:
                    millor_style = score_final
                    millor_toc = cand
        
        if millor_toc:
            _debug_latent(f"  ✨ TOC MÀGIC AFEGIT: {millor_toc}")
            nou_plat['ingredients'].append(millor_toc)
            msg_extra = " (evitant repeticions al menú)" if ingredients_estil_usats else ""
            log.append(f"Estil {nom_estil}: Afegit {millor_toc} com a toc final{msg_extra}.")
            ingredients_estil_usats.add(_normalize_text(millor_toc))

    # =================================================================
    # FASE C: FALLBACK SIMBÒLIC (Només si no hi ha canvis)
    # =================================================================
    if not log:
        candidats_estil = list((base_estils_latents.get(nom_estil) or {}).get("ingredients", []))
        candidats_filtrats = []
        for cand in candidats_estil:
            cand_norm = _normalize_text(cand)
            if cand_norm in ingredients_estil_usats:
                continue
            info_cand = kb.get_info_ingredient(cand)
            if not info_cand:
                continue
            if not _check_compatibilitat(info_cand, perfil_usuari):
                continue
            cat_cand = _normalize_text(info_cand.get("macro_category") or "unknown")
            if es_postres and cat_cand not in {"fruit", "sweet", "dairy", "nut", "sweetener"}:
                continue
            candidats_filtrats.append((cand, cat_cand))

        vec_context_final = _calcular_vector_context(nou_plat.get("ingredients", []))
        puntuats = []
        for cand, cat_cand in candidats_filtrats:
            sim_pair = 0.0
            if vec_context_final is not None:
                vec_c = FG_WRAPPER.get_vector(cand)
                if vec_c is not None:
                    norm_c, norm_ctx = np.linalg.norm(vec_c), np.linalg.norm(vec_context_final)
                    if norm_c > 0 and norm_ctx > 0:
                        sim_pair = np.dot(vec_c, vec_context_final) / (norm_c * norm_ctx)
            puntuats.append((cand, cat_cand, sim_pair))

        puntuats.sort(key=lambda x: x[2], reverse=True)
        if puntuats:
            _debug_latent(f"  [DEBUG] Fase C: {len(puntuats)} candidats simbòlics vàlids, top pairing {puntuats[0][2]:.2f}")

        boring = {
            "oil", "sunflower oil", "sugar", "water", "vinegar", "cream", "milk"
        }
        accio = None

        # Estratègia 1: Substitució genèrica amb pairing vectorial
        if puntuats:
            for i, ing in enumerate(nou_plat.get("ingredients", [])):
                if _normalize_text(ing) not in boring:
                    continue
                info_ing = kb.get_info_ingredient(ing)
                if not info_ing:
                    continue
                cat_ing = _normalize_text(info_ing.get("macro_category") or "unknown")
                for cand, cat_cand, sim_pair in puntuats:
                    if cat_cand == cat_ing:
                        nou_plat["ingredients"][i] = cand
                        accio = f"Substituït {ing} per {cand} (Fallback Simbòlic + Pairing Vectorial)"
                        ingredients_estil_usats.add(_normalize_text(cand))
                        _debug_latent(f"  [DEBUG] Fase C: swap genèric {ing} -> {cand} (pairing {sim_pair:.2f})")
                        break
                if accio:
                    break

        # Estratègia 2: Inserció forçada amb pairing vectorial
        if not accio and puntuats:
            cand, _, sim_pair = puntuats[0]
            nou_plat["ingredients"].append(cand)
            accio = f"Afegit {cand} (Fallback Simbòlic + Pairing Vectorial)"
            ingredients_estil_usats.add(_normalize_text(cand))
            _debug_latent(f"  [DEBUG] Fase C: inserció {cand} (pairing {sim_pair:.2f})")

        if accio:
            log.append(f"Estil {nom_estil}: {accio}")

    nou_plat['log_transformacio'] = log
    return nou_plat
