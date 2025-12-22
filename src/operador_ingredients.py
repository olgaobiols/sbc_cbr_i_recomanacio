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
    alergies_usuari = {_normalize_text(a) for a in perfil_usuari.get('alergies', []) if a}
    if alergies_usuari:
        alergens_ing = {_normalize_text(p) for p in str(ingredient_info.get('allergens', '')).split('|') if p}
        familia_ing = _normalize_text(ingredient_info.get('family'))
        if alergies_usuari.intersection(alergens_ing) or (familia_ing and familia_ing in alergies_usuari):
            return False

    # Validació de dieta
    dieta_usuari = _normalize_text(perfil_usuari.get('dieta'))
    if dieta_usuari:
        dietes_ing = {_normalize_text(p) for p in str(ingredient_info.get('allowed_diets', '')).split('|') if p}
        if dieta_usuari and dieta_usuari not in dietes_ing:
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
    for tag in str(info_prohibit.get('allergens', '')).split('|'):
        if t := _normalize_text(tag): alergies.add(t)
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
    return prohibits

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
# LÒGICA ONTOLÒGICA (Simbolisme)
# ---------------------------------------------------------------------
_PROTEINA_CATEGORIES = {"protein_animal", "protein_vegetal", "fish_white", "fish_oily", "proteina_animal"}
_DAIRY_CATEGORIES = {"dairy", "dairy_cheese", "dairy_cream", "lacti"}
_GRAIN_CATEGORIES = {"grain", "processed_cereal", "cereal_feculent"}
_EGG_CATEGORIES = {"egg"}
_MAIN_ROLES = {"main", "principal", "main course"}
_SIDE_ROLES = {"side", "base"}


# CONSTANTS ONTOLÒGIQUES 
# MILLORRRRRRRRRRRRRRRRRRRRAAAAAAAAAAAAAAAAAAR



def _get_candidats_per_categoria(categoria: str, kb: Any) -> List[str]:
    cat_norm = _normalize_text(categoria)
    return [
        info['ingredient_name'] for info in kb.ingredients.values()
        if _normalize_text(info.get('macro_category') or info.get('categoria_macro')) == cat_norm
    ]

def _categoria_fallbacks(categoria_norm: str, perfil_usuari: Optional[Dict]) -> List[str]:
    """Defineix substitucions ontològiques segures quan la categoria original està prohibida."""
    if not perfil_usuari: return []
    dieta = _normalize_text(perfil_usuari.get('dieta'))
    fallbacks = []
    
    if dieta in {"vegan", "vegetarian"}:
        if categoria_norm in _PROTEINA_CATEGORIES:
            fallbacks.extend(["protein_vegetal", "plant_vegetal", "grain"])
        elif categoria_norm in _DAIRY_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "fat", "other"])
        elif categoria_norm in _EGG_CATEGORIES:
            fallbacks.extend(["plant_vegetal", "grain"])
            
    return [_normalize_text(c) for c in fallbacks]

def _es_candidat_coherent(info_orig: Dict, info_cand: Dict, cat_orig_norm: str) -> bool:
    """Verifica que el substitut mantingui el rol estructural al plat (Coherència)."""
    r_orig = _normalize_text(info_orig.get('typical_role'))
    r_cand = _normalize_text(info_cand.get('typical_role'))
    
    if cat_orig_norm in _PROTEINA_CATEGORIES or r_orig in _MAIN_ROLES:
        return r_cand in _MAIN_ROLES
    if r_orig in _SIDE_ROLES and r_cand:
        return r_cand in _SIDE_ROLES or r_cand in _MAIN_ROLES
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
                                     ingredients_usats: Optional[Set[str]] = None, parelles_prohibides: Optional[Set[str]] = None) -> Dict[str, Any]:
    """
    Substitueix ingredients que violen restriccions dures.
    Utilitza una estratègia híbrida: cerca candidats via ontologia i selecciona el millor via FlavorGraph.
    """
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
            
            # Context i candidats potencials
            perfil_context = _build_perfil_context(perfil_usuari, info_orig)
            cat_macro = info_orig.get('macro_category') or info_orig.get('categoria_macro')
            if not cat_macro: continue
            
            cats_candidats = [_normalize_text(cat_macro)]
            cats_candidats.extend(_categoria_fallbacks(_normalize_text(cat_macro), perfil_usuari))
            
            candidats_map = {}
            context_ingredients = [ing for k, ing in enumerate(nou_plat['ingredients']) if k != i]

            # Filtratge de candidats
            for cat in cats_candidats:
                for cand_nom in _get_candidats_per_categoria(cat, kb):
                    c_norm = _normalize_text(cand_nom)
                    
                    if c_norm == ing_norm or c_norm in prohibits_norm: continue
                    if whitelist_norm and c_norm not in whitelist_norm: continue
                    if parelles_prohibides and _check_parelles_prohibides(cand_nom, context_ingredients, parelles_prohibides): continue

                    info_cand = kb.get_info_ingredient(cand_nom)
                    if info_cand and _check_compatibilitat(info_cand, perfil_context) and _es_candidat_coherent(info_orig, info_cand, _normalize_text(cat_macro)):
                        candidats_map[c_norm] = cand_nom

            candidats_finals = list(candidats_map.values())
            if not candidats_finals:
                log_canvis.append(f"Avís: No s'ha trobat substitut segur per {ing_nom}")
                continue

            # Selecció Híbrida: Vectors (FlavorGraph) > Ontologia
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

                    score = (0.65 * (sim_self or 0)) + (0.35 * (sim_pair or 0))
                    if score > millor_score:
                        millor_score = score
                        millor_substitut = cand

            if millor_substitut:
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
        cat_orig = str(info_orig.get('macro_category') or "unknown").lower()
        
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
            
            cat_cand = str(info_cand.get('macro_category') or "unknown").lower()
            if es_postres:
                if cat_cand not in {"sweet", "fruit", "dairy", "sweetener", "nut", "alcohol", "spice"}: continue
                # Relaxació per postres (permet canviar fruita per dolç si intensitat és alta)
                if cat_orig == "fruit" and cat_cand not in {"fruit", "nut"} and not (intensitat > 0.6 and cat_cand == "sweetener"): continue
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
                
                if es_postres and str(info_cand.get('macro_category')).lower() not in {"sweet", "fruit", "dairy", "sweetener", "nut", "alcohol"}:
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
                cat = _normalize_text(info.get("macro_category") or "unknown")
                if es_postres and cat not in {"fruit", "sweet", "dairy", "nut", "sweetener"}: continue
                
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
                    cat_ing = _normalize_text(info_ing.get("macro_category") or "unknown") if info_ing else ""
                    
                    for cand, cat_cand, sim_pair in puntuats:
                        if cat_cand == cat_ing: # Match categòric
                            nou_plat["ingredients"][i] = cand
                            accio = f"Substituït {ing} per {cand} (Fallback Simbòlic)"
                            ingredients_estil_usats.add(_normalize_text(cand))
                            break
                if accio: break

        # Inserció directa si falla la substitució
        if not accio and puntuats:
            cand = puntuats[0][0]
            nou_plat["ingredients"].append(cand)
            accio = f"Afegit {cand} (Fallback Simbòlic)"
            ingredients_estil_usats.add(_normalize_text(cand))

        if accio: log.append(f"Estil {nom_estil}: {accio}")

    nou_plat['log_transformacio'] = log
    return nou_plat