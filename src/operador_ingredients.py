import random
import unicodedata
from typing import List, Dict, Set, Any, Optional
import numpy as np

from flavorgraph_embeddings import FlavorGraphWrapper

FG_WRAPPER = FlavorGraphWrapper()

# ---------------------------------------------------------------------
# 1. FUNCIONS AUXILIARS (Gestió de Dades i Compatibilitat)
# ---------------------------------------------------------------------

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
    parelles_prohibides: Optional[Set[str]] = None # <--- NOU
) -> Dict[str, Any]:
    
    if mode == "latent":
        return _adaptar_latent_core(plat, target, kb, estils_latents, intensitat, parelles_prohibides)
    else:
        # Wrapper per compatibilitat si algú crida amb mode='restriccio' directament
        pass
    return plat

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

            # Selecció Híbrida
            millor_substitut = None
            justificacio = ""
            
            vec_orig = FG_WRAPPER.get_vector(ing_nom)
            if vec_orig is not None:
                fg_ranking = FG_WRAPPER._find_nearest_to_vector(
                    vec_orig, n=10, exclude_names=list(prohibits_norm | {ing_norm})
                )
                for nom_fg, score in fg_ranking:
                    if _normalize_text(nom_fg) in candidats_map:
                        millor_substitut = candidats_map[_normalize_text(nom_fg)]
                        justificacio = f"FlavorGraph (similitud {score:.2f})"
                        break
            
            if not millor_substitut:
                ordenats = _ordenar_candidats_per_afinitat(candidats_finals, kb, info_orig)
                if ordenats:
                    millor_substitut = ordenats[0]
                    justificacio = "Ontologia (família/rol)"
            
            if not millor_substitut:
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

def _adaptar_latent_core(
    plat: Dict, 
    nom_estil: str, 
    kb: Any, 
    base_estils_latents: Dict, 
    intensitat: float,
    parelles_prohibides: Optional[Set[str]] = None # <--- NOU
):
    if not base_estils_latents: return plat
    
    estil_data = base_estils_latents.get(nom_estil, {})
    ings_estil = estil_data.get('ingredients', [])
    vector_estil = FG_WRAPPER.compute_concept_vector(ings_estil)
    if vector_estil is None: return plat

    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients'])
    log = []
    
    temperatura = min(0.9, 0.2 + intensitat)
    es_postres = "postres" in str(plat.get('curs')).lower()
    
    for i, ing_nom in enumerate(nou_plat['ingredients']):
        sim_actual = FG_WRAPPER.similarity_with_vector(ing_nom, vector_estil)
        if sim_actual and sim_actual > (0.8 if es_postres else 0.7):
            continue 
            
        candidats = FG_WRAPPER.get_creative_candidates(
            ing_nom, n=8, temperature=temperatura, style_vector=vector_estil
        )
        
        # Context per check de parelles
        context_ingredients = [nou_plat['ingredients'][k] for k in range(len(nou_plat['ingredients'])) if k != i]

        info_orig = kb.get_info_ingredient(ing_nom)
        cat_orig = info_orig.get('macro_category') if info_orig else None
        
        millor = None
        millor_gain = 0.05 
        
        for cand, _ in candidats:
            # 1. Check Parelles (Canal A/B)
            if parelles_prohibides and _check_parelles_prohibides(cand, context_ingredients, parelles_prohibides):
                continue
            
            info_cand = kb.get_info_ingredient(cand)
            if not info_cand: continue
            
            cat_cand = info_cand.get('macro_category')
            if es_postres:
                if str(cat_cand).lower() not in {"sweet", "fruit", "dairy", "sweetener"}: continue
            elif cat_orig and cat_cand != cat_orig and intensitat < 0.7:
                 continue 
                 
            sim_cand = FG_WRAPPER.similarity_with_vector(cand, vector_estil)
            if sim_cand and (sim_cand - (sim_actual or 0) > millor_gain):
                millor = cand
                millor_gain = sim_cand - (sim_actual or 0)
        
        if millor:
            nou_plat['ingredients'][i] = millor
            log.append(f"Estil {nom_estil}: {ing_nom} -> {millor} (+{millor_gain:.2f} afinitat)")

    nou_plat['log_transformacio'] = log
    return nou_plat