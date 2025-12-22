import json
import os
from typing import List, Set, Dict, Any, Optional, Tuple
import numpy as np

from estructura_cas import DescripcioProblema
from Retriever import Retriever
from knowledge_base import KnowledgeBase
from gestor_feedback import GestorRevise
from operadors_transformacio_realista import (
    substituir_ingredient, 
    triar_tecniques_2_operadors_per_menu, 
    genera_descripcio_llm, 
    construir_prompt_imatge_menu, 
    genera_imatge_menu_hf,
    llista_tecniques_applicables_per_ingredient,
)

from operador_ingredients import (
    FG_WRAPPER,
    ingredients_incompatibles,
    substituir_ingredients_prohibits,
)

from operadors_begudes import recomana_beguda_per_plat, get_ingredient_principal, passa_filtre_dur, score_beguda_per_plat

# =========================
#   INICIALITZACIÓ GLOBAL
# =========================

# 1. Carreguem el cervell (Ontologia + Estils)
kb = KnowledgeBase()

print("\nEstils latents disponibles a la carta:")
print(" - " + "\n - ".join(sorted(kb.estils_latents.keys())))

# Com que els operadors antics esperen llistes de diccionaris, 
# creem referències compatibles per no trencar res:
base_ingredients_list = list(kb.ingredients.values())

# =========================
#   FUNCIONS AUXILIARS CLI
# =========================

def input_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default

def input_choice(prompt, options, default, indiferent_value=None):
    opts = list(options)
    if "indiferent" not in opts:
        opts.append("indiferent")
    opts_txt = "/".join(opts)
    while True:
        txt = input(f"{prompt} ({opts_txt}) [{default}]: ").strip().lower()
        if not txt:
            return default
        if txt == "indiferent":
            return indiferent_value if indiferent_value is not None else "indiferent"
        if txt in opts:
            return txt
        print(f"  Valor no vàlid. Opcions: {opts_txt}")

def input_int_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt: return default
    try:
        return int(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return default

def input_float_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt: return float(default)
    try:
        return float(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return float(default)

def parse_list_input(txt: str) -> Set[str]:
    """Converteix 'gluten, vegan' en {'gluten', 'vegan'} normalitzat."""
    if not txt: return set()
    return {x.strip().lower() for x in txt.split(",") if x.strip()}

def _normalize_item(value: str) -> str:
    if not value:
        return ""
    text = str(value).strip().lower()
    return " ".join(text.replace("-", " ").replace("_", " ").split())

def _format_list(items: List[str]) -> str:
    if not items:
        return "—"
    return ", ".join(items)

def _format_pairs(pairs: List[str]) -> str:
    if not pairs:
        return "—"
    pretty = []
    for raw in pairs:
        if "|" in raw:
            a, b = [p.strip() for p in raw.split("|", 1)]
            pretty.append(f"{a} + {b}")
        else:
            pretty.append(raw)
    return ", ".join(pretty)

def _parse_pairs_input(txt: str) -> List[str]:
    if not txt:
        return []
    out = []
    for part in txt.split(","):
        token = part.strip()
        if not token:
            continue
        if "+" in token:
            a, b = [p.strip() for p in token.split("+", 1)]
        elif "|" in token:
            a, b = [p.strip() for p in token.split("|", 1)]
        else:
            continue
        if a and b:
            out.append("|".join(sorted([_normalize_item(a), _normalize_item(b)])))
    return out

def _normalize_pair_key(raw: str) -> str:
    if not raw:
        return ""
    if "|" in raw:
        a, b = raw.split("|", 1)
    elif "+" in raw:
        a, b = raw.split("+", 1)
    else:
        return ""
    a_norm = _normalize_item(a)
    b_norm = _normalize_item(b)
    if not a_norm or not b_norm:
        return ""
    return "|".join(sorted([a_norm, b_norm]))

def _collect_vetats(perfil: Dict[str, Any], learned_rules: Dict[str, Any]) -> tuple[Set[str], Set[str]]:
    user_ings = {_normalize_item(x) for x in (perfil.get("rejected_ingredients", []) or []) if x}
    user_pairs = {_normalize_pair_key(x) for x in (perfil.get("rejected_pairs", []) or []) if x}
    global_rules = learned_rules.get("global_rules", {}) if isinstance(learned_rules, dict) else {}
    glob_ings = {_normalize_item(x) for x in (global_rules.get("ingredients", []) or []) if x}
    glob_pairs = {_normalize_pair_key(x) for x in (global_rules.get("pairs", []) or []) if x}
    user_pairs.discard("")
    glob_pairs.discard("")
    return user_ings | glob_ings, user_pairs | glob_pairs

def _plat_te_ingredient_vetat(ingredients: List[str], vetats: Set[str]) -> bool:
    if not vetats:
        return False
    for ing in ingredients:
        if _normalize_item(ing) in vetats:
            return True
    return False

def _plat_te_parella_vetada(ingredients: List[str], parelles_vetades: Set[str]) -> bool:
    if not parelles_vetades:
        return False
    norm_ings = [_normalize_item(i) for i in ingredients if i]
    for i in range(len(norm_ings)):
        for j in range(i + 1, len(norm_ings)):
            key = "|".join(sorted([norm_ings[i], norm_ings[j]]))
            if key in parelles_vetades:
                return True
    return False

def _parelles_detectades(ingredients: List[str], parelles_vetades: Set[str]) -> List[str]:
    if not parelles_vetades:
        return []
    norm_ings = [_normalize_item(i) for i in ingredients if i]
    found = []
    for i in range(len(norm_ings)):
        for j in range(i + 1, len(norm_ings)):
            key = "|".join(sorted([norm_ings[i], norm_ings[j]]))
            if key in parelles_vetades:
                found.append(key)
    return found

def _trobar_plat_alternatiu(
    curs: str,
    resultats: list,
    vetats: Set[str],
    parelles_vetades: Set[str],
    case_id_actual: Any,
) -> Optional[dict]:
    curs_norm = str(curs).lower()
    for r in resultats:
        cas = r.get("cas") or {}
        if cas.get("id_cas") == case_id_actual:
            continue
        plats = cas.get("solucio", {}).get("plats", []) or []
        for p in plats:
            if str(p.get("curs", "")).lower() != curs_norm:
                continue
            ings = list(p.get("ingredients", []) or [])
            if _plat_te_ingredient_vetat(ings, vetats):
                continue
            if _plat_te_parella_vetada(ings, parelles_vetades):
                continue
            return p.copy()
    return None

def _check_compatibilitat_local(ingredient_info: Dict, perfil_usuari: Optional[Dict]) -> bool:
    if not ingredient_info:
        return False
    if not perfil_usuari:
        return True
    alergies = {_normalize_item(a) for a in perfil_usuari.get("alergies", []) if a}
    if alergies:
        alergens_ing = {_normalize_item(p) for p in str(ingredient_info.get("allergens", "")).split("|") if p}
        familia_ing = _normalize_item(ingredient_info.get("family"))
        if alergies.intersection(alergens_ing) or (familia_ing and familia_ing in alergies):
            return False
    dieta = _normalize_item(perfil_usuari.get("dieta"))
    if dieta:
        dietes_ing = {_normalize_item(p) for p in str(ingredient_info.get("allowed_diets", "")).split("|") if p}
        if dieta and dieta not in dietes_ing:
            return False
    return True

def _try_add_preferred_touch(
    plats: List[dict],
    preferits: List[str],
    perfil_usuari: Optional[Dict],
    vetats: Set[str],
    parelles_vetades: Set[str],
) -> None:
    if not preferits:
        return
    best = None
    best_score = 0.0
    threshold = 0.35

    for pref in preferits:
        pref_norm = _normalize_item(pref)
        if not pref_norm or pref_norm in vetats:
            continue
        info = kb.get_info_ingredient(pref_norm)
        if not _check_compatibilitat_local(info, perfil_usuari):
            continue
        pref_name = info.get("ingredient_name") or pref_norm
        for plat in plats:
            ings = list(plat.get("ingredients", []) or [])
            if pref_norm in {_normalize_item(i) for i in ings}:
                continue
            if parelles_vetades and _plat_te_parella_vetada(ings + [pref_name], parelles_vetades):
                continue
            vec_plat = _vector_mitja(ings)
            if vec_plat is None:
                continue
            score = FG_WRAPPER.similarity_with_vector(pref_name, vec_plat)
            if score is None or score < threshold:
                continue
            if score > best_score:
                best_score = score
                best = (plat, pref_name, score)

    if best:
        plat, pref_norm, score = best
        plat.setdefault("ingredients", []).append(pref_norm)
        logs = list(plat.get("log_transformacio", []) or [])
        logs.append(f"Preferència: Afegit {pref_norm} com a toc (afinitat {score:.2f})")
        plat["log_transformacio"] = logs

def _print_section(title: str) -> None:
    print("\n" + "—" * 50)
    print(title)
    print("—" * 50)

EU_ALLERGENS = [
    ("gluten", "Cereals amb gluten"),
    ("crustaceans", "Crustacis"),
    ("egg", "Ous"),
    ("fish", "Peix"),
    ("peanuts", "Cacauets"),
    ("soybeans", "Soja"),
    ("milk", "Llet"),
    ("nuts", "Fruits secs"),
    ("celery", "Api"),
    ("mustard", "Mostassa"),
    ("sesame", "Sesam"),
    ("sulfites", "Sulfitos"),
    ("lupin", "Lupi"),
    ("molluscs", "Moluscs"),
]

def seleccionar_alergens() -> List[str]:
    print("\nALLERGENS UE (14)")
    for i, (key, label) in enumerate(EU_ALLERGENS, start=1):
        print(f"  {i:>2}. {label} [{key}]")
    txt = input_default("Quins al·lèrgens hem d'evitar? (números separats per comes, Enter si cap)", "")
    if not txt:
        return []
    seleccionats = []
    for part in txt.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(EU_ALLERGENS):
                seleccionats.append(EU_ALLERGENS[idx - 1][0])
            continue
        for key, _ in EU_ALLERGENS:
            if token == key:
                seleccionats.append(key)
                break
    vistos = set()
    resultat = []
    for key in seleccionats:
        if key not in vistos:
            resultat.append(key)
            vistos.add(key)
    return resultat

PATH_USER_PROFILES = "data/user_profiles.json"
PATH_LEARNED_RULES = "data/learned_rules.json"

def _load_user_profiles(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _load_learned_rules(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_user_profiles(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)

def _get_user_profile(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    uid = str(user_id)
    perfil = data.get(uid)
    if perfil is None:
        perfil = {}
        data[uid] = perfil
    return perfil

def _store_user_alergies(data: Dict[str, Any], user_id: str, alergies: List[str]) -> None:
    perfil = _get_user_profile(data, user_id)
    perfil["alergies"] = list(alergies)


def _dedup_preserve_order(items: List[str]) -> List[str]:
    vistos = set()
    resultat = []
    for x in items:
        if x not in vistos:
            resultat.append(x)
            vistos.add(x)
    return resultat

def _vector_mitja(ingredients: List[str]) -> Optional[np.ndarray]:
    vectors = []
    for ing in ingredients:
        vec = FG_WRAPPER.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def _similitud_plat_estil(ingredients: List[str], estils_latents: Dict, nom_estil: str) -> float:
    estil_data = estils_latents.get(nom_estil, {})
    ings_estil = estil_data.get("ingredients", [])
    vec_estil = FG_WRAPPER.compute_concept_vector(ings_estil)
    vec_plat = _vector_mitja(ingredients)
    if vec_estil is None or vec_plat is None:
        return 0.0
    norm_a = np.linalg.norm(vec_plat)
    norm_b = np.linalg.norm(vec_estil)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_plat, vec_estil) / (norm_a * norm_b))

def imprimir_resum_adaptacio(
    etiqueta_plat: str,
    plat: dict,
    ingredients_abans: List[str],
    nom_estil: str,
    intensitat: float,
    kb: Any,
    ingredients_original: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    """
    Mostra un resum compacte de l'adaptació i retorna un resum per al final.
    """
    ingredients_despres = plat.get("ingredients", []) or []
    ingredients_despres_unics = _dedup_preserve_order(ingredients_despres)

    set_abans = set(ingredients_abans)
    set_despres = set(ingredients_despres_unics)
    afegits = [x for x in ingredients_despres_unics if x not in set_abans]
    trets = [x for x in ingredients_abans if x not in set_despres]

    logs = plat.get("log_transformacio", []) or []
    log_text = " ".join(logs).lower()

    sim_abans = _similitud_plat_estil(ingredients_abans, kb.estils_latents, nom_estil)
    sim_despres = _similitud_plat_estil(ingredients_despres_unics, kb.estils_latents, nom_estil)
    delta = sim_despres - sim_abans

    print(f"\n[{etiqueta_plat}] {plat.get('nom','—')}")
    print(f"- Ingredients: {len(ingredients_abans)} -> {len(ingredients_despres_unics)}")
    if ingredients_original and ingredients_original != ingredients_abans:
        print(f"- Abans (original): {', '.join(ingredients_original) if ingredients_original else '—'}")
    print(f"- Abans (post-restriccions): {', '.join(ingredients_abans) if ingredients_abans else '—'}")
    print(f"- Despres: {', '.join(ingredients_despres_unics) if ingredients_despres_unics else '—'}")
    condiment = plat.get("condiment")
    if condiment:
        print(f"- Condiment: {condiment}")

    duplicat_proposat = None
    if "afegit" in log_text:
        for ing in ingredients_abans:
            if f"afegit {ing}" in log_text:
                duplicat_proposat = ing
                break

    canvi_text = "CAP CANVI"
    motiu = "ja és coherent amb l'estil i no cal tocar res"
    if trets and afegits:
        canvi_text = "SUBSTITUCIO"
        motiu = "substitució coherent amb l'estil del menú"
    elif afegits:
        canvi_text = "INSERCIO"
        if "fallback simbòlic" in log_text:
            motiu = "petit toc simbòlic per reforçar l'estil"
        else:
            motiu = "afegit un toc que harmonitza amb l'estil"
    elif duplicat_proposat:
        print(f"- Nota: l'ingredient ja hi era ({duplicat_proposat}) i no l'he duplicat")
        canvi_text = "CAP CANVI"
        motiu = "toc ja present al plat"

    if afegits:
        print(f"- Canvi: +{', '.join(afegits)}")
    elif trets:
        print(f"- Canvi: -{', '.join(trets)}")

    print(f"- Afinitat amb l'estil: {sim_abans:.2f} -> {sim_despres:.2f} ({delta:+.2f})")
    print(f"- Decisió: {canvi_text}")
    print(f"- Motiu: {motiu}")
    print("")

    if canvi_text == "INSERCIO" and afegits:
        resum = f"+{', '.join(afegits)}"
    elif canvi_text == "SUBSTITUCIO" and afegits and trets:
        resum = f"{trets[0]} -> {afegits[0]}"
    else:
        resum = "cap canvi"

    if condiment:
        if resum == "cap canvi":
            resum = f"condiment: {condiment}"
        else:
            resum = f"{resum} | condiment: {condiment}"

    return afegits, resum


def imprimir_resum_plat_net(
    etiqueta_plat: str,
    plat: dict,
    transf: list[dict],
    estil_cultural: str | None,
    estil_alta: str | None,
):
    nom_plat = plat.get("nom", "—")
    ings = ", ".join(plat.get("ingredients", []) or [])
    print(f"\n{etiqueta_plat}: {nom_plat}")
    print(f"  Ingredients: {ings if ings else '—'}")

    if not transf:
        print("  Tècniques: cap")
        return

    # intentem etiquetar cada tècnica amb l'estil d'origen (cultural vs alta)
    tecs_culturals = set()
    if estil_cultural:
        row_c = kb.estils.get(estil_cultural, {}) or {}
        tecs_culturals = {t.strip() for t in (row_c.get("tecnniques_clau", "") or "").split("|") if t.strip()}

    for i, t in enumerate(transf, 1):
        nom = (t.get("nom") or "").strip()
        disp = (t.get("display") or nom or "tecnica").strip()
        obj = (t.get("objectiu_ingredient") or "—").strip()

        origen = "mixt"
        if estil_cultural and not estil_alta:
            origen = "cultural"
        elif estil_alta and not estil_cultural:
            origen = "alta"
        else:
            # mixt: intentem detectar si és cultural
            origen = "alta"
            if nom in tecs_culturals:
                origen = "cultural"


        print(f"  - ({origen}) {disp} → {obj}")

def imprimir_menu_final_net(
    plat1, transf_1,
    plat2, transf_2,
    postres, transf_post,
    estil_cultural: str | None,
    estil_alta: str | None,
):
    _print_section("Menú adaptat — ingredients i tècniques")

    imprimir_resum_plat_net("PRIMER", plat1, transf_1, estil_cultural, estil_alta)
    imprimir_resum_plat_net("SEGON",  plat2, transf_2, estil_cultural, estil_alta)
    imprimir_resum_plat_net("POSTRES", postres, transf_post, estil_cultural, estil_alta)




def imprimir_casos(candidats, top_k=5):
    """Mostra els resultats del Retriever de forma ordenada."""
    if not candidats:
        print("\nNo he trobat cap cas prou similar.")
        return

    print(f"\nHe trobat {len(candidats)} opcions. Et mostro les {min(top_k, len(candidats))} millors:")

    for i, c in enumerate(candidats[:top_k], start=1):
        cas = c["cas"]
        score = c["score_final"]
        sol = cas.get("solucio", {})
        pr = cas.get("problema", {})

        etiqueta = "⭐ Recomanat" if i == 1 else f"Opció {i}"
        
        # Dades clau
        event = pr.get("tipus_esdeveniment", "?")
        restr = pr.get("restriccions", [])
        if restr:
            str_restr = f" (Restr: {', '.join(restr)})"
        else:
            str_restr = ""

        # Menú resumit
        plats = sol.get("plats", []) or []

        def _nom_plat(curs: str) -> str:
            curs = str(curs).lower()
            for p in plats:
                if str(p.get("curs", "")).lower() == curs:
                    return p.get("nom", "—")
            return "—"

        p1 = _nom_plat("primer")
        p2 = _nom_plat("segon")
        p3 = _nom_plat("postres")

        print(f"\n{etiqueta} — Afinitat {score:.1%} (ID: {cas.get('id_cas', '?')})")
        print(f"   Context: {event} | {pr.get('temporada','?')} | {pr.get('servei','?')}{str_restr}")
        print(f"   Menú: 1) {p1}  2) {p2}  3) {p3}")

        # Detall complet dels plats
        def _ordre_plat(p: dict) -> int:
            curs = str(p.get("curs", "")).lower()
            ordre = {"primer": 0, "segon": 1, "postres": 2}
            return ordre.get(curs, 99)

        plats_ordenats = sorted(plats, key=_ordre_plat)
        print("   Plats detallats:")
        for p in plats_ordenats:
            curs = p.get("curs", "?")
            nom = p.get("nom", "—")
            ings = ", ".join(p.get("ingredients", []) or [])
            rols = ", ".join(p.get("rols_principals", []) or [])
            preu = p.get("preu", None)
            print(f"     - {curs}: {nom}")
            print(f"       Ingredients: {ings if ings else '—'}")
            if rols:
                print(f"       Rols principals: {rols}")
            if preu not in ("", None):
                print(f"       Preu: {preu}")

        # Detall intern eliminat per simplicitat en l'experiència

def imprimir_menu_final(
    plat1, transf_1, info_llm_1, beguda1, score1,
    plat2, transf_2, info_llm_2, beguda2, score2,
    postres, transf_post, info_llm_post, beguda_postres, score_postres
):
    _print_section("🍽️  Proposta final de menú")

    for etiqueta, plat, info_llm, beguda, score in [
        ("PRIMER PLAT", plat1, info_llm_1, beguda1, score1),
        ("SEGON PLAT",  plat2, info_llm_2, beguda2, score2),
        ("POSTRES",     postres, info_llm_post, beguda_postres, score_postres),
    ]:
        nom = info_llm.get("nom_nou", plat.get("nom", "—")) if info_llm else plat.get("nom", "—")
        desc = info_llm.get("descripcio_carta", "") if info_llm else "Plat clàssic."
        
        print(f"\n🍽️  {etiqueta}: {nom}")
        ings = ", ".join(plat.get("ingredients", []))
        print(f"   Ingredients clau: {ings if ings else '—'}")
        if plat.get("condiment"):
            print(f"   Condiment: {plat.get('condiment')}")
        if desc:
            print(f"   Descripció: {desc}")
        
        if beguda is None:
            print("   🍷 Maridatge: no he trobat una opció adequada.")
        else:
            print(f"   🍷 Maridatge: {beguda.get('nom', '—')} (afinitat {score:.2f})")

        # Si hi ha logs de canvis, els mostrem (Explicabilitat XCBR)
        logs = plat.get("log_transformacio", [])
        if logs:
            print("   Ajustos del xef:")
            for log in logs:
                print(f"      - {log}")


def debug_kb_match(plat, kb, etiqueta=""):
    print(f"\n[KB CHECK] {etiqueta} — {plat.get('nom','—')}")
    for ing in plat.get("ingredients", []):
        ok = kb.get_info_ingredient(ing) is not None
        print(f"  - {ing}  ->  {'✅' if ok else '❌ NO A KB'}")



# =========================
#   MAIN INTERACTIU
# =========================

def main():
    print("==================================================")
    print("   🍷 Maître Digital — Recomanador de Menús 3.0")
    print("==================================================\n")

    user_id_raw = input_default("Com et puc anomenar? (per guardar preferències)", "guest").strip()
    user_id = (user_id_raw or "guest").lower()
    user_profiles = _load_user_profiles(PATH_USER_PROFILES)
    learned_rules = _load_learned_rules(PATH_LEARNED_RULES)
    perfil_guardat = user_profiles.get(str(user_id))
    if perfil_guardat is None:
        existing_key = next(
            (k for k in user_profiles.keys() if str(k).lower() == user_id),
            None
        )
        if existing_key is not None:
            perfil_guardat = user_profiles.get(existing_key)
    if perfil_guardat is None:
        perfil_guardat = {}
    if not isinstance(perfil_guardat, dict):
        perfil_guardat = {}
    display_name = perfil_guardat.get("display_name") or user_id_raw or user_id

    stored_alergies = list(perfil_guardat.get("alergies", []) or [])
    stored_pref = list(
        perfil_guardat.get("ingredients_preferits", [])
        or perfil_guardat.get("preferencies", [])
        or perfil_guardat.get("preferred_ingredients", [])
        or []
    )
    stored_restr = list(
        perfil_guardat.get("restriccions", [])
        or perfil_guardat.get("restriccions_dietetiques", [])
        or perfil_guardat.get("dietary_restrictions", [])
        or []
    )
    stored_dieta = perfil_guardat.get("dieta")
    if stored_dieta and stored_dieta not in stored_restr:
        stored_restr.append(stored_dieta)
    stored_rejected_ing = list(perfil_guardat.get("rejected_ingredients", []) or [])
    stored_rejected_pairs = list(perfil_guardat.get("rejected_pairs", []) or [])

    _print_section(f"Hola {display_name}! Benvinguda/o al teu servei")
    print("Això és el que tinc guardat del teu perfil:")
    print(f"- Al·lèrgens: {_format_list(stored_alergies)}")
    print(f"- Ingredients preferits: {_format_list(stored_pref)}")
    print(f"- Restriccions dietètiques: {_format_list(stored_restr)}")
    print(f"- Ingredients vetats: {_format_list(stored_rejected_ing)}")
    print(f"- Parelles vetades: {_format_pairs(stored_rejected_pairs)}")

    if input_default("Vols actualitzar alguna dada del teu perfil? (s/n)", "n").strip().lower() == "s":
        if input_default("Volem revisar els al·lèrgens? (s/n)", "n").strip().lower() == "s":
            stored_alergies = seleccionar_alergens()
            perfil_guardat["alergies"] = list(stored_alergies)
        if input_default("Vols actualitzar ingredients preferits? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Escriu-los separats per comes (Enter per cap)", "")
            stored_pref = sorted(parse_list_input(txt))
            perfil_guardat["ingredients_preferits"] = stored_pref
        if input_default("Vols actualitzar restriccions dietètiques? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Restriccions (ex: vegan, celiac) [Enter per cap]", "")
            stored_restr = sorted(parse_list_input(txt))
            perfil_guardat["restriccions"] = stored_restr
            perfil_guardat["dieta"] = "vegan" if "vegan" in stored_restr else (
                "vegetarian" if "vegetarian" in stored_restr else perfil_guardat.get("dieta")
            )
        if input_default("Vols actualitzar ingredients vetats? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Ingredients vetats (separats per comes) [Enter per cap]", "")
            stored_rejected_ing = sorted(parse_list_input(txt))
            perfil_guardat["rejected_ingredients"] = stored_rejected_ing
        if input_default("Vols actualitzar parelles vetades? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Parella en format A+B, separades per comes [Enter per cap]", "")
            stored_rejected_pairs = _parse_pairs_input(txt)
            perfil_guardat["rejected_pairs"] = stored_rejected_pairs

        perfil_guardat.setdefault("display_name", display_name)
        user_profiles[str(user_id)] = perfil_guardat
        _save_user_profiles(PATH_USER_PROFILES, user_profiles)
        print("Perfecte, actualització guardada.")

    usar_alergies_guardades = bool(stored_alergies)
    if stored_alergies:
        usar_alergies_guardades = (
            input_default("Vols tenir en compte aquests al·lèrgens avui? (s/n)", "s")
            .strip()
            .lower()
            == "s"
        )

    # 1) Inicialitzem el Retriever
    retriever = Retriever(os.path.join("data", "base_de_casos.json"))

    while True:
        _print_section("Comencem una nova petició")
        print("Fem-ho fàcil: quatre preguntes i ens posem mans a l'obra.")

        # 2) Recollida de Dades (Inputs)
        tipus_esdeveniment = input_choice(
            "Per a quina mena d'ocasió és el menú?",
            ["casament", "aniversari", "empresa", "congres", "comunio"],
            "casament"
        )
        print("Perfecte, prenc nota.")
        temporada = input_choice(
            "En quina temporada el servirem?",
            ["primavera", "estiu", "tardor", "hivern"],
            "estiu"
        )
        print("Molt bona elecció.")
        servei = input_choice(
            "Com vols el servei?",
            ["assegut", "cocktail", "finger_food"],
            "assegut"
        )
        print("D'acord.")
        n_comensals = input_int_default("Quants comensals esperem?", 80)
        print("Perfecte.")
        preu_pers = input_float_default("Quin pressupost per persona tenim (€)?", 50.0)
        print("Genial.")
        
        # [NOU] Restriccions
        restr_default = ", ".join(stored_restr) if stored_restr else ""
        restr_input = input_default(
            "Hi ha alguna restricció dietètica a tenir en compte? (ex: celiac, vegan)",
            restr_default,
        )
        restriccions = parse_list_input(restr_input)
        if restriccions:
            print("Perfecte, ho tindré en compte.")
        else:
            print("Cap restricció dietètica, perfecte.")
        if restriccions != set(stored_restr):
            perfil_guardat = _get_user_profile(user_profiles, user_id)
            perfil_guardat["restriccions"] = sorted(restriccions)
            _save_user_profiles(PATH_USER_PROFILES, user_profiles)
            stored_restr = sorted(restriccions)
        if usar_alergies_guardades:
            alergies = list(stored_alergies)
            if alergies:
                print("Mantinc els al·lèrgens guardats.")
        else:
            print("Revisem els al·lèrgens per seguretat.")
            alergies = seleccionar_alergens()

        if alergies != stored_alergies:
            _store_user_alergies(user_profiles, user_id, alergies)
            _save_user_profiles(PATH_USER_PROFILES, user_profiles)
            stored_alergies = list(alergies)
            if alergies:
                usar_alergies_guardades = True
        dieta = None
        if "vegan" in restriccions:
            dieta = "vegan"
        elif "vegetarian" in restriccions:
            dieta = "vegetarian"

        perfil_usuari = {}
        if alergies:
            perfil_usuari["alergies"] = alergies
        if dieta:
            perfil_usuari["dieta"] = dieta
        if not perfil_usuari:
            perfil_usuari = None
        
        alcohol = input_choice("Vols que proposem begudes amb alcohol?", ["si", "no"], "si")
        print("Perfecte.")
        
        estil_culinari = ""

        # 3) Construcció del Problema
        problema = DescripcioProblema(
            tipus_esdeveniment=tipus_esdeveniment,
            temporada=temporada,
            n_comensals=n_comensals,
            preu_pers_objectiu=preu_pers, # Compte amb el nom del camp a la dataclass
            servei=servei,
            restriccions=restriccions,
            alcohol=alcohol,
            estil_culinari=estil_culinari
        )

        # 4) Recuperació (Retrieve)
        _print_section("Buscant propostes a la nostra carta")
        enfoc = ", ".join(restriccions) if restriccions else "estructura general"
        print(f"He trobat coincidències tenint en compte: {enfoc}.")
        resultats = retriever.recuperar_casos_similars(problema, k=5)
        imprimir_casos(resultats, top_k=3)

        if not resultats:
            if input_default("Vols provar amb una altra combinació? (s/n)", "s").lower() != 's':
                break
            continue

        # 5) Selecció del Cas
        idx = input_int_default("\nQuina proposta vols que prenguem com a base? (1-3)", 1)
        print("Molt bé, treballarem amb aquesta proposta.")
        cas_seleccionat = resultats[idx-1]["cas"]
        sol = cas_seleccionat["solucio"]

        plats = sol.get("plats", []) or []
        vetats_ingredients, parelles_vetades = _collect_vetats(perfil_guardat, learned_rules)

        def _agafa_plat(curs: str) -> dict:
            curs = str(curs).lower()
            for p in plats:
                if str(p.get("curs", "")).lower() == curs:
                    return p.copy()
            # fallback perquè no peti si falta algun curs
            return {"curs": curs, "nom": "—", "ingredients": []}

        plat1 = _agafa_plat("primer")
        plat2 = _agafa_plat("segon")
        postres = _agafa_plat("postres")
        vetats_per_curs = {"primer": set(), "segon": set(), "postres": set()}

        for plat in (plat1, plat2, postres):
            ings = list(plat.get("ingredients", []) or [])
            parelles_detectades = _parelles_detectades(ings, parelles_vetades)
            if parelles_detectades:
                alternatiu = _trobar_plat_alternatiu(
                    plat.get("curs", ""),
                    resultats,
                    vetats_ingredients,
                    parelles_vetades,
                    cas_seleccionat.get("id_cas"),
                )
                if alternatiu:
                    plat.clear()
                    plat.update(alternatiu)
                    plat.setdefault("log_transformacio", []).append(
                        "Substitució completa per parella vetada"
                    )
                else:
                    a, b = parelles_detectades[0].split("|", 1)
                    norm_ings = {_normalize_item(i) for i in ings}
                    ing_forcat = b if b in norm_ings else a
                    vetats_per_curs[str(plat.get("curs", "")).lower()].add(ing_forcat)
                    plat.setdefault("log_transformacio", []).append(
                        f"Substitució parcial per parella vetada ({a} + {b})"
                    )
        ingredients_originals = {
            "primer": list(plat1.get("ingredients", []) or []),
            "segon": list(plat2.get("ingredients", []) or []),
            "postres": list(postres.get("ingredients", []) or []),
        }

        # 5.5) Substitucio previa d'ingredients prohibits (al.lergens i vetos)
        if perfil_usuari or vetats_ingredients or any(vetats_per_curs.values()):
            _print_section("Primer pas: seguretat alimentària")
            print("Reviso al·lèrgens i dietes per evitar riscos.")
            ingredients_usats = set()
            resums_prohibits = []
            plats_pre = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]
            for etiqueta, p in plats_pre:
                ingredients = list(p.get("ingredients", []) or [])
                prohibits = ingredients_incompatibles(ingredients, kb, perfil_usuari)
                prohibits.update(vetats_ingredients)
                prohibits.update(vetats_per_curs.get(str(p.get("curs", "")).lower(), set()))
                if not prohibits:
                    continue
                plat_tmp = {"nom": p.get("nom", ""), "ingredients": ingredients}
                adaptat = substituir_ingredients_prohibits(
                    plat_tmp,
                    prohibits,
                    kb,
                    perfil_usuari=perfil_usuari,
                    ingredients_usats=ingredients_usats,
                    parelles_prohibides=parelles_vetades,
                    preferits=stored_pref,
                )
                if isinstance(adaptat, dict):
                    p["ingredients"] = adaptat.get("ingredients", ingredients)
                    logs = list(p.get("log_transformacio", []) or [])
                    logs_sub = adaptat.get("log_transformacio", []) or []
                    logs.extend(logs_sub)
                    if logs:
                        p["log_transformacio"] = logs
                    if logs_sub:
                        resums_prohibits.append((etiqueta, logs_sub))

            if resums_prohibits:
                print("\nAjustos per seguretat:")
                for etiqueta, logs_sub in resums_prohibits:
                    print(f"- {etiqueta}:")
                    for log in logs_sub:
                        print(f"  {log}")

        # 6) Adaptació d'ingredients
        _print_section("Personalitzem els ingredients")
        suggeriment = estil_culinari if estil_culinari in kb.estils_latents else ""
        estils_txt = " | ".join(sorted(kb.estils_latents.keys()))
        estil_latent = input_default(
            f"Vols donar un toc d'estil latent? ({estils_txt})",
            suggeriment
        ).strip().lower()

        if estil_latent:
            if estil_latent not in kb.estils_latents:
                print(f"\nNo tinc l'estil '{estil_latent}' a la carta.")
                print(f"Opcions disponibles: {estils_txt}")

            intensitat = float(input_default("Quina intensitat vols? (0.1 - 0.9)", "0.5"))
            print(f"\nPerfecte. Apliquem l'estil {estil_latent} amb intensitat {intensitat}.")

            plats = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]
            ingredients_estil_usats = set()

            resums = []
            for etiqueta, p in plats:
                etiqueta_short = etiqueta.split()[0]
                ingredients_abans = list(p.get("ingredients", []) or [])

                resultat = substituir_ingredient(
                    p,
                    estil_latent,
                    kb,
                    mode="latent",
                    intensitat=intensitat,
                    ingredients_estil_usats=ingredients_estil_usats,
                    perfil_usuari=perfil_usuari,
                    parelles_prohibides=parelles_vetades,
                )

                # Si l’operador retorna un plat nou, enganxem resultats al dict original
                if isinstance(resultat, dict) and resultat is not p:
                    p.clear()
                    p.update(resultat)

                _, resum = imprimir_resum_adaptacio(
                    etiqueta_short,
                    p,
                    ingredients_abans,
                    estil_latent,
                    intensitat,
                    kb,
                    ingredients_original=ingredients_originals.get(
                        str(p.get("curs", "")).lower()
                    ),
                )
                resums.append((etiqueta_short, resum))

            print(f"\nResum del toc d'estil ({estil_latent}):")
            for etiqueta, resum in resums:
                print(f"{etiqueta.capitalize()}: {resum}")

        _try_add_preferred_touch(
            [plat1, plat2, postres],
            stored_pref,
            perfil_usuari,
            vetats_ingredients,
            parelles_vetades,
        )

        if vetats_ingredients:
            plats_post = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]
            for _, p in plats_post:
                ingredients = list(p.get("ingredients", []) or [])
                prohibits = ingredients_incompatibles(ingredients, kb, perfil_usuari)
                prohibits.update(vetats_ingredients)
                prohibits.update(vetats_per_curs.get(str(p.get("curs", "")).lower(), set()))
                if not prohibits:
                    continue
                adaptat = substituir_ingredients_prohibits(
                    {"nom": p.get("nom", ""), "ingredients": ingredients},
                    prohibits,
                    kb,
                    perfil_usuari=perfil_usuari,
                    parelles_prohibides=parelles_vetades,
                    preferits=stored_pref,
                )
                if isinstance(adaptat, dict):
                    p["ingredients"] = adaptat.get("ingredients", ingredients)
                    logs = list(p.get("log_transformacio", []) or [])
                    logs_sub = adaptat.get("log_transformacio", []) or []
                    logs.extend(logs_sub)
                    if logs:
                        p["log_transformacio"] = logs
        # debug_kb_match(plat1, kb, "PRIMER")
        # debug_kb_match(plat2, kb, "SEGON")
        # debug_kb_match(postres, kb, "POSTRES")

        # 7) Adaptació 2: Tècniques i Presentació
        # --- IMPORTANT: sempre inicialitzar per evitar UnboundLocalError ---
        estil_cultural = ""   # pot quedar buit si l'usuari no tria cultural

        if estil_latent:
            suggerits = kb.suggerir_estils_culturals_per_latent(estil_latent, top_k=6)

            if suggerits:
                print(f"\nJa hem donat el toc '{estil_latent}'.")
                print("Si vols, podem afegir un estil cultural que hi combini:")
                for i, nom_estil in enumerate(suggerits, 1):
                    row = kb.get_info_estil(nom_estil) or {}
                    alias = row.get("alias") or nom_estil
                    sabors = row.get("sabors_clau") or "—"
                    print(f"  {i}) {alias} ({nom_estil}) | sabors: {sabors}")

                txt = input_default("En vols triar un? (0 = no)", "0").strip()
                try:
                    idxc = int(txt)
                except ValueError:
                    idxc = 0

                if 1 <= idxc <= len(suggerits):
                    estil_cultural = suggerits[idxc - 1]
            else:
                print(f"\nJa hem adaptat a '{estil_latent}', però no tinc estils culturals clars per aquest toc.")

        _print_section("Tècniques i presentació")
        vol = input_default("Vols donar un toc d'alta cuina? (s/n)", "n").strip().lower()

        estil_tecnic = ""
        if vol == "s":
            estils_alta = kb.imprimir_estils_per_tipus("alta_cuina")
            if estils_alta:
                txt = input_default("Tria el número d'estil (0 = cap)", "0").strip()
                try:
                    idx = int(txt)
                except ValueError:
                    idx = 0

                if 1 <= idx <= len(estils_alta):
                    estil_tecnic = estils_alta[idx - 1]
                else:
                    estil_tecnic = ""

        transf_1, transf_2, transf_post = [], [], []
        info_llm_1, info_llm_2, info_llm_post = None, None, None

        te_cultural = bool(estil_cultural)
        te_alta = bool(estil_tecnic)

        if te_cultural and te_alta:
            mode_ops = "mixt"
        elif te_cultural:
            mode_ops = "cultural"
        elif te_alta:
            mode_ops = "alta"
        else:
            mode_ops = ""

        if mode_ops:
            print(f"Perfecte. Apliquem tècniques en mode '{mode_ops}' (fins a 2 per plat).")

            plats_llista = [plat1, plat2, postres]

            t_menu = triar_tecniques_2_operadors_per_menu(
                plats=plats_llista,
                mode=mode_ops,
                estil_cultural=estil_cultural if te_cultural else None,
                estil_alta=estil_tecnic if te_alta else None,
                base_estils=kb.estils,
                base_tecnniques=kb.tecniques,
                kb=kb,
                min_score=5,
                debug=False,
            )

            transf_1, transf_2, transf_post = t_menu[0], t_menu[1], t_menu[2]
        else:
            print("Cap estil cultural ni d'alta cuina seleccionat. Ho deixem clàssic.")

        print("\nResum de tècniques aplicades:")
        imprimir_menu_final_net(
            plat1, transf_1,
            plat2, transf_2,
            postres, transf_post,
            estil_cultural=estil_cultural if te_cultural else None,
            estil_alta=estil_tecnic if te_alta else None,
        )

        # Generació de Text (Gemini)
        if input_default("Vols que redacti noms i descripcions més elegants? (s/n)", "n").lower() == 's':
            estil_tecnic_llm = estil_tecnic if estil_tecnic else "classic"
            estil_row = kb.estils.get(estil_tecnic)
            info_llm_1 = genera_descripcio_llm(plat1, transf_1, estil_tecnic_llm, servei, estil_row)
            info_llm_2 = genera_descripcio_llm(plat2, transf_2, estil_tecnic_llm, servei, estil_row)
            info_llm_post = genera_descripcio_llm(postres, transf_post, estil_tecnic_llm, servei, estil_row)


        # 8) Afegir begudes
        # --- NOU: combinar restriccions + al·lèrgies guardades ---
        restriccions_beguda = set()
        if perfil_usuari and perfil_usuari.get("alergies"):
            restriccions_beguda.update(perfil_usuari["alergies"])  
        
        if restriccions:
            restriccions_beguda.update(r.lower() for r in restriccions)

        restriccions_beguda = list(restriccions_beguda)
        _print_section("Maridatge de begudes")
        begudes_usades = set()
        beguda1, score1 = recomana_beguda_per_plat(
            plat1,
            list(kb.begudes.values()),
            base_ingredients_list,
            restriccions_beguda,
            alcohol,
            begudes_usades,
        )
        beguda2, score2 = recomana_beguda_per_plat(
            plat2,
            list(kb.begudes.values()),
            base_ingredients_list,
            restriccions_beguda,
            alcohol,
            begudes_usades,
        )
        beguda_postres, score_postres = recomana_beguda_per_plat(
            postres,
            list(kb.begudes.values()),
            base_ingredients_list,
            restriccions_beguda,
            alcohol,
            begudes_usades,
        )

        
        # 9) Resultat Final
        imprimir_menu_final(plat1, transf_1, info_llm_1, beguda1, score1, plat2, transf_2, info_llm_2, beguda2, score2, postres, transf_post, info_llm_post, beguda_postres, score_postres)

        # 9.1) Imatge del menú (opcional)
        if input_default("Vols generar una imatge detallada del menú? (s/n)", "n").lower() == 's':
            plats_info = []
            for plat, info in [
                (plat1, info_llm_1),
                (plat2, info_llm_2),
                (postres, info_llm_post),
            ]:
                plats_info.append({
                    "curs": plat.get("curs", ""),
                    "nom": info.get("nom_nou", plat.get("nom", "—")) if info else plat.get("nom", "—"),
                    "ingredients": plat.get("ingredients", []) or [],
                    "descripcio": info.get("descripcio_carta", "") if info else "",
                    "presentacio": info.get("presentacio", "") if info else "",
                })

            prompt_imatge = construir_prompt_imatge_menu(
                tipus_esdeveniment=tipus_esdeveniment,
                temporada=temporada,
                espai="interior",
                formalitat=problema.formalitat,
                plats_info=plats_info,
            )
            genera_imatge_menu_hf(prompt_imatge, output_path="menu_event.png")
        
        # 10) FASE REVISE (Dual Memory)
        gestor_revise = GestorRevise()
        cas_proposat = {
            "problema": problema,
            "solucio": {"primer": plat1, "segon": plat2, "postres": postres}
        }
        resultat_avaluacio = gestor_revise.avaluar_proposta(cas_proposat, user_id)
        print(f"\nGràcies pel feedback. Resultat global: {resultat_avaluacio['tipus_resultat']}")

        # 11) FASE RETAIN (Política de memòria)
        _print_section("Memòria del sistema")
        print("Decideixo si aquest cas s'ha de recordar per al futur.")
        map_resultat = {
            "CRITICAL_FAILURE": "fracas_critic",
            "SOFT_FAILURE": "fracas_suau",
            "SUCCESS": "exit",
        }
        resultat_retain = map_resultat.get(resultat_avaluacio["tipus_resultat"], "fracas_suau")

        transformation_log = []
        for p in [plat1, plat2, postres]:
            transformation_log.extend(p.get("log_transformacio", []) or [])
        for t in (transf_1 or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")
        for t in (transf_2 or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")
        for t in (transf_post or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")
        for curs, beguda in [
            ("primer", beguda1),
            ("segon", beguda2),
            ("postres", beguda_postres),
        ]:
            if beguda:
                transformation_log.append(
                    f"Maridatge: Generat nou maridatge per {curs} ({beguda.get('nom', '—')})"
                )

        saved = kb.retain_case(
            new_case=cas_proposat,
            evaluation_result=resultat_retain,
            transformation_log=transformation_log,
            user_score=resultat_avaluacio["puntuacio_global"],
            retriever_instance=retriever,
        )
        if saved:
            print("Perfecte. Guardem aquest cas com a referència.")
        else:
            print("D'acord. No el guardem aquesta vegada.")

        if input_default("\nVols preparar un altre menú? (s/n)", "n").lower() != 's':
            print("Gràcies! Bon profit i fins aviat. 👋")
            break

if __name__ == "__main__":
    main()
