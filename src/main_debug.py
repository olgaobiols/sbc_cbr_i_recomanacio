import copy
import json
import os
from typing import List, Set, Dict, Any, Optional, Tuple
import numpy as np
import sys

from estructura_cas import DescripcioProblema
from Retriever import Retriever
from knowledge_base import KnowledgeBase
from gestor_feedback import GestorRevise
from operadors_transformacio_realista import (
    substituir_ingredient,
    triar_tecniques_2_operadors_per_menu,
    genera_fitxes_menu_llm_1call, 
    construir_prompt_imatge_menu,
    genera_imatge_menu_hf_o_prompt,
    ingredients_ca_llista,
    ingredient_ca,
    _resum_ambient_esdeveniment,
    model_gemini
)


from operador_ingredients import (
    FG_WRAPPER,
    ingredients_incompatibles,
    substituir_ingredients_prohibits,
)

from operadors_begudes import (
    recomana_beguda_per_plat,
    get_ingredient_principal,
    passa_filtre_dur,
    passa_restriccions,
    score_beguda_per_plat,
)

# =========================
#   INICIALITZACIÓ GLOBAL
# =========================

# 1. Carreguem el cervell (Ontologia + Estils)
kb = KnowledgeBase()


# Com que els operadors antics esperen llistes de diccionaris, 
# creem referències compatibles per no trencar res:
base_ingredients_list = list(kb.ingredients.values())

# =========================
#   FUNCIONS AUXILIARS CLI
# =========================

UI_WIDTH = 80
PROMPT_PAD = 32

def _line(ch: str = "=", width: int = UI_WIDTH) -> str:
    return ch * width

def _print_banner(title: str) -> None:
    line = _line("=")
    print(line)
    print(title)
    print(line)
    print("")

def _print_block(title: str, ch: str = "-") -> None:
    line = _line(ch)
    print("\n" + line)
    print(title)
    print(line)

def _print_section_line(title: str, ch: str = "=") -> None:
    prefix = f"=== {title} "
    line = prefix + (ch * max(0, UI_WIDTH - len(prefix)))
    print("\n\n" + line)

def _prompt(label: str) -> str:
    return f"{label:<{PROMPT_PAD}}"

def _prompt_inline(label: str) -> str:
    return f"> {label:<{PROMPT_PAD}}"

def _print_kv(label: str, value: str) -> None:
    key = f"{label}:"
    print(f"{key:<18} {value}")

def _print_options(options: List[str], label: str = "Opcions") -> None:
    if not options:
        return
    text = " | ".join(options)
    _print_kv(label, text)

def input_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default

def input_optional(prompt, placeholder="Enter"):
    return input(f"{prompt} [{placeholder}]: ").strip()

def input_choice(prompt, options, default, indiferent_value=None, show_options_in_prompt=True):
    opts = list(options)
    if "indiferent" not in opts:
        opts.append("indiferent")
    opts_txt = "/".join(opts)
    while True:
        if show_options_in_prompt:
            txt = input(f"{prompt} ({opts_txt}) [{default}]: ").strip().lower()
        else:
            txt = input(f"{prompt} [{default}]: ").strip().lower()
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

def input_float_default(prompt, default, display_default: Optional[str] = None):
    shown = display_default if display_default is not None else default
    txt = input(f"{prompt} [{shown}]: ").strip()
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

def parse_list_input_list(txt: str) -> List[str]:
    """Converteix 'gluten, vegan' en ['gluten', 'vegan'] preservant l'ordre."""
    if not txt:
        return []
    items = []
    vistos = set()
    for part in txt.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token not in vistos:
            items.append(token)
            vistos.add(token)
    return items

def parse_restriccions_input(txt: str) -> Set[str]:
    """Admet codis d'al·lèrgens (01-14), dietes i ingredients en anglès."""
    if not txt:
        return set()
    resultat = set()
    for part in txt.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(EU_ALLERGENS):
                resultat.add(EU_ALLERGENS[idx - 1][0])
            resultat.add(str(idx))
            continue
        resultat.add(token)
    return resultat

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

def _display_dieta_tag(value: str) -> str:
    if not value:
        return value
    norm = _normalize_item(value)
    if norm in {"halal friendly", "halal_friendly"}:
        return "halal"
    if norm in {"kosher friendly", "kosher_friendly"}:
        return "kosher"
    return value

def _normalize_dieta_tag(value: str) -> str:
    if not value:
        return ""
    norm = _normalize_item(value)
    if norm in {"vegan", "vegetarian"}:
        return norm
    if norm in {"halal", "halal friendly", "halal_friendly"}:
        return "halal_friendly"
    if norm in {"kosher", "kosher friendly", "kosher_friendly"}:
        return "kosher_friendly"
    return norm

def _infer_dieta_from_restriccions(restriccions_set: Set[str]) -> Optional[str]:
    mapped = {_normalize_dieta_tag(r) for r in restriccions_set if r}
    for cand in ("vegan", "vegetarian", "halal_friendly", "kosher_friendly"):
        if cand in mapped:
            return cand
    return None

def _format_restriccions(items: List[str]) -> str:
    if not items:
        return "—"
    return ", ".join(_display_dieta_tag(x) for x in items)

def _format_techniques(transf: Optional[List[Any]], kb: Optional[Any] = None) -> str:
    if not transf:
        return "cap"
    noms = []
    for t in transf:
        if isinstance(t, str):
            name = t.strip()
        elif isinstance(t, dict):
            name = (t.get("display") or t.get("nom") or t.get("name") or "").strip()
            obj = (t.get("objectiu_ingredient") or "").strip()
            if obj:
                if kb:
                    obj = ingredient_ca(kb, obj) or obj
                name = f"{name} de {obj}" if name else f"{obj}"
        else:
            name = str(t).strip()
        if name:
            noms.append(name)
    if not noms:
        return "cap"
    vistos = set()
    resultat = []
    for n in noms:
        if n not in vistos:
            resultat.append(n)
            vistos.add(n)
    return ", ".join(resultat)

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
    line = _line("-")
    print("\n" + line)
    print(title)
    print(line)

EU_ALLERGENS = [
    ("gluten", "Gluten"),
    ("crustaceans", "Crustacis"),
    ("egg", "Ous"),
    ("fish", "Peix"),
    ("peanuts", "Cacauets"),
    ("soybeans", "Soja"),
    ("milk", "Llet"),
    ("nuts", "Fruits secs"),
    ("celery", "Api"),
    ("mustard", "Mostassa"),
    ("sesame", "Sèsam"),
    ("sulfites", "Sulfits"),
    ("lupin", "Tramussos"),
    ("molluscs", "Mol·luscs"),
]

def imprimir_allergens_taula() -> None:
    line = _line("-")
    print("\n" + line)
    print("LLISTA D'AL·LÈRGENS UE")
    print(line)
    col_width = max(18, UI_WIDTH // 4)
    row = []
    for idx, (_, label) in enumerate(EU_ALLERGENS, start=1):
        cell = f"{idx:02d}. {label}"
        row.append(cell.ljust(col_width))
        if len(row) == 4:
            print("".join(row).rstrip())
            row = []
    if row:
        print("".join(row).rstrip())
    print(line)

def seleccionar_alergens() -> List[str]:
    imprimir_allergens_taula()
    txt = input_default(
        "Quins al·lèrgens hem d'evitar? (números separats per comes, Enter si cap)",
        "",
    )
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

def _safe_ascii_prompt(text: str) -> str:
    if not text:
        return text
    return (
        text
        .replace("“", '"')
        .replace("”", '"')
        .replace("’", "'")
        .replace("–", "-")
        .replace("—", "-")
        .replace("°", " degrees")
    )

def imprimir_resum_adaptacio(
    etiqueta_plat: str,
    plat: dict,
    ingredients_abans: List[str],
    nom_estil: str,
    intensitat: float,
    kb: Any,
    ingredients_original: Optional[List[str]] = None,
    verbose: bool = True,
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

    if verbose:
        print(f"\n[{etiqueta_plat}] {plat.get('nom','—')}")
        print(f"- Ingredients: {len(ingredients_abans)} -> {len(ingredients_despres_unics)}")
        if ingredients_original and ingredients_original != ingredients_abans:
            print(f"- Abans (original): {', '.join(ingredients_original) if ingredients_original else '—'}")
        print(f"- Abans (post-restriccions): {', '.join(ingredients_abans) if ingredients_abans else '—'}")
        print(f"- Despres: {', '.join(ingredients_despres_unics) if ingredients_despres_unics else '—'}")
    condiment = plat.get("condiment")
    if condiment and verbose:
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
        if verbose:
            print(f"- Nota: l'ingredient ja hi era ({duplicat_proposat}) i no l'he duplicat")
        canvi_text = "CAP CANVI"
        motiu = "toc ja present al plat"

    if verbose:
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


def _build_plats_info_for_image(plat1, plat2, postres, info_llm_1, info_llm_2, info_llm_post, beguda1, beguda2, beguda_postres):
    plats_info = []
    for plat, info, beguda in [
        (plat1, info_llm_1, beguda1),
        (plat2, info_llm_2, beguda2),
        (postres, info_llm_post, beguda_postres),
    ]:
        plats_info.append({
            "curs": plat.get("curs", "") or "",
            "nom": (info or {}).get("nom_nou") or plat.get("nom", "—"),
            "ingredients": plat.get("ingredients", []) or [],
            # IMPORTANT: si tens el teu nou construir_prompt sense LLM, pots prescindir de descripcio/presentacio
            "descripcio": (info or {}).get("descripcio_carta", "") or "",
            "presentacio": (info or {}).get("presentacio", "") or "",
            # clau: usa la frase ja feta del 1call (si existeix)
            "image_sentence_en": (info or {}).get("image_sentence_en", "") or "",
            # per begudes (si ho vols al prompt)
            "beguda_nom": (beguda or {}).get("nom", "") or "",
        })
    return plats_info


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

import re

def imprimir_menu_final(
    kb,
    plat1, transf_1, info_llm_1, beguda1, score1,
    plat2, transf_2, info_llm_2, beguda2, score2,
    postres, transf_post, info_llm_post, beguda_postres, score_postres,
    mostrar_logs=True,
):
    print("\n" + _line("="))
    print("PROPOSTA DE MENÚ DEFINITIVA")
    print(_line("="))

    def _cap(text: str) -> str:
        return (text or "").strip()

    def _ingredients_ca(plat: dict) -> str:
        ings_en = plat.get("ingredients", []) or []
        ings_ca = ingredients_ca_llista(kb, ings_en) if kb else ings_en
        return ", ".join([x for x in ings_ca if x]) if ings_ca else "—"

    def _bloc(titol: str, text: str, indent="   "):
        txt = _cap(text)
        if not txt:
            return
        lines = [line.strip() for line in txt.splitlines() if line.strip()]
        if len(lines) == 1:
            print(f"{indent}{titol}: {lines[0]}")
            return
        print(f"{indent}{titol}:")
        for line in lines:
            print(f"{indent}  {line}")

    # --- NOU: neteja del log (treu fallback + ingredient en català quan toca) ---
    def _neteja_log(log: str) -> str:
        txt = _cap(log)
        if not txt:
            return ""

        # 1) treu "(Fallback Simbòlic)" (i variants)
        txt = re.sub(r"\(\s*fallback\s*simb[oò]lic\s*\)", "", txt, flags=re.IGNORECASE)
        txt = re.sub(r"\s{2,}", " ", txt).strip(" -·")

        # 2) casos típics: "Afegit X" o "Tret X" -> X en català via KB
        m = re.search(r"\b(Afegit|Tret)\s+([A-Za-z_\-]+)\b", txt, flags=re.IGNORECASE)
        if m and kb:
            verb = m.group(1)
            ing_en = m.group(2)
            ing_ca = ingredient_ca(kb, ing_en.lower())
            if ing_ca and ing_ca.lower() != ing_en.lower():
                txt = re.sub(rf"\b{re.escape(ing_en)}\b", ing_ca, txt)

        return txt.strip()

    # --- NOU (opcional): catalanitza 4 tokens típics dins condiment ---
    def _neteja_condiment(cond: str) -> str:
        c = _cap(cond)
        if not c or not kb:
            return c
        # només intentem substituir paraules “ingredient-like”
        tokens = re.findall(r"[A-Za-z][A-Za-z_\-]*", c)
        for tok in sorted(set(tokens), key=len, reverse=True):
            ca = ingredient_ca(kb, tok.lower())
            if ca and ca.lower() != tok.lower():
                c = re.sub(rf"\b{re.escape(tok)}\b", ca, c)
        return c

    preu_total_menu = 0.0

    plats_iter = [
        ("PRIMER PLAT", plat1, transf_1, info_llm_1, beguda1, score1),
        ("SEGON PLAT",  plat2, transf_2, info_llm_2, beguda2, score2),
        ("POSTRES",     postres, transf_post, info_llm_post, beguda_postres, score_postres),
    ]

    def _line_with_price(text: str, price: float) -> str:
        price_txt = f"{price:.2f}€"
        fill_width = max(10, UI_WIDTH - len(price_txt) - 1)
        return f"{text:.<{fill_width}} {price_txt}"

    for etiqueta, plat, transf, info_llm, beguda, score in plats_iter:
        nom = _cap((info_llm or {}).get("nom_nou")) or _cap(plat.get("nom")) or "—"
        desc = _cap((info_llm or {}).get("descripcio_carta"))
        pres = _cap((info_llm or {}).get("presentacio"))
        notes = _cap((info_llm or {}).get("notes_tecnniques"))
        beguda_llm = _cap((info_llm or {}).get("beguda_llm"))
        condiment = _neteja_condiment(plat.get("condiment", ""))  # <-- canvi petit

        preu_plat = float(plat.get("preu", 0.0) or 0.0)
        preu_total_menu += preu_plat

        print(f"\n{etiqueta}")
        print(_line("-"))
        print(_line_with_price(nom.upper(), preu_plat))
        print(f"   Ingredients: {_ingredients_ca(plat)}")
        if condiment:
            print(f"   Condiment: {condiment}")
        print(f"   Tècniques: {_format_techniques(transf, kb)}")
        _bloc("Descripció", desc)
        _bloc("Presentació", pres)
        if notes:
            _bloc("Notes del xef", notes)

        logs = plat.get("log_transformacio", []) or []
        if mostrar_logs and logs:
            print("   Ajustos aplicats:")
            for log in logs:
                log_net = _neteja_log(log)
                if log_net:
                    print(f"     - {log_net}")

        if beguda is not None:
            preu_beguda = float(beguda.get("preu_cost", 0.0) or 0.0)
            preu_total_menu += preu_beguda
            nom_beguda = _cap(beguda.get("nom")) or "—"
            print(f"   Maridatge: {nom_beguda} (afinitat {score:.2f}) {preu_beguda:.2f}€")
        else:
            if beguda_llm and beguda_llm.lower() != "sense recomanació de beguda":
                print(f"   Maridatge: {beguda_llm}")
            else:
                print("   Maridatge: —")

    print("\n" + _line("-"))
    print(f"{'TOTAL PER PERSONA':.<{UI_WIDTH - 10}} {preu_total_menu:>7.2f}€")
    print(_line("="))


def debug_kb_match(plat, kb, etiqueta=""):
    print(f"\n[KB CHECK] {etiqueta} — {plat.get('nom','—')}")
    for ing in plat.get("ingredients", []):
        ok = kb.get_info_ingredient(ing) is not None
        print(f"  - {ing}  ->  {'✅' if ok else '❌ NO A KB'}")



# =========================
#   MAIN INTERACTIU
# =========================

def main():
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    _print_banner("SISTEMA DE RECOMANACIÓ DE MENÚS RICO RICO 2.0")

    COST_INGREDIENT_EXTRA = 3
    COST_TECNICA_ALTA = 10
    COST_TECNICA_CULTURAL = 5

    user_id_raw = input_default("Identificació d'usuari", "guest").strip()
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
    if stored_dieta:
        etiqueta_dieta = _display_dieta_tag(stored_dieta)
        if etiqueta_dieta and etiqueta_dieta not in stored_restr:
            stored_restr.append(etiqueta_dieta)
    stored_rejected_ing = list(perfil_guardat.get("rejected_ingredients", []) or [])
    stored_rejected_pairs = list(perfil_guardat.get("rejected_pairs", []) or [])

    print(f"\n[PERFIL DESAT - {display_name.upper()}]:")
    print(_line("-"))
    _print_kv("Al·lèrgens", _format_list(stored_alergies))
    _print_kv("Preferits", _format_list(stored_pref))
    _print_kv("Restriccions", _format_restriccions(stored_restr))
    _print_kv("Vetats", _format_list(stored_rejected_ing))
    _print_kv("Parelles vetades", _format_pairs(stored_rejected_pairs))
    print(_line("-"))

    if input_default("Vols actualitzar alguna d'aquestes dades? (s/n)", "n").strip().lower() == "s":
        if input_default("Volem revisar els al·lèrgens? (s/n)", "n").strip().lower() == "s":
            stored_alergies = seleccionar_alergens()
            perfil_guardat["alergies"] = list(stored_alergies)
        if input_default("Vols actualitzar ingredients preferits? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Escriu-los en anglès separats per comes (Enter per cap)", "")
            stored_pref = sorted(parse_list_input(txt))
            perfil_guardat["ingredients_preferits"] = stored_pref
        if input_default("Vols actualitzar restriccions dietètiques? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Restriccions (ex: vegan, halal) [Enter per cap]", "")
            stored_restr = sorted(parse_list_input(txt))
            perfil_guardat["restriccions"] = stored_restr
            dieta_inf = _infer_dieta_from_restriccions(set(stored_restr))
            if dieta_inf:
                perfil_guardat["dieta"] = dieta_inf
        if input_default("Vols actualitzar ingredients vetats? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Ingredients vetats (en anglès, separats per comes) [Enter per cap]", "")
            stored_rejected_ing = sorted(parse_list_input(txt))
            perfil_guardat["rejected_ingredients"] = stored_rejected_ing
        if input_default("Vols actualitzar parelles vetades? (s/n)", "n").strip().lower() == "s":
            txt = input_default("Parella en format A+B, separades per comes [Enter per cap]", "")
            stored_rejected_pairs = _parse_pairs_input(txt)
            perfil_guardat["rejected_pairs"] = stored_rejected_pairs

        perfil_guardat.setdefault("display_name", display_name)
        user_profiles[str(user_id)] = perfil_guardat
        _save_user_profiles(PATH_USER_PROFILES, user_profiles)
        print(f"\nPerfecte {display_name}, hem actualitzat les teves preferències!")

    # 1) Inicialitzem el Retriever
    retriever = Retriever(os.path.join("data", "base_de_casos.json"))

    def _perfil_from_restriccions(restriccions_set: Set[str]) -> Optional[Dict[str, Any]]:
        if not restriccions_set:
            return None
        allergens_keys = {k for k, _ in EU_ALLERGENS}
        alergies = sorted({r for r in restriccions_set if r in allergens_keys})
        dieta = _infer_dieta_from_restriccions(restriccions_set)
        perfil = {}
        if alergies:
            perfil["alergies"] = alergies
        if dieta:
            perfil["dieta"] = dieta
        return perfil or None

    def _prohibits_per_plat(
        ingredients: List[str],
        restriccions_set: Set[str],
        perfil: Optional[Dict[str, Any]],
    ) -> Set[str]:
        prohibits = set()
        if perfil:
            prohibits.update(ingredients_incompatibles(ingredients, kb, perfil))
        if restriccions_set:
            norm_map = {_normalize_item(i): i for i in ingredients if i}
            for r in restriccions_set:
                r_norm = _normalize_item(r)
                if r_norm in norm_map:
                    prohibits.add(norm_map[r_norm])
        return prohibits

    def _aplica_restriccions_plat(
        plat: Dict[str, Any],
        restriccions_set: Set[str],
        perfil: Optional[Dict[str, Any]],
    ) -> None:
        ingredients = list(plat.get("ingredients", []) or [])
        prohibits = _prohibits_per_plat(ingredients, restriccions_set, perfil)
        if not prohibits:
            return
        prohibits_total = set(prohibits) | set(restriccions_set)
        adaptat = substituir_ingredients_prohibits(
            plat,
            prohibits_total,
            kb,
            perfil_usuari=perfil,
        )
        if isinstance(adaptat, dict):
            plat.clear()
            plat.update(adaptat)

    def _get_plat(plats: List[dict], curs: str) -> dict:
        curs_norm = str(curs).lower()
        for p in plats:
            if str(p.get("curs", "")).lower() == curs_norm:
                return p
        return {"curs": curs_norm, "nom": "—", "ingredients": []}

    def _diff_ingredients(base: List[str], variant: List[str]) -> List[str]:
        canvis = []
        max_len = max(len(base), len(variant))
        for i in range(max_len):
            ing_base = base[i] if i < len(base) else None
            ing_var = variant[i] if i < len(variant) else None
            if ing_base == ing_var:
                continue
            if ing_base and ing_var:
                canvis.append(f"{ing_base} -> {ing_var}")
            elif ing_base and not ing_var:
                canvis.append(f"{ing_base} -> (eliminat)")
            elif ing_var and not ing_base:
                canvis.append(f"(afegit) {ing_var}")
        return canvis

    while True:
        _print_section_line("NOVA PETICIÓ")

        # 2) Recollida de Dades (Inputs)
        opc_esdeveniment = ["casament", "aniversari", "empresa", "congres", "comunio"]
        tipus_esdeveniment = input_choice(
            _prompt(f"1. Tipus d'esdeveniment ({' | '.join(opc_esdeveniment)}):"),
            opc_esdeveniment,
            "casament",
            show_options_in_prompt=False,
        )

        opc_temporada = ["primavera", "estiu", "tardor", "hivern"]
        temporada = input_choice(
            _prompt(f"2. Temporada ({' | '.join(opc_temporada)}):"),
            opc_temporada,
            "estiu",
            show_options_in_prompt=False,
        )

        opc_servei = ["assegut", "cocktail", "finger_food"]
        servei = input_choice(
            _prompt(f"3. Tipus de servei ({' | '.join(opc_servei)}):"),
            opc_servei,
            "assegut",
            show_options_in_prompt=False,
        )
        n_comensals = input_int_default(_prompt("4. Nombre de comensals:"), 80)
        preu_pers = input_float_default(
            _prompt("5. Pressupost aprox. per persona (€):"),
            50.0,
            "50.0 €",
        )
        
        # 2.1) Fase d'inputs de restriccions (globals i subgrups)
        imprimir_allergens_taula()

        print("Dietes disponibles: vegan | vegetarian | halal | kosher")
        restr_txt = input_optional(
            _prompt_inline(
                "Restriccions GLOBALS (tothom) (codi 01-14, ingredient anglès o dieta):"
            ),
            "Enter",
        )
        restriccions = parse_restriccions_input(restr_txt)

        txt = input_default(_prompt_inline("Subgrups especials (número):"), "0")
        try:
            n_subgrups = int(txt) if txt else 0
        except ValueError:
            print("  Valor no vàlid. Assumim 0 subgrups.")
            n_subgrups = 0
        if n_subgrups < 0:
            n_subgrups = 0

        subgroups = []
        for i in range(n_subgrups):
            nom_grup = input("  Nom del grup: ").strip()
            if not nom_grup:
                nom_grup = f"Grup {i + 1}"
            restr_grup_txt = input("  Restriccions: ").strip()
            restr_grup = parse_list_input_list(restr_grup_txt)
            subgroups.append(
                {"name": nom_grup, "restrictions": restr_grup, "is_vip": False}
            )
            restr_disp = ", ".join(restr_grup) if restr_grup else "—"
            print(f"  Confirmat: Grup {i + 1} '{nom_grup}' | Restriccions: {restr_disp}")

        if n_comensals > 1 and (stored_rejected_ing or stored_alergies or stored_restr or stored_dieta):
            vol_vip = input_default(
                _prompt_inline(f"Menú personalitzat per l'amfitrió {display_name.title()}? (tindrem en compte preferències!) (s/n):"),
                "n",
            ).strip().lower()
            if vol_vip == "s":
                vip_restr = []
                vip_restr.extend(stored_alergies)
                vip_restr.extend(stored_restr)
                if stored_dieta and stored_dieta not in vip_restr:
                    vip_restr.append(stored_dieta)
                vip_restr.extend(stored_rejected_ing)
                vip_restr = _dedup_preserve_order(
                    [_normalize_item(x) for x in vip_restr if _normalize_item(x)]
                )
                subgroups.append(
                    {
                        "name": f"VIP {display_name}",
                        "restrictions": vip_restr,
                        "is_vip": True,
                    }
                )

        perfil_usuari = _perfil_from_restriccions(restriccions)
        
        alcohol = input_choice(
            _prompt_inline("Begudes amb alcohol? (s/n):"),
            ["si", "no"],
            "si",
            show_options_in_prompt=False,
        )
        
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
        resultats = retriever.recuperar_casos_similars(problema, k=5)

        if not resultats:
            if input_default("Vols provar amb una altra combinació? (s/n)", "s").lower() != 's':
                break
            continue

        opcions_preparades = []
        signatures = set()

        _print_section_line("PROPOSTES DISPONIBLES (Menús Generals recuperats pel Retriever)")

        for res in resultats:
            if len(opcions_preparades) >= 3:
                break
            cas = res.get("cas", {})
            score = res.get("score_final", 0.0)
            sol = cas.get("solucio", {}) or {}
            plats_originals = sol.get("plats", []) or []

            menu_general = copy.deepcopy(plats_originals)
            for plat in menu_general:
                _aplica_restriccions_plat(plat, restriccions, perfil_usuari)

            sig = tuple(
                _get_plat(menu_general, curs).get("nom", "").strip().lower()
                for curs in ("primer", "segon", "postres")
            )
            if sig in signatures:
                continue
            signatures.add(sig)

            opcions_preparades.append(
                {
                    "cas": cas,
                    "score": score,
                    "menu_general": menu_general,
                }
            )

            i = len(opcions_preparades)
            pr = cas.get("problema", {}) or {}
            estil = (pr.get("estil_culinari") or pr.get("estil") or "Estàndard").strip()
            if estil:
                estil = estil[0].upper() + estil[1:]
            print(f"\nOPCIÓ {i} | Estil: {estil or 'Estàndard'} | Afinitat: {score * 100:.0f}%")
            print(_line("-"))

            p1 = _get_plat(menu_general, "primer")
            p2 = _get_plat(menu_general, "segon")
            p3 = _get_plat(menu_general, "postres")

            print(f"   1. {p1.get('nom','—')}")
            print(f"   2. {p2.get('nom','—')}")
            print(f"   3. {p3.get('nom','—')}")

        # 5) Selecció del Cas
        max_opcions = max(1, len(opcions_preparades))
        idx = input_int_default(_prompt_inline(f"Selecció (1-{max_opcions}):"), 1)
        if idx < 1 or idx > len(opcions_preparades):
            idx = 1
        cas_seleccionat = opcions_preparades[idx - 1]["cas"]
        sol = cas_seleccionat["solucio"]

        plats = copy.deepcopy(opcions_preparades[idx - 1]["menu_general"])
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
        _print_section_line("PERSONALITZACIÓ DEL MENÚ")
        suggeriment = estil_culinari if estil_culinari in kb.estils_latents else ""
        estils_opts = sorted(kb.estils_latents.keys())
        estils_txt = " | ".join(estils_opts)
        _print_options(estils_opts, label="Opcions d'estil latent")
        estil_latent = input_default(
            _prompt_inline("Estil latent (ex: tropical):"),
            suggeriment
        ).strip().lower()

        if estil_latent:
            if estil_latent not in kb.estils_latents:
                print(f"\nNo tinc l'estil '{estil_latent}' a la carta.")
                print(f"Opcions disponibles: {estils_txt}")

            intensitat = float(
                input_default(_prompt_inline("Intensitat (0.1 - 0.9):"), "0.5")
            )
            print("\nProcessant adaptació d'estil...")

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
                n_abans = len(ingredients_abans)
                
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

                ingredients_despres = p.get("ingredients", []) or []
                n_despres = len(ingredients_despres)
                
                diferencia = n_despres - n_abans
                if diferencia > 0:
                    increment = diferencia * COST_INGREDIENT_EXTRA
                    preu_actual = float(p.get("preu", 0.0) or 0.0)
                    p["preu"] = preu_actual + increment
                
                
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
                    verbose=False,
                )
                resums.append((etiqueta_short, resum))

            for etiqueta, resum in resums:
                if resum == "cap canvi":
                    text = "Sense canvis."
                elif resum.startswith("+"):
                    text = f"Afegit {resum[1:]}"
                elif "->" in resum:
                    text = f"Substitució {resum}"
                elif resum.startswith("condiment:"):
                    text = f"Condiment: {resum.split(':', 1)[1].strip()}"
                else:
                    text = resum
                print(f"   - {etiqueta.capitalize()}: {text}")

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

        _print_section("TÈCNIQUES I PRESENTACIÓ")
        vol = input_default(_prompt_inline("Toc d'alta cuina? (s/n):"), "n").strip().lower()

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
            print("\nAplicant tècniques...")

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
            
            # --- ACTUALITZACIÓ DE PREUS PER TÈCNIQUES ---
            # Mapegem cada llista de transformacions al seu plat corresponent
            vincle_t = [
                ("Primer", plat1, transf_1), 
                ("Segon", plat2, transf_2), 
                ("Postres", postres, transf_post)
            ]

            for nom_curs, p, llista_t in vincle_t:
                if llista_t and isinstance(llista_t, list):
                    n_tecs = len(llista_t)
                    if n_tecs > 0:
                        # Determinem el cost segons si és alta cuina o només cultural
                        # Si és mode 'mixt' o 'alta', apliquem el preu més alt
                        preu_u = COST_TECNICA_ALTA if (mode_ops in ["alta", "mixt"]) else COST_TECNICA_CULTURAL
                        increment_total = n_tecs * preu_u
                        
                        preu_actual = float(p.get("preu", 0.0) or 0.0)
                        p["preu"] = preu_actual + increment_total
                        
        else:
            print("\nCap tècnica seleccionada.")

        if mode_ops and (transf_1 or transf_2 or transf_post):
            print(f"   - Primer: {_format_techniques(transf_1, kb)}")
            print(f"   - Segon: {_format_techniques(transf_2, kb)}")
            print(f"   - Postres: {_format_techniques(transf_post, kb)}")

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
        
        beguda1, score1, detail1 = recomana_beguda_per_plat(plat1, list(kb.begudes.values()), base_ingredients_list, restriccions_beguda, alcohol, begudes_usades)
        beguda2, score2, detail2 = recomana_beguda_per_plat(plat2, list(kb.begudes.values()), base_ingredients_list, restriccions_beguda, alcohol, begudes_usades)
        beguda_postres, score_postres, detail_postres = recomana_beguda_per_plat(postres, list(kb.begudes.values()), base_ingredients_list, restriccions_beguda, alcohol, begudes_usades)


        # Generació de Text (Gemini) - DESPRÉS de begudes
        if input_default(_prompt_inline("Descripcions més elegants? (s/n):"), "n").lower() == 's':

            plats_llm = [plat1, plat2, postres]
            transf_llm = [transf_1, transf_2, transf_post]

            # begudes per plat, en el mateix ordre
            begudes_llm = [beguda1, beguda2, beguda_postres]

            fitxes = genera_fitxes_menu_llm_1call(
                plats=plats_llm,
                transformacions_per_plat=transf_llm,
                estil_cultural=estil_cultural if te_cultural else None,
                estil_alta=estil_tecnic if te_alta else None,
                servei=servei,
                kb=kb,
                beguda_per_plat=begudes_llm,
                model_gemini= model_gemini,  # o passa-hi el model si el tens accessible; si no, que la funció internament el gestioni
            )

            # Converteix al format que espera la teva funció d'impressió actual
            by_id = {f["id"]: f for f in (fitxes or [])}
            f1 = by_id.get(0, {})
            f2 = by_id.get(1, {})
            f3 = by_id.get(2, {})

            def _map_fitxa(f: dict) -> dict:
                return {
                    "nom_nou": f.get("nom_plat_ca", ""),
                    "descripcio_carta": f.get("descripcio_ca", ""),
                    "presentacio": f.get("presentacio_ca", ""),
                    "beguda_llm": f.get("beguda_recomanada_ca", ""),
                    "notes_tecnniques": f.get("notes_tecnniques_ca", ""),
                    "image_sentence_en": f.get("image_sentence_en", ""),
                }

            info_llm_1 = _map_fitxa(f1)
            info_llm_2 = _map_fitxa(f2)
            info_llm_post = _map_fitxa(f3)


        # 9) Resultat Final
        imprimir_menu_final(
                kb,
                plat1, transf_1, info_llm_1, beguda1, score1,
                plat2, transf_2, info_llm_2, beguda2, score2,
                postres, transf_post, info_llm_post, beguda_postres, score_postres,
                mostrar_logs=False,
            )

        if subgroups:
            _print_section_line("VARIANTS PER A GRUPS")
            plats_base = [plat1, plat2, postres]
            curs_labels = ["Primer", "Segon", "Postres"]

            for grup in subgroups:
                total_restr = set(restriccions) | set(grup.get("restrictions", []))
                perfil_variant = _perfil_from_restriccions(total_restr)
                plats_variant = [copy.deepcopy(p) for p in plats_base]
                canvis = []

                for idx_plat, (plat_base, plat_variant) in enumerate(
                    zip(plats_base, plats_variant)
                ):
                    ingredients = list(plat_variant.get("ingredients", []) or [])
                    prohibits = _prohibits_per_plat(ingredients, total_restr, perfil_variant)
                    if prohibits:
                        adaptat = substituir_ingredients_prohibits(
                            plat_variant,
                            set(prohibits) | set(total_restr),
                            kb,
                            perfil_usuari=perfil_variant,
                        )
                        if isinstance(adaptat, dict):
                            plat_variant = adaptat

                    if (
                        plat_base.get("ingredients", []) != plat_variant.get("ingredients", [])
                        or plat_base.get("nom") != plat_variant.get("nom")
                    ):
                        diffs = _diff_ingredients(
                            list(plat_base.get("ingredients", []) or []),
                            list(plat_variant.get("ingredients", []) or []),
                        )
                        motiu = ", ".join(sorted(prohibits)) if prohibits else ""
                        canvis.append(
                            {
                                "curs": curs_labels[idx_plat],
                                "base": plat_base,
                                "variant": plat_variant,
                                "diffs": diffs,
                                "motiu": motiu,
                            }
                        )

                nom_grup = grup.get("name", "Grup")
                etiqueta_grup = "VIP" if grup.get("is_vip") else "GRUP"
                if not canvis:
                    print(f"[{etiqueta_grup}: {nom_grup}]")
                    print("   Compatible sense necessitat d'adaptació.")
                else:
                    print(f"[{etiqueta_grup}: {nom_grup}]")
                    print("   (!) ADAPTACIÓ APLICADA")
                    for canvi in canvis:
                        plat_base = canvi["base"]
                        plat_variant = canvi["variant"]
                        if plat_base.get("nom") != plat_variant.get("nom") and plat_variant.get("nom"):
                            canvi_txt = f"{plat_base.get('nom','—')} -> {plat_variant.get('nom','—')}"
                        else:
                            diffs = canvi["diffs"]
                            canvi_txt = ", ".join(diffs) if diffs else "canvis d'ingredients"
                        print(f"   - {canvi['curs']}: {canvi_txt}")
                print("")

        # 9.1) CONTROL PRESSUPOSTARI
        print("\n=== CONTROL PRESSUPOSTARI ===")

        target_budget = float(preu_pers or 0.0)

        def _tecnica_cost_unit():
            if mode_ops in ("alta", "mixt"):
                return float(COST_TECNICA_ALTA)
            if mode_ops == "cultural":
                return float(COST_TECNICA_CULTURAL)
            return 0.0

        def _calcula_totals_menu():
            cost_unit = _tecnica_cost_unit()
            tec_count = sum(len(t) for t in (transf_1, transf_2, transf_post) if t)
            tec_total = tec_count * cost_unit
            plats_total = sum(
                float(p.get("preu", 0.0) or 0.0) for p in (plat1, plat2, postres)
            )
            begudes_total = sum(
                float(b.get("preu_cost", 0.0) or 0.0)
                for b in (beguda1, beguda2, beguda_postres)
                if b
            )
            total_actual = (plats_total - tec_total) + begudes_total + tec_total
            return total_actual, plats_total, begudes_total, tec_total

        def _nom_tecnica_local(tecnica):
            if isinstance(tecnica, dict):
                nom = (tecnica.get("display") or tecnica.get("nom") or tecnica.get("name") or "").strip()
                return nom if nom else "tècnica"
            nom = str(tecnica).strip()
            return nom if nom else "tècnica"

        def _recollir_tecniques(cost_unit):
            items = []
            ordre = 0
            for idx_plat, transf in enumerate((transf_1, transf_2, transf_post)):
                if not transf:
                    continue
                for idx_t, t in enumerate(transf):
                    items.append(
                        {
                            "plat_idx": idx_plat,
                            "t_idx": idx_t,
                            "nom": _nom_tecnica_local(t),
                            "cost": cost_unit,
                            "ordre": ordre,
                        }
                    )
                    ordre += 1
            return items

        def _recomana_beguda_premium(
            plat, base_begudes, base_ingredients, restriccions, alcohol, begudes_usades
        ):
            candidates = []
            ing_main, llista_ing = get_ingredient_principal(plat, base_ingredients)

            for row in base_begudes:
                if not passa_filtre_dur(plat, row, begudes_usades):
                    continue
                if not passa_restriccions(row, restriccions, alcohol):
                    continue
                sc, breakdown = score_beguda_per_plat(row, ing_main, llista_ing)
                candidates.append((row, sc, breakdown))

            if not candidates:
                return None, None, None

            candidates.sort(key=lambda x: x[1], reverse=True)
            top = candidates[:5]

            def _preu_beguda(row):
                try:
                    return float(row.get("preu_cost", 0.0) or 0.0)
                except (TypeError, ValueError):
                    return 0.0

            top.sort(key=lambda x: _preu_beguda(x[0]), reverse=True)
            beguda, score, detail = top[0]
            if beguda is not None:
                begudes_usades.add(beguda.get("id"))
            return beguda, score, detail

        current_total, _, _, _ = _calcula_totals_menu()
        print(f"Objectiu: {target_budget:.2f}€ | Actual: {current_total:.2f}€")

        diff = current_total - target_budget
        canvis_aplicats = False

        if diff > 5:
            resposta = input_default(
                f"El preu supera el pressupost en {diff:.2f}€. Voleu que ajusti automàticament les tècniques per reduir costos mantenint l'estil del menú? (s/n)",
                "n",
            ).strip().lower()
            if resposta == "s":
                cost_unit = _tecnica_cost_unit()
                eliminades = 0
                while True:
                    current_total, _, _, _ = _calcula_totals_menu()
                    if current_total <= target_budget:
                        break
                    if cost_unit <= 0:
                        break
                    items = _recollir_tecniques(cost_unit)
                    if not items:
                        break
                    item = max(items, key=lambda x: (x["cost"], x["ordre"]))
                    transf_llista = [transf_1, transf_2, transf_post]
                    plats_llista = [plat1, plat2, postres]
                    transf_sel = transf_llista[item["plat_idx"]]
                    if not transf_sel:
                        break
                    transf_sel.pop(item["t_idx"])
                    plat_sel = plats_llista[item["plat_idx"]]
                    preu_actual = float(plat_sel.get("preu", 0.0) or 0.0)
                    plat_sel["preu"] = max(0.0, preu_actual - item["cost"])
                    eliminades += 1
                    current_total, _, _, _ = _calcula_totals_menu()
                    print(
                        f" > Eliminant tècnica '{item['nom']}' (-{item['cost']:.2f}€)... Nou total: {current_total:.2f}€"
                    )
                if eliminades > 0:
                    canvis_aplicats = True
        elif diff < -10:
            resposta = input_default(
                f"Tens un marge de {abs(diff):.2f}€. Vols millorar l'experiència amb un Maridatge Prèmium (Vins de gamma alta)? (s/n)",
                "n",
            ).strip().lower()
            if resposta == "s":
                print(" > Aplicant selecció de Sommelier Prèmium...")
                begudes_usades_premium = set()
                base_begudes = list(kb.begudes.values())
                b1, s1, _ = _recomana_beguda_premium(
                    plat1,
                    base_begudes,
                    base_ingredients_list,
                    restriccions_beguda,
                    alcohol,
                    begudes_usades_premium,
                )
                b2, s2, _ = _recomana_beguda_premium(
                    plat2,
                    base_begudes,
                    base_ingredients_list,
                    restriccions_beguda,
                    alcohol,
                    begudes_usades_premium,
                )
                b3, s3, _ = _recomana_beguda_premium(
                    postres,
                    base_begudes,
                    base_ingredients_list,
                    restriccions_beguda,
                    alcohol,
                    begudes_usades_premium,
                )

                if b1 is not None:
                    beguda1 = b1
                    score1 = s1
                    canvis_aplicats = True
                if b2 is not None:
                    beguda2 = b2
                    score2 = s2
                    canvis_aplicats = True
                if b3 is not None:
                    beguda_postres = b3
                    score_postres = s3
                    canvis_aplicats = True
        elif abs(diff) <= 5:
            print("El menú s'ajusta perfectament al pressupost objectiu.")
        else:
            print("El menú queda lleugerament per sota del pressupost, però dins d'un marge acceptable.")

        if canvis_aplicats:
            current_total, _, _, _ = _calcula_totals_menu()

            def _nom_plat_final(plat, info_llm):
                nom = (info_llm or {}).get("nom_nou") or plat.get("nom") or "—"
                nom = str(nom).strip()
                return nom if nom else "—"

            def _nom_beguda_final(beguda):
                if beguda:
                    nom = (beguda.get("nom") or "—").strip()
                    return nom if nom else "—"
                return "—"

            imprimir_menu_final(
                kb,
                plat1, transf_1, info_llm_1, beguda1, score1,
                plat2, transf_2, info_llm_2, beguda2, score2,
                postres, transf_post, info_llm_post, beguda_postres, score_postres,
                mostrar_logs=False,
            )

            print("\nResum final del menú:")
            print(
                f" - Primer: {_nom_plat_final(plat1, info_llm_1)} | Beguda: {_nom_beguda_final(beguda1)}"
            )
            print(
                f" - Segon: {_nom_plat_final(plat2, info_llm_2)} | Beguda: {_nom_beguda_final(beguda2)}"
            )
            print(
                f" - Postres: {_nom_plat_final(postres, info_llm_post)} | Beguda: {_nom_beguda_final(beguda_postres)}"
            )
            print(f"Total final: {current_total:.2f}€")
        # 9.2) IMATGE DEL MENÚ (DESPRÉS de pressupost i variants)
        vol_imatge = input_default(_prompt_inline("Generar imatge del menú final? (s/n):"), "n").strip().lower()
        if vol_imatge == "s":
            # 1) construeix ambient
            ambient = _resum_ambient_esdeveniment(
                tipus_esdeveniment=tipus_esdeveniment,
                temporada=temporada,
                espai="interior",
                formalitat = "formal" if servei == "assegut" else "informal",
            )

            # 2) IMPORTANT: aquí has de passar la llista de fitxes (no plats_info)
            # Si ja tens info_llm_1/info_llm_2/info_llm_post en format fitxa, fes això:
            fitxes_menu = [info_llm_1, info_llm_2, info_llm_post]

            prompt_imatge = construir_prompt_imatge_menu(
                ambient=ambient,
                fitxes_menu=fitxes_menu,
                servei = servei,
                allow_dessert_glass=True,
                incloure_decoracio=True,
                incloure_begudes=False,
            )
            prompt_imatge = _safe_ascii_prompt(prompt_imatge)

            genera_imatge_menu_hf_o_prompt(prompt_imatge, output_path="menu_event.png")


        # 10) FASE REVISE (Dual Memory)
        gestor_revise = GestorRevise()
        cas_proposat = {
            "problema": problema,
            "solucio": {"primer": plat1, "segon": plat2, "postres": postres}
        }
        resultat_avaluacio = gestor_revise.avaluar_proposta(cas_proposat, user_id)
        print(f"Resultat global intern: {resultat_avaluacio['tipus_resultat']}")

        # 11) FASE RETAIN (Política de memòria)
        _print_section_line("MEMÒRIA DEL SISTEMA")
        print("Avaluant si aquest cas s'ha de recordar per al futur...")
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

        if input_default(_prompt_inline("Vols preparar un altre menú? (s/n):"), "n").lower() != 's':
            print("\nD'acord! Bon profit i fins aviat.")
            break

if __name__ == "__main__":
    main()