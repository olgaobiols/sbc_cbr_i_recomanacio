import random
import os
from typing import List, Dict, Set, Any, Optional
import google.generativeai as genai
import requests
from collections import defaultdict
import re


# Importem la lògica latent ja adaptada a KB
from operador_ingredients import adaptar_plat_a_estil_latent

# Configuració API (Idealment en un .env, però mantenim la teva estructura)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    # Per evitar que peti si no tens la clau posada mentre proves altres coses
    print("\n[AVÍS] Falta GEMINI_API_KEY. Les funcions LLM no funcionaran.\n")
else:
    genai.configure(api_key=API_KEY)

GEMINI_MODEL_NAME = "gemini-2.5-flash"
try:
    model_gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)
except:
    model_gemini = None

# ---------------------------------------------------------------------
# 1. FUNCIONS AUXILIARS DE SCORE (Adaptades a KB)
# ---------------------------------------------------------------------
def _split_pipe(val: Any) -> List[str]:
    if not val:
        return []
    if isinstance(val, list):
        return [str(x).strip() for x in val if str(x).strip()]
    return [x.strip() for x in str(val).split("|") if x.strip()]

def _split_priority(val: Any) -> List[str]:
    # format: "sauce>other>fruit"
    if not val:
        return []
    return [x.strip() for x in str(val).split(">") if x.strip()]

def _rank_from_priority(priority_list: List[str]) -> Dict[str, int]:
    # menor = millor
    return {name: i for i, name in enumerate(priority_list)}

def _estat_ingredient(info_ing: Dict) -> str:
    # 1) si la KB ja té estat, el respectem
    for k in ("estat", "state", "aplicable_estat", "physical_state", "form"):
        v = (info_ing.get(k) or "").strip().lower()
        if v:
            if "powder" in v or "pols" in v:
                return "powder"
            if "semi" in v:
                return "semi_liquid"
            if "liquid" in v or "líquid" in v:
                return "liquid"
            if "solid" in v or "sòlid" in v:
                return "solid"

    macro = (info_ing.get("macro_category") or info_ing.get("categoria_macro") or "").lower()
    fam = (info_ing.get("family") or info_ing.get("familia") or "").lower()
    name = (info_ing.get("nom_ingredient") or info_ing.get("ingredient_name") or info_ing.get("name") or "").lower()

    # 2) Overrides “obvis” per família/nom
    if name in ("water", "aigua"):
        return "liquid"

    # Pols clares
    if fam in ("salt", "sugar", "sweetener", "spice") or macro in ("spice", "seasoning"):
        return "powder"

    # Lactis: distingim formatges vs cremes/iogurts
    if macro == "dairy":
        if "cheese" in fam:       # dairy_cheese, soft_dairy_cheese, etc.
            return "solid"
        if fam in ("dairy_cream", "dairy_yogurt"):
            return "semi_liquid"
        # fallback per lacti desconegut:
        return "solid"

    # Salses / brous / reduccions / vinagres
    if macro in ("sauce", "sweet_sauce") or fam in ("cooking_stock", "wine_reduction", "acetic", "asian_acetic"):
        return "liquid"

    # Greixos: oli = liquid, greix sòlid (mantega) depèn de family si ho tens
    if macro == "fat":
        return "liquid"

    # Sweet/sweetener: normalment sucre (powder) o xarop (liquid) -> si no tens family, millor powder
    if macro in ("sweetener", "sweet"):
        return "powder"

    # "other"/"emulsion": sol ser semi_liquid SI és una emulsió real (maionesa), però l’aigua no hauria de ser emulsion
    if fam == "emulsion":
        return "semi_liquid"

    return "solid"

def _norm_macro(m: str) -> str:
    m = (m or "").strip().lower()
    mapa = {
        "plant_vegetal": "vegetable",
        "green_leaf": "vegetable",   # si mai t'arriba així
        "vegetable": "vegetable",
        "fruit": "fruit",
        "grain": "grain",
        "protein_animal": "protein",
        "meat_cured": "protein",
        "egg": "protein",
        "sauce": "sauce",
        "fat": "fat",
        "dairy": "dairy",
    }
    return mapa.get(m, m)


def _get_info_ingredients_plat(plat: Dict, kb: Any) -> List[Dict]:
    """Recupera la info de tots els ingredients del plat usant la KB.
    Si algun ingredient no existeix a KB, retorna un dict mínim amb el nom.
    """
    infos = []
    for nom in plat.get("ingredients", []) or []:
        info = kb.get_info_ingredient(nom) if kb is not None else None
        if info:
            infos.append(info)
        else:
            # Fallback mínim: així no perdem ingredients
            infos.append({"nom_ingredient": nom, "categoria_macro": "", "familia": ""})
    return infos


def _llista_ingredients_aplicables(tecnica_row: Dict, info_ings: List[Dict]) -> list[str]:
    aplica_estat = set(_split_pipe(tecnica_row.get("aplica_estat")))
    aplica_macro = {_norm_macro(x) for x in _split_pipe(tecnica_row.get("aplica_macro"))}
    aplica_family = set(_split_pipe(tecnica_row.get("aplica_family")))

    evita_macro = set(_split_pipe(tecnica_row.get("evita_macro")))
    evita_family = set(_split_pipe(tecnica_row.get("evita_family")))

    # “aigua” fora si hi ha alternatives
    alternatives_no_portadores = []
    for info in info_ings:
        nom0 = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if nom0 and not _es_ingredient_buit_o_portador(nom0, info):
            alternatives_no_portadores.append(info)

    possibles = []
    for info in info_ings:
        nom = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if not nom:
            continue

        if _es_ingredient_buit_o_portador(nom, info) and alternatives_no_portadores:
            continue

        macro = _norm_macro(info.get("macro_category") or info.get("categoria_macro") or "")
        fam = (info.get("family") or info.get("familia") or "").lower()
        estat = _estat_ingredient(info)

        if macro in evita_macro:
            continue
        if fam in evita_family:
            continue
        if aplica_estat and estat not in aplica_estat:
            continue
        if aplica_macro and macro not in aplica_macro:
            continue
        # family NO és dur (com al teu _troba_ingredient_aplicable), però si vols fer-la dura aquí ho podem canviar.
        possibles.append(nom)

    return possibles

def _compta_compat_per_ingredients(tecniques_raw: list[dict], base_tecnniques: dict, info_ings: list[dict]) -> dict:
    """
    Per cada ingredient del plat, compta en quantes de les tècniques seleccionades
    podria ser objectiu. Com més baix, més 'escàs' -> l'hem de protegir.
    """
    counts = {}
    for info in info_ings:
        nom = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if nom:
            counts[nom] = 0

    for r in tecniques_raw:
        tec_row = base_tecnniques.get(r["nom"]) or {}
        for ing in _llista_ingredients_aplicables(tec_row, info_ings):
            if ing in counts:
                counts[ing] += 1

    return counts

def _troba_ingredient_aplicable(
    tecnica_row: Dict,
    plat: Dict,
    info_ings: List[Dict],
    ingredients_usats: Set[str],
    compat_counts: Optional[Dict[str, int]] = None,   # <-- AFEGIT
):

    aplica_estat = set(_split_pipe(tecnica_row.get("aplica_estat")))
    aplica_macro = {_norm_macro(x) for x in _split_pipe(tecnica_row.get("aplica_macro"))}
    aplica_family = set(_split_pipe(tecnica_row.get("aplica_family")))

    evita_macro = set(_split_pipe(tecnica_row.get("evita_macro")))
    evita_family = set(_split_pipe(tecnica_row.get("evita_family")))

    prio_macro = _rank_from_priority(_split_priority(tecnica_row.get("prioritat_macro")))
    prio_family = _rank_from_priority(_split_priority(tecnica_row.get("prioritat_family")))
    # abans del loop candidates
    alternatives_no_portadores = []
    for info in info_ings:
        nom0 = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if not nom0:
            continue
        if _es_ingredient_buit_o_portador(nom0, info):
            continue
        alternatives_no_portadores.append(info)

    # candidates scored
    candidates = []

    for info in info_ings:
        nom = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if not nom or nom in ingredients_usats:
            continue

        # si és water/aigua i hi ha altres opcions, el descartem
        if _es_ingredient_buit_o_portador(nom, info) and alternatives_no_portadores:
            continue


        macro_raw = (info.get("macro_category") or info.get("categoria_macro") or "")
        macro = _norm_macro(macro_raw)
        fam = (info.get("family") or info.get("familia") or "").lower()
        estat = _estat_ingredient(info)

        # filtres d'exclusió
        if macro in evita_macro:
            continue
        if fam in evita_family:
            continue

        # filtre d'aplicabilitat (si el camp és buit, no obliga)
        if aplica_estat and estat not in aplica_estat:
            continue
        if aplica_macro and macro not in aplica_macro:
            continue

        # family: el considerem "bonus", no filtre dur (per ser robustos)
        family_bonus = 1 if (aplica_family and fam in aplica_family) else 0

        # prioritat: com més baix, millor
        macro_rank = prio_macro.get(macro, 999)
        fam_rank = prio_family.get(fam, 999)

        # score global (tu pots ajustar pesos)
        score = 0
        score += 5  # passa filtres
        score += 3 * family_bonus
        score += max(0, 10 - macro_rank) if macro_rank < 999 else 0
        score += max(0, 6 - fam_rank) if fam_rank < 999 else 0
        # Penalització "anti-robatori": protegim ingredients escassos
        if compat_counts is not None:
            c = compat_counts.get(nom, 0)
            if c <= 1:
                score -= 4
            elif c == 2:
                score -= 2

        candidates.append((score, nom, macro, fam, estat))

    if candidates:
        candidates.sort(reverse=True, key=lambda x: x[0])
        _, nom, macro, fam, estat = candidates[0]
        ingredients_usats.add(nom)
        return f"l'ingredient '{nom}' ({estat}, {macro}, {fam})", nom

    # si no hi ha cap ingredient, retornem None (la tècnica queda “sense objectiu”)
    return "un element del plat", None



def substituir_ingredient(
    plat,
    tipus_cuina,
    kb,
    mode="regles",
    intensitat=0.4,
    ingredients_estil_usats=None,
    perfil_usuari: Optional[Dict] = None,
    parelles_prohibides: Optional[Set[str]] = None,
):
    """
    Wrapper que connecta amb l'Operador d'Ingredients Refactoritzat.
    """
    if mode == "latent":
        # Ara passem 'kb' i no llistes crues
        return adaptar_plat_a_estil_latent(
            plat=plat,
            nom_estil=tipus_cuina,
            kb=kb, # <--- CLAU: Passem la KB
            base_estils_latents=kb.estils_latents,
            intensitat=intensitat,
            ingredients_estil_usats=ingredients_estil_usats,
            perfil_usuari=perfil_usuari,
            parelles_prohibides=parelles_prohibides,
        )
    return plat


def _completa_fins_a_n(
    transformacions: list[dict],
    plat: dict,
    nom_estil: str,
    base_estils: dict,
    base_tecnniques: dict,
    kb,
    n_objectiu: int,
    tecniques_ja_usades: set,
    min_score: int,
    debug: bool = False,
) -> list[dict]:
    """
    Si n'hi ha menys de n_objectiu, intenta afegir més tècniques relaxant restriccions:
      1) baixa min_score de forma suau
      2) permet repeticions de tècnica (només si és l'únic que encaixa)
    """
    if len(transformacions) >= n_objectiu:
        return transformacions

    # Per evitar duplicats exactes
    noms_ja = {t["nom"] for t in transformacions if t.get("nom")}

    # intent 1: baixar llindar
    for relax in (min_score - 1, min_score - 2, min_score - 3):
        if len(transformacions) >= n_objectiu:
            break
        if relax < 1:
            continue

        extra = triar_tecniques_per_plat(
            plat=plat,
            nom_estil=nom_estil,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            max_tecniques=n_objectiu,   # volem tenir marge
            min_score=relax,
            tecniques_ja_usades=tecniques_ja_usades,
            debug=debug,
        )

        for t in extra:
            if len(transformacions) >= n_objectiu:
                break
            if t["nom"] in noms_ja:
                continue
            transformacions.append(t)
            noms_ja.add(t["nom"])

    # intent 2 (últim recurs): permetre repetir tècniques del menú si cal
    if len(transformacions) < n_objectiu:
        extra2 = triar_tecniques_per_plat(
            plat=plat,
            nom_estil=nom_estil,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            max_tecniques=n_objectiu,
            min_score=max(1, min_score - 3),
            tecniques_ja_usades=set(),  # <- elimina penalització menu
            debug=debug,
        )
        for t in extra2:
            if len(transformacions) >= n_objectiu:
                break
            if t["nom"] in noms_ja:
                continue
            transformacions.append(t)
            noms_ja.add(t["nom"])

    return transformacions





# ---------------------------------------------------------------------
#  OPERADOR 2: APLICAR TÈCNIQUES A UN PLAT
# ---------------------------------------------------------------------
def llista_tecniques_applicables_per_ingredient(
    plat: dict,
    kb,
    base_tecnniques: dict,
    inclou_curs: bool = True,
    ordenar_per: str = "nom",  # "nom" o "match"
    debug: bool = False,
) -> dict:
    """
    Retorna un dict:
      { ingredient_nom: [ {nom_tecnica, display, match, motius_ok, motius_no} , ... ] }
    on match = nombre de dimensions que matxegen (estat/macro/family/curs)
    i només inclou tècniques que passen exclusions i que NO fallen cap filtre dur.

    NOTE:
    - Considero 'aplica_*' com a filtre dur NOMÉS si el camp no és buit.
    - 'aplica_family' també el faig filtre dur aquí, perquè tu demanes "es pot aplicar o no".
      (Si prefereixes family com a "bonus", t’ho canvio fàcil.)
    """

    curs = (plat.get("curs", "") or "").strip().lower()
    info_ings = _get_info_ingredients_plat(plat, kb)

    # index per nom d'ingredient (tal com surt de KB)
    result = defaultdict(list)

    for info in info_ings:
        ing_nom = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if not ing_nom:
            continue

        macro_raw = (info.get("macro_category") or info.get("categoria_macro") or "")
        macro = _norm_macro(macro_raw)
        fam = (info.get("family") or info.get("familia") or "").strip().lower()
        estat = _estat_ingredient(info)

        for nom_tecnica, tec_row in base_tecnniques.items():
            if tec_row is None:
                continue

            # camps tècnica
            aplica_estat = set(_split_pipe(tec_row.get("aplica_estat")))
            aplica_macro = {_norm_macro(x) for x in _split_pipe(tec_row.get("aplica_macro"))}
            aplica_family = set(_split_pipe(tec_row.get("aplica_family")))
            aplica_curs = set(_split_pipe(tec_row.get("aplicable_curs") or ""))

            evita_macro = set(_split_pipe(tec_row.get("evita_macro")))
            evita_family = set(_split_pipe(tec_row.get("evita_family")))

            motius_ok = []
            motius_no = []

            # 1) exclusions (dures)
            if macro in evita_macro:
                motius_no.append(f"EXCLÒS: macro '{macro}' ∈ evita_macro")
                continue
            if fam and fam in evita_family:
                motius_no.append(f"EXCLÒS: family '{fam}' ∈ evita_family")
                continue

            # 2) aplica_curs (si vols)
            if inclou_curs and aplica_curs:
                if curs in aplica_curs:
                    motius_ok.append(f"curs OK ({curs})")
                else:
                    motius_no.append(f"curs NO ({curs}) no ∈ {sorted(aplica_curs)}")
                    continue

            # 3) estat (dur si tècnica defineix aplica_estat)
            if aplica_estat:
                if estat in aplica_estat:
                    motius_ok.append(f"estat OK ({estat})")
                else:
                    motius_no.append(f"estat NO ({estat}) no ∈ {sorted(aplica_estat)}")
                    continue

            # 4) macro (dur si tècnica defineix aplica_macro)
            if aplica_macro:
                if macro in aplica_macro:
                    motius_ok.append(f"macro OK ({macro})")
                else:
                    motius_no.append(f"macro NO ({macro}) no ∈ {sorted(aplica_macro)}")
                    continue

            # 5) family (dur si tècnica defineix aplica_family)
            if aplica_family:
                if fam in aplica_family:
                    motius_ok.append(f"family OK ({fam})")
                else:
                    motius_no.append(f"family NO ({fam}) no ∈ {sorted(aplica_family)}")
                    continue

            # match score (quantes dimensions han matxejat, només sobre dimensions que existien)
            match = 0
            if inclou_curs and aplica_curs:
                match += 1
            if aplica_estat:
                match += 1
            if aplica_macro:
                match += 1
            if aplica_family:
                match += 1

            result[ing_nom].append({
                "nom_tecnica": nom_tecnica,
                "display": tec_row.get("display_nom", nom_tecnica),
                "match": match,
                "motius_ok": motius_ok,
                "motius_no": motius_no,
                "categoria": (tec_row.get("categoria") or "").lower(),
                "impacte_textura": tec_row.get("impacte_textura", ""),
                "impacte_sabor": tec_row.get("impacte_sabor", ""),
            })

        # ordenar per ingredient
        if ordenar_per == "match":
            result[ing_nom].sort(key=lambda x: (x["match"], x["nom_tecnica"]), reverse=True)
        else:
            result[ing_nom].sort(key=lambda x: x["nom_tecnica"])

        if debug:
            print(f"[MAP] {ing_nom}: {len(result[ing_nom])} tècniques aplicables")

    return dict(result)

def triar_tecniques_2_operadors_per_plat(
    plat: dict,
    mode: str,  # "cultural", "alta", "mixt"
    estil_cultural: str | None,
    estil_alta: str | None,
    base_estils: dict,
    base_tecnniques: dict,
    kb,
    tecniques_ja_usades: set,
    min_score: int = 5,
    ingredients_usats_plat = set(),
    debug: bool = False,
) -> list[dict]:
    """
    Retorna EXACTAMENT (si es pot) 2 transformacions totals segons el mode:
      - cultural: 2 culturals
      - alta: 2 alta cuina
      - mixt: 1 cultural + 1 alta cuina
    """
    mode = (mode or "").strip().lower()
    transf = []

    if mode == "mixt":
        # 1 cultural
        if estil_cultural:
            t_c = triar_tecniques_per_plat(
                plat=plat,
                nom_estil=estil_cultural,
                base_estils=base_estils,
                base_tecnniques=base_tecnniques,
                kb=kb,
                max_tecniques=1,
                min_score=min_score,
                tecniques_ja_usades=tecniques_ja_usades,
                ingredients_usats_global=ingredients_usats_plat,
                debug=debug,
            )
            if t_c:
                transf.extend(t_c)
                tecniques_ja_usades.update(x["nom"] for x in t_c)

        # 1 alta
        if estil_alta:
            t_a = triar_tecniques_per_plat(
                plat=plat,
                nom_estil=estil_alta,
                base_estils=base_estils,
                base_tecnniques=base_tecnniques,
                kb=kb,
                max_tecniques=1,
                min_score=min_score,
                tecniques_ja_usades=tecniques_ja_usades,
                ingredients_usats_global=ingredients_usats_plat,
                debug=debug,
            )
            if t_a:
                transf.extend(t_a)
                tecniques_ja_usades.update(x["nom"] for x in t_a)

        # completa fins 2 (prioritat: el que falti segons disponibilitat)
        # completa fins 2 amb quota: primer intenta el que falti
        if len(transf) < 2:
            falta = 2 - len(transf)

            # 1) Si NO hem aconseguit cultural, relaxa cultural abans de saltar a alta
            # 1) Si NO hem aconseguit cultural, relaxa cultural abans de saltar a alta
            if estil_cultural:
                tecs_culturals_str = (base_estils.get(estil_cultural, {}) or {}).get("tecnniques_clau", "") or ""
                tecs_culturals = {t.strip() for t in tecs_culturals_str.split("|") if t.strip()}
                te_cultural_real = any((x.get("nom") or "").strip() in tecs_culturals for x in transf)
            else:
                te_cultural_real = False

            if estil_cultural and not te_cultural_real:
                extra_c = triar_tecniques_per_plat(
                    plat=plat,
                    nom_estil=estil_cultural,
                    base_estils=base_estils,
                    base_tecnniques=base_tecnniques,
                    kb=kb,
                    max_tecniques=falta,
                    min_score=max(1, min_score - 2),   # relax suau
                    tecniques_ja_usades=tecniques_ja_usades,
                    debug=debug,
                )
                for t in extra_c:
                    if len(transf) >= 2:
                        break
                    if t["nom"] not in {x["nom"] for x in transf}:
                        transf.append(t)
                        tecniques_ja_usades.add(t["nom"])

            # 2) Si encara falta, llavors sí: omple amb alta
            if len(transf) < 2 and estil_alta:
                extra_a = triar_tecniques_per_plat(
                    plat=plat,
                    nom_estil=estil_alta,
                    base_estils=base_estils,
                    base_tecnniques=base_tecnniques,
                    kb=kb,
                    max_tecniques=2 - len(transf),
                    min_score=max(1, min_score - 2),
                    tecniques_ja_usades=tecniques_ja_usades,
                    debug=debug,
                )
                for t in extra_a:
                    if len(transf) >= 2:
                        break
                    if t["nom"] not in {x["nom"] for x in transf}:
                        transf.append(t)
                        tecniques_ja_usades.add(t["nom"])

        return transf[:2]

    elif mode == "cultural":
        if not estil_cultural:
            return []
        transf = triar_tecniques_per_plat(
            plat=plat,
            nom_estil=estil_cultural,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            max_tecniques=2,
            min_score=min_score,
            tecniques_ja_usades=tecniques_ja_usades,
            debug=debug,
        )
        transf = _completa_fins_a_n(
            transformacions=transf,
            plat=plat,
            nom_estil=estil_cultural,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            n_objectiu=2,
            tecniques_ja_usades=tecniques_ja_usades,
            min_score=min_score,
            debug=debug,
        )
        tecniques_ja_usades.update(x["nom"] for x in transf)
        return transf[:2]

    else:
        # "alta" (default)
        if not estil_alta:
            return []
        transf = triar_tecniques_per_plat(
            plat=plat,
            nom_estil=estil_alta,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            max_tecniques=2,
            min_score=min_score,
            tecniques_ja_usades=tecniques_ja_usades,
            debug=debug,
        )
        transf = _completa_fins_a_n(
            transformacions=transf,
            plat=plat,
            nom_estil=estil_alta,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            n_objectiu=2,
            tecniques_ja_usades=tecniques_ja_usades,
            min_score=min_score,
            debug=debug,
        )
        tecniques_ja_usades.update(x["nom"] for x in transf)
        return transf[:2]


def triar_tecniques_2_operadors_per_menu(
    plats: list[dict],
    mode: str,                 # "cultural" | "alta" | "mixt"
    estil_cultural: str | None,
    estil_alta: str | None,
    base_estils: dict,
    base_tecnniques: dict,
    kb,
    min_score: int = 5,
    debug: bool = False,
) -> list[list[dict]]:
    tecniques_ja_usades = set()
    result = []

    for plat in plats:
        transf = triar_tecniques_2_operadors_per_plat(
            plat=plat,
            mode=mode,
            estil_cultural=estil_cultural,
            estil_alta=estil_alta,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            kb=kb,
            tecniques_ja_usades=tecniques_ja_usades,
            min_score=min_score,
            debug=debug,
        )
        result.append(transf)

    return result

def _score_tecnica_per_plat(tecnica_row: Dict, plat: Dict, info_ings: List[Dict]) -> int:
    curs = (plat.get("curs", "") or "").lower()
    categoria_tecnica = (tecnica_row.get("categoria") or "").lower()

    aplica_estat = set(_split_pipe(tecnica_row.get("aplica_estat")))
    aplica_macro = {_norm_macro(x) for x in _split_pipe(tecnica_row.get("aplica_macro"))}
    aplica_family = set(_split_pipe(tecnica_row.get("aplica_family")))
    aplica_curs = set(_split_pipe(tecnica_row.get("aplicable_curs") or ""))  # safe
 

    score = 0

    # curs (si existeix)
    if aplica_curs and curs in aplica_curs:
        score += 2

    # match ingredient-level: comptem si existeix com a mínim 1 ingredient compatible
    any_estat = False
    any_macro = False
    any_family = False

    for info in info_ings:
        macro_raw = (info.get("macro_category") or info.get("categoria_macro") or "")
        macro = _norm_macro(macro_raw)
        fam = (info.get("family") or info.get("familia") or "").lower()
        estat = _estat_ingredient(info)

        if aplica_estat and estat in aplica_estat:
            any_estat = True
        if aplica_macro and macro in aplica_macro:
            any_macro = True
        if aplica_family and fam in aplica_family:
            any_family = True

    if any_macro:
        score += 4
    if any_estat:
        score += 3
    if any_family:
        score += 2

    # bonus molecular si hi ha líquid/semi-líquid al plat
    if categoria_tecnica == "molecular":
        if any(_estat_ingredient(i) in ("liquid", "semi_liquid") for i in info_ings):
            score += 2

    return score

def debug_tecniques_applicables_per_ingredient(plat, kb, base_tecnniques):
    info_ings = _get_info_ingredients_plat(plat, kb)
    curs = (plat.get("curs", "") or "").lower()

    out = {}
    for info in info_ings:
        ing = info.get("nom_ingredient") or info.get("ingredient_name") or info.get("name")
        if not ing:
            continue
        macro = _norm_macro(info.get("macro_category") or info.get("categoria_macro") or "")
        fam = (info.get("family") or info.get("familia") or "").lower()
        estat = _estat_ingredient(info)

        aplicables = []
        for nom_tecnica, tec in base_tecnniques.items():
            aplica_estat = set(_split_pipe(tec.get("aplica_estat")))
            aplica_macro = {_norm_macro(x) for x in _split_pipe(tec.get("aplica_macro"))}
            aplica_family = set(_split_pipe(tec.get("aplica_family")))
            aplica_curs = set(_split_pipe(tec.get("aplicable_curs") or ""))

            evita_macro = set(_split_pipe(tec.get("evita_macro")))
            evita_family = set(_split_pipe(tec.get("evita_family")))

            if macro in evita_macro:
                continue
            if fam in evita_family:
                continue
            if aplica_curs and curs not in aplica_curs:
                continue
            if aplica_estat and estat not in aplica_estat:
                continue
            if aplica_macro and macro not in aplica_macro:
                continue
            # aquí decideixes si family és dur o no:
            if aplica_family and fam not in aplica_family:
                continue

            aplicables.append(nom_tecnica)

        out[ing] = sorted(aplicables)

    return out


def triar_tecniques_per_plat(
    plat,
    nom_estil,
    base_estils,
    base_tecnniques,
    kb,
    max_tecniques=2,
    min_score=5,
    tecniques_ja_usades=None,
    ingredients_usats_global: Optional[Set[str]] = None,  # <-- AFEGIT
    debug=False,
):
    """
    Selecciona fins a `max_tecniques` tècniques de l'estil donat que encaixen
    amb el plat. Per a cada tècnica, associa un ingredient (o el curs) on aplicar-la.

    NOVETATS:
      - Penalitza tècniques ja usades en altres plats del menú (si es passa
        el paràmetre `tecniques_ja_usades` com un set de noms).
      - Prioritza la diversitat de textures: evita tècniques massa semblants
        dins del mateix plat (impacte_textura gairebé igual).
      - Manté la restricció de no repetir ingredients entre tècniques,
        mitjançant `ingredients_usats`.

    Retorna:
        list[dict]: cada element té:
            - nom: id de la tècnica (p.ex. 'esferificacio')
            - display: nom bonic (p.ex. 'Esferificació')
            - objectiu_frase: text tipus "el líquid 'salsa de soja'"
            - objectiu_ingredient: nom cru de l'ingredient o None
            - descripcio: text de la tècnica
            - impacte_textura: llista de tags de textura
            - impacte_sabor: llista de tags de sabor
    """
    if tecniques_ja_usades is None:
        tecniques_ja_usades = set()

    nom_plat = plat.get("nom", "<sense_nom>")
    info_ings = _get_info_ingredients_plat(plat,kb)

    estil_row = base_estils.get(nom_estil)
    if estil_row is None:
        if debug:
            print(f"[TEC] Estil '{nom_estil}' no trobat per al plat '{nom_plat}'.")
        return []

    tipus_estil = (estil_row.get("tipus") or "").strip().lower()


    tecnniques_str = estil_row.get("tecnniques_clau", "")
    if not tecnniques_str:
        if debug:
            print(f"[TEC] L'estil '{nom_estil}' no té tecnniques_clau definides.")
        return []

    tecniques_candidats = tecnniques_str.split("|")

    # 1) Scorem totes les tècniques candidates
    scored = []
    for nom_tecnica in tecniques_candidats:
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        base_score = _score_tecnica_per_plat(tec_row, plat, info_ings)

        # Penalització suau per tècniques ja usades en altres plats del menú
        if nom_tecnica in tecniques_ja_usades:
            base_score -= 2  # ajustable: 2 = penalitza però no prohibeix

        if debug:
            print(f"[SCORE] Plat '{nom_plat}', tècnica '{nom_tecnica}' → {base_score}")

        if base_score >= min_score:
            # Pre-check: la tècnica ha de tenir com a mínim 1 ingredient objectiu possible
            tmp_usats = set()
            _, obj_ing_test = _troba_ingredient_aplicable(
                tec_row, plat, info_ings, tmp_usats, compat_counts=None
            )
            if obj_ing_test is None:
                if debug:
                    print(f"[SKIP] '{nom_tecnica}' sense objectiu aplicable a '{nom_plat}'")
                continue

            scored.append({"nom": nom_tecnica, "score": base_score})

    if not scored:
        if debug:
            print(f"[TEC] Cap tècnica de '{nom_estil}' supera el mínim per a '{nom_plat}'.")
        return []

    # 2) Ordenem de millor a pitjor
    scored.sort(key=lambda x: x["score"], reverse=True)

    # 2b) Selecció amb diversitat de textures dins del plat
    def _textures_de_tecnica(nom_tecnica: str) -> set:
        row = base_tecnniques.get(nom_tecnica) or {}
        return {t for t in (row.get("impacte_textura") or "").split("|") if t}

    seleccionades_raw = []
    # Fem servir un pool una mica més ampli que max_tecniques per poder triar diversitat
    pool_limit = max(len(scored), max_tecniques * 3)
    pool = scored[:pool_limit]

    for cand in pool:
        if len(seleccionades_raw) >= max_tecniques:
            break

        nom_tecnica = cand["nom"]
        textures_cand = _textures_de_tecnica(nom_tecnica)

        # Evitem tècniques molt semblants en textura a les ja triades
        massa_semblant = False
        for sel in seleccionades_raw:
            textures_sel = _textures_de_tecnica(sel["nom"])
            unio = textures_cand | textures_sel
            if not unio:
                continue
            interseccio = textures_cand & textures_sel
            jaccard = len(interseccio) / len(unio)
            # Si la coincidència és molt alta (>= 0.7), les considerem massa semblants
            if jaccard >= 0.7:
                massa_semblant = True
                break

        if massa_semblant:
            continue

        seleccionades_raw.append(cand)

    # Si no hem pogut triar res per diversitat però hi havia candidates, agafem la millor
    if not seleccionades_raw and scored:
        seleccionades_raw = [scored[0]]

    # Comptem quins ingredients són "escassos" entre les tècniques seleccionades,
    # per evitar que una tècnica flexible "robi" l'ingredient que només serveix per una altra.
    compat_counts = _compta_compat_per_ingredients(
        tecniques_raw=seleccionades_raw,
        base_tecnniques=base_tecnniques,
        info_ings=info_ings,
    )

    # 3) Assignem ingredient/curs a cada tècnica
    # 3) Assignem ingredient/curs a cada tècnica (robust: tècniques més restringides primer)
    transformacions = []
    ingredients_usats = ingredients_usats_global if ingredients_usats_global is not None else set()

    # calculem quants ingredients possibles té cada tècnica
    sel_ordenades = []
    for r in seleccionades_raw:
        nom_tecnica = r["nom"]
        tec_row = base_tecnniques.get(nom_tecnica) or {}
        poss = _llista_ingredients_aplicables(tec_row, info_ings)
        sel_ordenades.append((len(poss), r))

    # primer les més “difícils” (menys opcions)
    sel_ordenades.sort(key=lambda x: x[0])

    for _, r in sel_ordenades:
        nom_tecnica = r["nom"]
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        objectiu_frase, obj_ing = _troba_ingredient_aplicable(
            tec_row, plat, info_ings, ingredients_usats, compat_counts=compat_counts
        )

        if obj_ing is None:
            continue

        impacte_textura = [t for t in (tec_row.get("impacte_textura") or "").split("|") if t]
        impacte_sabor = [s for s in (tec_row.get("impacte_sabor") or "").split("|") if s]

        transformacions.append({
            "nom": nom_tecnica,
            "display": tec_row.get("display_nom", nom_tecnica),
            "objectiu_frase": objectiu_frase,
            "objectiu_ingredient": obj_ing,
            "descripcio": tec_row.get("descripcio", ""),
            "impacte_textura": impacte_textura,
            "impacte_sabor": impacte_sabor,
        })


    return transformacions

def _es_ingredient_buit_o_portador(nom: str, info: dict) -> bool:
    """
    Ingredients 'portadors' que sovint no volem com a objectiu d'una tècnica,
    perquè són massa genèrics (aigua, etc.).
    """
    n = (nom or "").strip().lower()
    fam = (info.get("family") or info.get("familia") or "").strip().lower()
    macro = _norm_macro(info.get("macro_category") or info.get("categoria_macro") or "")

    if n in {"water", "aigua"}:
        return True

    # si a la teva KB aigua surt com other/emulsion, també ho capturem
    if macro == "other" and fam in {"emulsion"} and n in {"water", "aigua"}:
        return True

    return False



import json
import re
from typing import Any, Dict, List, Optional

# Reutilitzes el teu model_gemini i _get_info_ingredients_plat si vols (no és obligatori)
# model_gemini = genai.GenerativeModel("gemini-2.5-flash")



def _json_from_text(text: str) -> Optional[dict]:
    if not text:
        return None
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not m:
            return None
        try:
            return json.loads(m.group(0))
        except Exception:
            return None

def _neteja(s: str) -> str:
    s = (s or "").strip()
    s = s.replace("...", "").replace("**", "")
    return s

def ingredient_ca(kb, nom_en: str) -> str:
    info = kb.get_info_ingredient(nom_en) if kb else None
    ca = (info or {}).get("nom_catala") or (info or {}).get("nom_ca")
    return (ca or nom_en).strip()

def ingredients_ca_llista(kb, ingredients_en: list[str]) -> list[str]:
    return [ingredient_ca(kb, x) for x in ingredients_en]


def genera_fitxes_menu_llm_1call(
    plats: List[dict],
    transformacions_per_plat: List[List[dict]],
    estil_cultural: Optional[str],
    estil_alta: Optional[str],
    servei: str,
    kb: Any,
    beguda_per_plat: Optional[List[Optional[dict]]] = None,
    model_gemini=None,
) -> List[Dict[str, Any]]:
    """
    UNA SOLA CRIDA LLM PER TOT EL MENÚ.
    - Ingredients_ca es construeixen amb KB (nom_catala) -> el LLM NO els inventa ni els tradueix.
    - El LLM retorna: nom_plat_ca, descripcio_ca, presentacio_ca, notes_tecnniques_ca,
      beguda_recomanada_ca, i una frase anglesa per imatge.
    """
    if beguda_per_plat is None:
        beguda_per_plat = [None] * len(plats)

    # Precompute (0 quota)
    plats_input = []
    for i, plat in enumerate(plats):
        ingredients_en = list(plat.get("ingredients", []) or [])
        ingredients_ca = ingredients_ca_llista(kb, ingredients_en)

        # tècniques en format curt
        tecs = []
        for t in (transformacions_per_plat[i] or []):
            tec = (t.get("display") or t.get("nom") or "").strip()
            obj_en = (t.get("objectiu_ingredient") or "").strip()
            obj_ca = ingredient_ca(kb, obj_en) if obj_en else ""
            if tec:
                tecs.append({
                    "tecnica": tec,
                    "objectiu_ingredient_ca": obj_ca,
                })

        beg = beguda_per_plat[i] or {}
        beg_en = (beg.get("nom") or beg.get("name") or "").strip()

        plats_input.append({
            "id": i,
            "nom_original_en": plat.get("nom", "Unnamed dish"),
            "curs": (plat.get("curs", "") or "").strip().lower(),
            "ingredients_en": ingredients_en,
            "ingredients_ca_fixos": ingredients_ca,  # <- Llista tancada, ja traduïda
            "tecnniques": tecs,
            "beguda_recomanada_en": beg_en or None,
        })

    guia_servei = {
        "assegut": "Servei assegut: plat individual gran, emplatament de restaurant. No format mini.",
        "cocktail": "Servei còctel: peça petita 1–2 mossegades (cullera, vaset, mini platet o broqueta). No plat gran.",
        "finger_food": "Servei finger food: unitats agafables amb la mà. Evita ganivet i forquilla.",
    }.get((servei or "indiferent").strip().lower(), "Servei indiferent: mantén coherència amb el plat.")

    if model_gemini is None:
        # fallback local sense LLM
        out = []
        for p in plats_input:
            out.append({
                "id": p["id"],
                "nom_plat_ca": p["nom_original_en"],
                "ingredients_ca": p["ingredients_ca_fixos"],
                "descripcio_ca": "Descripció no disponible (LLM no configurat).",
                "presentacio_ca": "Presentació no disponible (LLM no configurat).",
                "beguda_recomanada_ca": "Sense recomanació de beguda",
                "notes_tecnniques_ca": "Sense dades LLM.",
                "image_sentence_en": "A single white round plate with the listed ingredients arranged in a clean modern composition.",
            })
        return out

    prompt = f"""
Ets un xef català i redactor/a de carta professional.

OBJECTIU
Genera la fitxa de CADA plat (en català) i, per a cada plat, UNA frase en anglès per a un model d'imatge (top-down plating).

IMPORTANT (RESTRICCIONS)
- Els ingredients en català ja són FIXOS i te'ls dono com a llista tancada per a cada plat.
- NO pots afegir ingredients nous ni sinònims d'ingredients fora de la llista.
- Si no hi ha tècniques, no n'inventis.
- Estil català natural, professional i sobri. No poesia.

SERVEI
- {guia_servei}

ESTILS (poden ser buits)
- Estil cultural: {estil_cultural or "cap"}
- Estil d'alta cuina: {estil_alta or "cap"}

ENTRADA (JSON)
{json.dumps(plats_input, ensure_ascii=False)}
NOM DEL PLAT (molt important)
- El nom ha de ser atractiu i de carta (2–7 paraules), sense poesia.
- Si hi ha tècniques: el nom HA d’incloure com a mínim 1 tècnica (en català) i 1 ingredient clau de ingredients_ca_fixos.
- Si hi ha estils (cultural/alta): fes-ne una referència subtil (p.ex. "cítric", "caribeny", "creatiu", "contemporani"), sense inventar ingredients.
- No tradueixis ni canviïs ingredients: només els pots mencionar si estan a ingredients_ca_fixos.

SORTIDA
Retorna NOMÉS un JSON vàlid amb aquest esquema:

{{
  "plats": [
    {{
      "id": 0,
      "nom_plat_ca": "... (2–7 paraules)",
      "ingredients_ca": ["... EXACTAMENT els mateixos que ingredients_ca_fixos, mateix ordre i longitud ..."],
      "descripcio_ca": "... (minim 200 caràcters, 2–3 frases, sense llistes ni markdown)",
      "presentacio_ca": "... (mínim 300 caràcters, molt visual i concreta)",
      "beguda_recomanada_ca": "... (si beguda_recomanada_en és null -> 'Sense recomanació de beguda')",
      "notes_tecnniques_ca": "... (si hi ha tècniques: frase breu. si no: 'Sense tècniques especials.')",
      "image_sentence_en": "... (EXACTAMENT 1 frase en anglès, max 30 paraules, només visible on-plate; no emocions)"
    }}
  ]
}}

CONDICIÓ TÈCNIQUES (molt important)
- Si un plat té tècniques, a descripcio_ca has d'integrar-les en un mateix paràgraf com:
  "TÈCNICA sobre INGREDIENT" (ingredient en català). NO posis la primera lletra de la tècnica en majúscula
""".strip()

    resp = model_gemini.generate_content(prompt)
    data = _json_from_text((resp.text or "").strip()) or {}
    plats_out = data.get("plats") if isinstance(data.get("plats"), list) else []

    # Post-validació suau (sense segona crida!)
    out = []
    by_id = {p["id"]: p for p in plats_out if isinstance(p, dict) and "id" in p}

    for p in plats_input:
        got = by_id.get(p["id"], {}) or {}
        ingredients_ca_fixos = p["ingredients_ca_fixos"]

        # força ingredients exactes (0 invents)
        got["ingredients_ca"] = ingredients_ca_fixos

        got.setdefault("nom_plat_ca", p["nom_original_en"])
        got.setdefault("descripcio_ca", "—")
        got.setdefault("presentacio_ca", "—")
        got.setdefault("notes_tecnniques_ca", "Sense tècniques especials." if not p["tecnniques"] else "Tècniques aplicades segons fitxa.")
        got.setdefault("beguda_recomanada_ca", "Sense recomanació de beguda")
        got.setdefault("image_sentence_en", "A single white round plate with the listed ingredients arranged in a clean modern composition.")

        # neteja
        for k in ["nom_plat_ca","descripcio_ca","presentacio_ca","beguda_recomanada_ca","notes_tecnniques_ca","image_sentence_en"]:
            got[k] = _neteja(got.get(k, ""))

        # llindars mínims -> fallback local (sense reintentar)
        if len(got["descripcio_ca"]) < 220:
            got["descripcio_ca"] = "Plat executat amb precisió, amb els ingredients indicats i un perfil de sabor coherent amb l'estil. Textura i contrast equilibrats, amb un acabat net i professional."
        if len(got["presentacio_ca"]) < 320:
            got["presentacio_ca"] = "Plat sobre un plat rodó blanc. Emplatament net i contemporani: els elements principals centrats, guarnicions distribuïdes amb simetria suau i alçades moderades. Salses o cremes aplicades en traç fi o punts controlats. Colors i textures visibles dels ingredients, sense decoració externa ni elements no comestibles."

        out.append({
            "id": p["id"],
            "nom_plat_ca": got["nom_plat_ca"],
            "ingredients_ca": got["ingredients_ca"],
            "descripcio_ca": got["descripcio_ca"],
            "presentacio_ca": got["presentacio_ca"],
            "beguda_recomanada_ca": got["beguda_recomanada_ca"],
            "notes_tecnniques_ca": got["notes_tecnniques_ca"],
            "image_sentence_en": got["image_sentence_en"],
        })

    return out



def _resum_plat_en_angles_per_imatge(plat: dict) -> str:
    """
    Fa una crida curta a Gemini per obtenir UNA frase en anglès
    que descrigui el plat només a nivell visual (formes, colors i ingredients).

    La frase és pensada per alimentar un model d'imatge (FLUX),
    així que ha de ser molt concreta i sense emocions.
    """
    nom = plat.get("nom", "") or ""
    curs = plat.get("curs", "") or ""
    ingredients = plat.get("ingredients", []) or []
    descripcio = plat.get("descripcio", "") or ""
    presentacio = plat.get("presentacio", "") or ""

    if isinstance(ingredients, (list, tuple)):
        ingredients_txt = ", ".join(str(x) for x in ingredients)
    else:
        ingredients_txt = str(ingredients)

    prompt = f"""
You help to build text prompts for an image generation model.

DISH INFORMATION
- Course: {curs}
- Dish name: {nom}
- Ingredients (Catalan words are fine): {ingredients_txt}
- Menu description in Catalan: {descripcio}
- Plating description in Catalan: {presentacio}

TASK
Write EXACTLY ONE sentence in simple English (max 30 words)
describing ONLY what is visible on the plate:

- mention shapes, colours, approximate number of elements and positions on the plate,
- mention ONLY ingredients from the list, do NOT add new ingredients,
- do NOT talk about emotions, taste, smell, guests, table, background, cutlery or decorations.

Return ONLY the English sentence, without quotes or any extra text.
"""

    try:
        resp = model_gemini.generate_content(prompt)
        text = (resp.text or "").strip()
        # Per seguretat, ens quedem amb la primera línia només
        return text.splitlines()[0].strip()
    except Exception as e:
        print(f"[IMATGE] Error resumint plat per a la imatge: {e}")
        # Fallback molt simple
        return f"A single white round plate with {ingredients_txt} arranged in a neat, modern composition."


# ---------------------------------------------------------------------
#  OPERADOR: GENERACIÓ D'IMATGE DEL MENÚ AMB HUGGING FACE (FLUX.1)
# ---------------------------------------------------------------------

def _resum_ambient_esdeveniment(
    tipus_esdeveniment: str,
    temporada: str,
    espai: str,
    formalitat: str,
) -> dict:
    """
    Resumeix l'ambient en termes d'esdeveniment, decoració i llum.
    Retorna un dict amb:
      - event_desc: frase breu de context
      - decor_desc: decoració específica de la taula
    (tot en anglès, perquè el model d'imatges s'hi entén millor)
    """
    tipus = (tipus_esdeveniment or "").lower()
    temporada = (temporada or "").lower()
    espai = (espai or "").lower()
    formalitat = (formalitat or "").lower()

    # --- Tipus d'esdeveniment ---
    if "casament" in tipus:
        event_desc = "an intimate wedding dinner for a small group"
        decor_desc = "one small glass vase with soft pastel flowers and a couple of tiny candles near the top edge of the table"
    elif "aniversari" in tipus:
        event_desc = "a cosy birthday dinner"
        decor_desc = "a tiny birthday candle decoration and a few small colorful confetti dots near the top corners of the table"
    elif "comunio" in tipus or "baptisme" in tipus or "bateig" in tipus:
        event_desc = "a refined family celebration dinner"
        decor_desc = "a small vase with white flowers and a single small candle near the top edge of the table"
    elif "empresa" in tipus or "congres" in tipus:
        event_desc = "a neat corporate dinner"
        decor_desc = "one slim water glass and a closed dark notebook placed near the top of the table"
    else:
        event_desc = "a stylish small banquet dinner"
        decor_desc = "one simple vase with green leaves or a single candle near the top edge of the table"

    # --- Temporada: ajustem la llum i el to ---
    if temporada == "primavera":
        season_txt = "soft daylight and a fresh, slightly pastel color mood"
    elif temporada == "estiu":
        season_txt = "warm daylight and a bright summery color mood"
    elif temporada == "tardor":
        season_txt = "warm amber light with a slightly autumnal color mood"
    elif temporada == "hivern":
        season_txt = "soft cool light with a subtle winter mood"
    else:
        season_txt = "neutral soft light"

    # --- Interior / exterior ---
    if espai == "exterior":
        place_txt = "on a single outdoor dining table"
    else:
        place_txt = "on a single indoor dining table next to a soft background"

    # --- Formalitat → tipus de superfície ---
    if formalitat == "informal":
        table_surface = "a light wooden table without a tablecloth"
    else:
        table_surface = "a white tablecloth covering the table"

    return {
        "event_desc": event_desc,
        "decor_desc": decor_desc,
        "season_txt": season_txt,
        "place_txt": place_txt,
        "table_surface": table_surface,
    }

def construir_prompt_imatge_menu(
    tipus_esdeveniment: str,
    temporada: str,
    espai: str,
    formalitat: str,
    plats_info: list[dict],
) -> str:
    """
    Prompt perquè FLUX generi EXACTAMENT tres plats en vista zenital:

      - 1 plat a la part superior (centre horitzontal),
      - 2 plats a la part inferior (esquerra i dreta),
      - cap altre plat ni bol.
    """

    ambient = _resum_ambient_esdeveniment(
        tipus_esdeveniment, temporada, espai, formalitat
    )

    # Garantim que cada entrada tingui la info que ens cal
    plats = []
    for p in plats_info[:3]:
        plats.append({
            "curs": p.get("curs", "") or "",
            "nom": p.get("nom", "") or "",
            "ingredients": p.get("ingredients", []) or [],
            "descripcio": p.get("descripcio", "") or "",
            "presentacio": p.get("presentacio", "") or "",
        })

    while len(plats) < 3:
        plats.append({
            "curs": "extra",
            "nom": "Neutral dish",
            "ingredients": ["neutral ingredient"],
            "descripcio": "simple balanced dish",
            "presentacio": "simple round portion in the centre of the plate",
        })

    top_plate, bottom_left, bottom_right = plats

    # Resum curt en anglès per a cada plat
    top_desc = _resum_plat_en_angles_per_imatge(top_plate)
    bl_desc = _resum_plat_en_angles_per_imatge(bottom_left)
    br_desc = _resum_plat_en_angles_per_imatge(bottom_right)

    prompt = f"""
Ultra realistic top-down food photography of a single dining table.
Scene: {ambient['place_txt']} set for {ambient['event_desc']}, with {ambient['season_txt']}.
The table surface is {ambient['table_surface']}.

CAMERA AND VIEW:
- Strict 90 degree overhead view (perfectly vertical), no perspective.
- Orthographic feeling, no visible vanishing lines.
- Everything in sharp focus, no blur.

ABSOLUTE RULES ABOUT PLATES:
- There must be EXACTLY THREE LARGE ROUND WHITE PLATES WITH FOOD on the table, no more and no less.
- These are the ONLY dishes on the table. There are NO other plates, NO small side plates,
  NO bowls, NO saucers and NO extra serving dishes, even if they are empty.
- ONE of the three plates is placed in the UPPER HALF of the image, centred horizontally.
- The space between the two lower plates must remain EMPTY tablecloth:
  do NOT place any plate, bowl, dish or food in the central lower area.
- The top left and top right corners of the image MUST NOT contain plates, bowls or any round dish shapes.
- Do NOT add a fourth plate anywhere.


LAYOUT OF THE THREE PLATES:

TOP PLATE (first course: {top_plate['nom']}):
- {top_desc}

BOTTOM LEFT PLATE (main course: {bottom_left['nom']}):
- {bl_desc}

BOTTOM RIGHT PLATE (dessert: {bottom_right['nom']}):
- {br_desc}
- This is the ONLY DESSERT plate: it must clearly look sweet (cake, cream or chocolate),
  while the other two plates must look savoury and not like desserts.

TABLE DECORATION AND EVENT DETAILS:
- Add EXACTLY ONE small table decoration related to the event: {ambient['decor_desc']}.
- Place this decoration near the TOP edge of the table, between the plates,
  clearly smaller than the plates and NOT circular.
- The decoration must NOT be confused with food or plates.

OTHER OBJECTS:
- You may add at most one small fork or knife near each plate,
  but they must be subtle and must not draw attention away from the dishes.
- No menus, no phones, no people or hands, no text.

STRICT NEGATIVES:
- Do NOT generate more than three plates.
- Do NOT show a 2x2 grid of plates.
- Do NOT add any extra plates, bowls, saucers or serving dishes, even if they look empty.
- Do NOT show wine glasses, water glasses, coffee cups or any other drink containers.
- Do NOT add bread baskets, butter plates or any side dishes not implied by the three plate descriptions.
- Do NOT place decorations between the camera and the plates.
- Do NOT use strong depth-of-field blur.

TECHNICAL:
- Neutral natural lighting, realistic colours.
- 16:9 aspect ratio, high resolution.
""".strip()

    return prompt


def genera_imatge_menu_hf(prompt_imatge: str, output_path: str = "menu_event.png") -> str | None:
    """
    Envia el prompt al model d'imatge de Hugging Face (FLUX.1-dev)
    i desa la imatge generada a 'output_path'.

    Requereix la variable d'entorn HF_TOKEN amb un token de lectura de Hugging Face.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[IMATGE] ERROR: falta la variable d'entorn HF_TOKEN.")
        return None

    model_id = "black-forest-labs/FLUX.1-dev"
    api_url = f"https://router.huggingface.co/hf-inference/models/{model_id}"

    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json",
    }

    payload = {
        "inputs": prompt_imatge,
        "parameters": {
            "num_inference_steps": 28,
            "guidance_scale": 3.5,
        },
        "options": {
            "wait_for_model": True,
        },
    }

    try:
        resp = requests.post(api_url, headers=headers, json=payload, timeout=300)
    except Exception as e:
        print(f"[IMATGE] Error de connexió amb Hugging Face: {e}")
        return None

    if resp.status_code != 200:
        print(f"[IMATGE] ERROR: resposta {resp.status_code}")
        try:
            print(resp.json())
        except Exception:
            print(resp.content[:200])
        return None

    try:
        with open(output_path, "wb") as f:
            f.write(resp.content)
        print(f"[IMATGE] Imatge del menú generada a: {output_path}")
        return output_path
    except Exception as e:
        print(f"[IMATGE] No s'ha pogut desar la imatge: {e}")
        return None


# ---------------------------------------------------------------------
#  ESQUELETS D'ALTRES OPERADORS (per si més endavant els implementes)
# ---------------------------------------------------------------------

def transferir_estil(plat, nou_estil):
    """ Transfereix un plat a un nou estil culinari. (pendent d'implementar) """
    pass


def comprova_equilibri(plat):
    """ Comprova l'equilibri del plat (pendent d'implementar). """
    pass


def comprova_pairing(plat):
    """ Assegura que els sabors principals del plat no xoquen. (pendent d'implementar) """
    pass


def ajustar_presentacio(plat, estil):
    """ Ajusta la presentació del plat segons l'estil. (pendent d'implementar) """
    pass