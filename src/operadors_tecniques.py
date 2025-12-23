import os
from typing import List, Dict, Set, Any, Optional
from collections import defaultdict
import re
import json
import re
import google.generativeai as genai

"""
OPERADORS DE TÈCNIQUES + FITXES LLM + IMATGE DE MENÚ

Aquest mòdul agrupa la capa “creativa” del sistema:
1) Selecció i assignació de tècniques culinàries a cada plat segons l’estil (cultural/alta/mixt),
   validant aplicabilitat per estat físic, macro-categoria, família i curs, i promovent diversitat
   (evitant tècniques massa semblants i protegint ingredients “escassos”).
2) Generació en 1 sola crida a Gemini de les fitxes de carta (nom, descripció, presentació, notes i
   frase EN per a imatge), mantenint els ingredients en català com a llista tancada per impedir
   invencions o traduccions no controlades.
3) Construcció d’un prompt fotorealista per renderitzar el menú complet (3 plats + begudes), amb
   composició estricta, negatives i cues visuals derivades de tècniques/presentació, i generació
   opcional via Hugging Face (o fallback imprimint el prompt).
"""


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
# 1. FUNCIONS AUXILIARS DE SCORE 
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

# ---------------------------------------------------------------------
# 4. APLICAR TECNIQUES MENU
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# 5. GENERACIÓ DESCRIPCIÓ LLM
# ---------------------------------------------------------------------
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
- A image_sentence_en NO diguis "top-down view", "photography", "table", "scene".
- A image_cues_en posa només efectes visuals de tècniques: thin slices, laminations, char marks, clean cuts, gel texture, glossy glaze, etc.


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
      "image_sentence_en": "... (EXACTAMENT 1 frase en anglès, max 30 paraules, només el menjar, sense dir 'top-down view')",
      "image_cues_en": "... (EXACTAMENT 1 frase en anglès, max 18 paraules: technique cues only, no plate/bowl words)"
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



# ---------------------------------------------------------------------
#  OPERADOR: GENERACIÓ D'IMATGE DEL MENÚ AMB HUGGING FACE 
# ---------------------------------------------------------------------

def _resum_ambient_esdeveniment(
    tipus_esdeveniment: str,
    temporada: str,
    espai: str,
    formalitat: str,
) -> dict:
    """
    Descriptors d'ambient per al prompt (EN).
    IMPORTANT: decoració pensada per NO semblar un plat/bol (evitem formes circulars).
    A més, afegim 'photo_style' per donar vibe real i coherent amb l'event.
    """
    tipus = (tipus_esdeveniment or "").lower()
    temporada = (temporada or "").lower()
    espai = (espai or "").lower()
    formalitat = (formalitat or "").lower()

    # Tipus d'esdeveniment (frase + decor + estil fotogràfic)
    if "casament" in tipus:
        event_desc = "an intimate wedding dinner"
        decor_desc = "a slim folded white linen napkin with a small name card near the top edge"
        photo_style = "natural candid wedding editorial styling, elegant but realistic, subtle imperfections"
    elif "aniversari" in tipus:
        event_desc = "a cosy birthday dinner"
        decor_desc = "a tiny ribbon and a small paper party hat near the top edge"
        photo_style = "warm friendly dinner photo, relaxed and natural, not staged"
    elif "comunio" in tipus or "baptisme" in tipus or "bateig" in tipus:
        event_desc = "a refined family celebration dinner"
        decor_desc = "a folded linen napkin with a simple ribbon near the top edge"
        photo_style = "refined family event styling, soft and clean, realistic table setting"
    elif "empresa" in tipus or "congres" in tipus:
        event_desc = "a neat corporate dinner"
        decor_desc = "a closed dark notebook with a pen near the top edge"
        photo_style = "corporate hospitality photo, minimal, tidy, professional and realistic"
    else:
        event_desc = "a stylish small banquet dinner"
        decor_desc = "a folded linen napkin with a minimal ribbon near the top edge"
        photo_style = "restaurant editorial photo, natural and realistic, minimal styling"

    # Temporada (llum i to)
    if temporada == "primavera":
        season_txt = "soft daylight, fresh slightly pastel mood"
    elif temporada == "estiu":
        season_txt = "warm daylight, bright summery mood"
    elif temporada == "tardor":
        season_txt = "warm amber daylight, autumn mood"
    elif temporada == "hivern":
        season_txt = "soft cool daylight, subtle winter mood"
    else:
        season_txt = "neutral soft daylight"

    # Interior / exterior
    if espai == "exterior":
        place_txt = "on a single outdoor dining table"
    else:
        place_txt = "on a single indoor dining table"

    # Formalitat → superfície i estil
    if formalitat == "informal":
        table_surface = "a light wooden table without a tablecloth"
        styling = "casual modern plating, slightly rustic, less symmetry"
    else:
        table_surface = "a clean white tablecloth"
        styling = "fine dining plating, minimal and precise, clean negative space"

    return {
        "event_desc": event_desc,
        "decor_desc": decor_desc,
        "season_txt": season_txt,
        "place_txt": place_txt,
        "table_surface": table_surface,
        "styling": styling,
        "photo_style": photo_style,
    }


def _clean_one_sentence(s: str, max_words: int = 42) -> str:
    s = (s or "").strip().replace("\n", " ").replace("\r", " ")
    s = re.sub(r"\s+", " ", s).strip().strip("-•* ")
    if not s:
        return ""
    words = s.split()
    if len(words) > max_words:
        s = " ".join(words[:max_words]).rstrip(",.;:") + "."
    if s[-1] not in ".!?":
        s += "."
    return s

def _techniques_visual_cues(techniques: List[Dict[str, str]]) -> str:
    """
    techniques = [{"tecnica": "...", "objectiu_ingredient_ca": "..."}]
    Retorna cues visuals EN curtes (sense ingredients nous).
    """
    if not techniques:
        return ""

    txt = " | ".join((t.get("tecnica","") or "").lower() for t in techniques)
    cues = []

    if "laminat" in txt:
        cues.append("paper-thin laminations clearly visible")
    if "microtall" in txt or "ultrafina" in txt:
        cues.append("micro-thin slices arranged with precision")
    if "yakimono" in txt:
        cues.append("gentle roast marks / light char, glossy finish")
    if "kiritsuke" in txt:
        cues.append("clean knife work, precise cuts and sharp edges")
    if "gelific" in txt:
        cues.append("smooth gel texture, structured glossy elements")
    if "tare" in txt:
        cues.append("concentrated glaze accents, glossy dots or streaks")

    if not cues:
        return ""

    return _clean_one_sentence("Technique cues: " + "; ".join(cues), max_words=22)



def _drink_visual_from_ca(beguda_ca: str, default: str) -> str:
    """
    Converteix beguda (CA) a una descripció visual EN.
    Manté’s simple perquè no inventi ampolles ni copes extra.
    """
    t = (beguda_ca or "").lower().strip()
    if not t or "sense recoman" in t:
        return default

    # vi
    if "vi blanc" in t or "blanc" in t or "white wine" in t:
        return "a simple white wine glass, pale straw wine, no bottle"
    if "vi negre" in t or "negre" in t or "red wine" in t:
        return "a simple red wine glass, deep ruby wine, no bottle"

    # cava / escumós
    if "cava" in t or "escum" in t or "sparkling" in t:
        return "a champagne flute with sparkling wine, no bottle"

    # refresc / soda / llimona
    if "refresc" in t or "soda" in t or "llimona" in t or "lemon" in t:
        return "a tall clear glass of sparkling lemon soda with ice, no can"

    # aigua
    if "aigua" in t or "water" in t:
        return "a tall clear water glass, no bottle"

    return default


def _presentation_to_visual_cues_ca(presentacio_ca: str) -> str:
    t = (presentacio_ca or "").lower()
    cues = []

    if "yakimono" in t or "signes de la cocció" in t or "lleugerament caramel" in t:
        cues.append("subtle roast marks and slight caramelization visible on the tomato slices")
    if "laminat" in t or "làmin" in t:
        cues.append("paper-thin miso laminations or shavings clearly visible")
    if "microtall" in t or "ultrafina" in t or "translúcid" in t:
        cues.append("a bed of micro-thin, translucent potato slices arranged like a light rosette")
    if "gotetes" in t or "gotes" in t or "punts" in t or "traç" in t:
        cues.append("fine oil droplets or controlled sauce dots, very minimal")
    if "copa" in t or "cristall" in t or "vidre" in t:
        cues.append("dessert served in a clear crystal dessert glass")
    if "quenelle" in t or "cullerada" in t or "forma ovalada" in t:
        cues.append("one neat oval scoop / quenelle shape")

    if not cues:
        return ""

    # dedup order-preserving
    cues = list(dict.fromkeys(cues))
    return _clean_one_sentence("Visual cues: " + "; ".join(cues), max_words=34)


def construir_prompt_imatge_menu(
    ambient: Dict[str, str],
    fitxes_menu: List[Dict[str, Any]],
    servei: str = "assegut",   # <-- AFEGIT
    incloure_decoracio: bool = True,
    incloure_begudes: bool = True,
    allow_dessert_glass: bool = True,
) -> str:
    """
    Versió definitiva:
    - EXACTAMENT 3 dish items (top, bottom-left, bottom-right)
    - bottom-center MUST be empty
    - optional 3 drinks, placed upper-right of each item
    - blocks extra glassware/empty bowls/spoons
    """
    plats = (fitxes_menu or [])[:3]
    while len(plats) < 3:
        plats.append({
            "nom_plat_ca": "Neutral dish",
            "image_sentence_en": "Clean minimal plating with the listed ingredients, centered and tidy.",
            "presentacio_ca": "",
            "beguda_recomanada_ca": "Sense recomanació de beguda",
        })

    def _safe_name(p: Dict[str, Any]) -> str:
        return (p.get("nom_plat_ca") or p.get("nom_original_en") or "Dish").strip()

    top, left, right = plats

    def _dish_desc(p: Dict[str, Any]) -> str:
        base = _clean_one_sentence(p.get("image_sentence_en", ""), max_words=28)
        cues = _presentation_to_visual_cues_ca(p.get("presentacio_ca", "") or "")
        merged = " ".join([x for x in [base, cues] if x])
        return _clean_one_sentence(merged, max_words=48)

    top_desc = _dish_desc(top)
    left_desc = _dish_desc(left)
    right_desc = _dish_desc(right)

    # Postre: permès en copa
    dessert_rule = (
        "Bottom-right is the ONLY dessert. It may be served in a clear dessert glass (optionally on a small white plate)."
        if allow_dessert_glass else
        "Bottom-right is the ONLY dessert and must be on a white plate (no glass)."
    )

    # Decoració: wedding card però sense text
    decor_line = ""
    if incloure_decoracio:
        decor_line = (
            f"Decoration: {ambient['decor_desc']}. "
            "If there is a name card, it must be blank with NO readable text."
        )

    # Begudes (1 per plat)
    drinks_block = ""
    if incloure_begudes:
        d_top = _drink_visual_from_ca(top.get("beguda_recomanada_ca", ""), "a simple white wine glass, no bottle")
        d_left = _drink_visual_from_ca(left.get("beguda_recomanada_ca", ""), "a simple red wine glass, no bottle")
        d_right = _drink_visual_from_ca(right.get("beguda_recomanada_ca", ""), "a tall clear water glass, no bottle")

        drinks_block = f"""
DRINKS (STRICT):
- EXACTLY THREE drinks total, one per dish item, no more and no less.
- Place each drink slightly ABOVE-RIGHT of its corresponding dish.
- Clear glassware only. NO mugs, NO cups with handles. NO bottles, NO cans, NO carafes.
- Absolutely NO extra empty glass bowls or extra glasses anywhere.
- Top drink: {d_top}.
- Bottom-left drink: {d_left}.
- Bottom-right drink: {d_right}.
""".strip()

    servei_norm = (servei or "assegut").strip().lower()

    if servei_norm == "assegut":
        ware_block = """
    - The two savoury dishes (top and bottom-left) are LARGE white plates (shallow is ok).
    - Minimal cutlery: at most one small fork or knife total, subtle.
    """.strip()
    elif servei_norm == "cocktail":
        ware_block = """
    - The three items are SMALL cocktail servings: a small plate, a small plate, and one dessert glass.
    - No large dinner plates. No full-size cutlery.
    - At most one small cocktail pick or one tiny fork total.
    """.strip()
    else:  # finger_food
        ware_block = """
    - The three items are FINGER FOOD servings: small portions on small white plates, plus one dessert glass.
    - No large dinner plates. No cutlery at all.
    """.strip()

    prompt = f"""
Ultra realistic top-down food photography, strict 90° overhead, no perspective.
{ambient.get('photo_style','')}

Scene: {ambient['place_txt']} styled for {ambient['event_desc']}.
Lighting: {ambient['season_txt']}.
Table: {ambient['table_surface']}.
Plating: {ambient['styling']}.

COMPOSITION (VERY STRICT):
- EXACTLY THREE dish items on the table (no more, no less).
- Layout: one dish at TOP-CENTER, one at BOTTOM-LEFT, one at BOTTOM-RIGHT.
- The BOTTOM-CENTER area must be EMPTY table surface (no plate, no bowl, no glassware).
{ware_block}
- {dessert_rule}
- No duplicate servings, no side plates, no extra bowls.


DISH DETAILS:
- Top dish ({_safe_name(top)}): {top_desc}
- Bottom-left dish ({_safe_name(left)}): {left_desc}
- Bottom-right dessert ({_safe_name(right)}): {right_desc}

{drinks_block}

{decor_line}

STRICT NEGATIVES:
- No extra glass bowls, no empty cups, no extra spoons.
- No text anywhere (no readable labels).
- No hands, no people, no menus.

Photorealistic, sharp focus, natural colors, high resolution, 16:9.
""".strip()

    return prompt



# ---------------------------------------------------------------------
#  MODE HÍBRID: HF si es pot, sinó imprimeix prompt + link + copy
# ---------------------------------------------------------------------

def _try_copy_to_clipboard(text: str) -> bool:
    """
    Opcional: pip install pyperclip
    """
    try:
        import pyperclip
        pyperclip.copy(text)
        return True
    except Exception:
        return False


def genera_imatge_menu_hf_o_prompt(
    prompt_imatge: str,
    output_path: str = "menu_event.png",
    model_id: str = "black-forest-labs/FLUX.1-dev",
    guidance_scale: float = 3.5,
    num_inference_steps: int = 28,
) -> Optional[str]:
    """
    Intenta generar la imatge via HF Inference.
    Si falla, imprimeix prompt i deixa “link” per enganxar manualment.
    """
    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        print("[IMATGE] No hi ha HF_TOKEN configurat.")
        print("[IMATGE] Prompt per enganxar manualment:\n")
        print(prompt_imatge)
        if _try_copy_to_clipboard(prompt_imatge):
            print("\n[IMATGE] Prompt COPIAT al porta-retalls ✅")
        print("\n[IMATGE] Prova aquests Spaces (manual):")
        print("  - https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell")
        print("  - https://huggingface.co/spaces/black-forest-labs/FLUX.1-dev")
        return None

    try:
        from huggingface_hub import InferenceClient
    except Exception:
        print("[IMATGE] Falta huggingface_hub. Instal·la: pip install huggingface_hub")
        print("[IMATGE] Prompt:\n")
        print(prompt_imatge)
        return None

    try:
        client = InferenceClient(model=model_id, token=hf_token)

        img = client.text_to_image(
            prompt_imatge,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
        )

        if hasattr(img, "save"):
            img.save(output_path)
        else:
            with open(output_path, "wb") as f:
                f.write(img)

        print(f"[IMATGE] Imatge del menú generada a: {output_path}")
        return output_path

    except Exception as e:
        msg = str(e).lower()
        quota_like = any(k in msg for k in ["quota", "rate limit", "429", "402", "payment", "billing", "forbidden", "403", "401", "unauthorized"])
        print(f"[IMATGE] No s'ha pogut generar automàticament ({e}).")
        if quota_like:
            print("[IMATGE] Sembla un problema de quota/permisos/token.")
        print("\n[IMATGE] Prompt per enganxar manualment:\n")
        print(prompt_imatge)
        if _try_copy_to_clipboard(prompt_imatge):
            print("\n[IMATGE] Prompt COPIAT al porta-retalls ✅")
        print("\n[IMATGE] Prova aquests Spaces (manual):")
        print("  - https://huggingface.co/spaces/black-forest-labs/FLUX.1-schnell")
        return None