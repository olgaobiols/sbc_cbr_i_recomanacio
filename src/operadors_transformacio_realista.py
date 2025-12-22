import random
import os
from typing import List, Dict, Set, Any, Optional
import google.generativeai as genai
import requests

# Importem la lògica latent ja adaptada a KB
from operador_ingredients import adaptar_plat_a_estil_latent

# Configuració API (Idealment en un .env, però mantenim la teva estructura)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    # Per evitar que peti si no tens la clau posada mentre proves altres coses
    print("[AVÍS] Falta GEMINI_API_KEY. Les funcions LLM no funcionaran.")
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

def _get_info_ingredients_plat(plat: Dict, kb: Any) -> List[Dict]:
    """Recupera la info de tots els ingredients del plat usant la KB."""
    infos = []
    for nom in plat.get("ingredients", []):
        info = kb.get_info_ingredient(nom)
        if info:
            infos.append(info)
    return infos

def _troba_ingredient_aplicable(tecnica_row: Dict, plat: Dict, info_ings: List[Dict], ingredients_usats: Set[str]):
    """Busca sobre quin ingredient aplicar la tècnica."""
    tags = set(tecnica_row.get("aplicable_a", "").split("|"))
    curs = (plat.get("curs") or "").lower()

    # Prioritzem ingredients no usats que facin match amb els tags
    for info in info_ings:
        nom = info["ingredient_name"]
        if nom in ingredients_usats: continue
        
        cat = info.get("macro_category") or info.get("categoria_macro")
        fam = info.get("family")

        # Lògica de matching
        if ("liquids" in tags or "salsa" in tags) and (cat in ("salsa", "altre") or fam == "aigua"):
            ingredients_usats.add(nom)
            return f"el líquid '{nom}'", nom
        
        if "fruita" in tags and cat == "fruita":
            ingredients_usats.add(nom)
            return f"la fruita '{nom}'", nom
            
        if "proteina_animal" in tags and cat in ("proteina_animal", "protein_animal", "peix"):
            ingredients_usats.add(nom)
            return f"la proteïna '{nom}'", nom

    # Fallback: qualsevol no usat
    for info in info_ings:
        nom = info["ingredient_name"]
        if nom not in ingredients_usats:
            ingredients_usats.add(nom)
            return f"l'ingredient '{nom}'", nom

    # Fallback curs
    if "postres" in tags and "postres" in curs:
        return "el curs 'postres'", None

    return "un element del plat", None



def substituir_ingredient(
    plat,
    tipus_cuina,
    kb,
    mode="regles",
    intensitat=0.4,
    ingredients_estil_usats=None,
    perfil_usuari: Optional[Dict] = None,
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
        )
    return plat












def categoria_principal_plat(plat: dict, base_ingredients: List[Dict]) -> str:
    """
    Determina una categoria 'grossa' del plat a partir dels ingredients.
    Retorna valors tipus: 'peix', 'carn', 'vegetal', 'pasta_arros', 'postres', etc.
    És una heurística simple, suficient per a maridatge.
    """
    info_ings = _info_ingredients_plat(plat, base_ingredients)
    cats = {row["categoria_macro"] for row in info_ings}

    # Peix i marisc
    if "peix" in cats:
        return "peix"

    # Proteïna animal (carn) sense peix
    if "proteina_animal" in cats:
        return "carn"

    # Vegetarià/vegetal (verdura, cereal, lacti, etc.)
    if "verdura" in cats:
        return "vegetal"

    if "cereal_feculent" in cats:
        return "pasta_arros"

    if "lacti" in cats or "fruita" in cats:
        return "suau"

    # Si és postres, potser tens plat["curs"] == "postres"
    curs = plat.get("curs", "").lower()
    if "postres" in curs:
        return "postres"

    return "neutre"


def _score_tecnica_per_plat(tecnica_row, plat, info_ings):
    """
    Dona una puntuació (enter) que indica com bé encaixa aquesta tècnica
    amb el plat i els seus ingredients.

    Hi ha dues parts:
      1) SCORE BASE genèric (curs + categories + famílies)
      2) Ajustos específics segons categoria de la tècnica
         ('molecular', 'minimalista', 'nouvelle', 'classica',
          'nordica', 'fusio', 'creativa', 'mercat', ...).

    Com més alt el score, més coherent és aplicar la tècnica a aquest plat.
    """
    curs = plat.get("curs", "").lower()  # 'primer', 'segon', 'postres', ...
    tags = set(tecnica_row.get("aplicable_a", "").split("|")) if tecnica_row.get("aplicable_a") else set()
    categoria_tecnica = (tecnica_row.get("categoria") or "").lower()

    score = 0

    # -------------------------
    # 1) SCORE BASE GENERIC
    # -------------------------
    # Match per CURS (p.ex. 'postres', 'primer', 'segon' dins aplicable_a)
    if curs and curs in tags:
        score += 3

    # Match per CATEGORIA i FAMÍLIA d'ingredients (puntuació base)
    cats = {row["categoria_macro"] for row in info_ings}
    fams = {row["familia"] for row in info_ings}

    if tags & cats:
        score += 2
    if tags & fams:
        score += 1

    # Detecció de tipus d'ingredients presents al plat
    hi_ha_liquid = any(
        row["categoria_macro"] in ("salsa", "altre")
        or row["familia"] in ("aigua", "fons_cuina", "reducció_vi")
        for row in info_ings
    )
    hi_ha_fruta = any(row["categoria_macro"] == "fruita" for row in info_ings)
    hi_ha_lacti = any(row["categoria_macro"] == "lacti" for row in info_ings)
    hi_ha_feculent = any(row["categoria_macro"] == "cereal_feculent" for row in info_ings)
    hi_ha_proteina_o_verdura = any(
        row["categoria_macro"] in ("proteina_animal", "peix", "proteina_vegetal", "verdura")
        for row in info_ings
    )
    hi_ha_verdura = any(row["categoria_macro"] == "verdura" for row in info_ings)
    num_ingredients = len(info_ings)

    # -------------------------
    # 2) AJUSTOS PER CATEGORIA
    # -------------------------

    # 2.1) Criteris específics per a tècniques de CUINA MOLECULAR
    if categoria_tecnica == "molecular":
        # LÍQUIDS / SALSES → molt importants
        if ("liquids" in tags or "salsa" in tags or "altre" in tags) and hi_ha_liquid:
            score += 3

        # POSTRES moleculars funcionen molt bé
        if "postres" in tags and curs == "postres":
            score += 3

        # FRUITA, LACTIS, FECULENTS → bons candidats per gels, escumes, etc.
        if "fruita" in tags and hi_ha_fruta:
            score += 2
        if "lacti" in tags and hi_ha_lacti:
            score += 2
        if "cereal_feculent" in tags and hi_ha_feculent:
            score += 2

    # 2.2) Criteris específics per a tècniques MINIMALISTES / PLATING
    elif categoria_tecnica == "minimalista":
        # Plats amb pocs ingredients → més propensos a minimalisme
        if num_ingredients <= 4:
            score += 2

        # En postres, minimalisme acostuma a quedar molt bé
        if curs == "postres":
            score += 2

        # Presència de proteïna o verdura → encaixa amb plating en línia, contrast de volums…
        if hi_ha_proteina_o_verdura:
            score += 2

        # Minimalisme sol evitar salses pesades; plats “nets” tenen un plus
        hi_ha_salsa = any(row["categoria_macro"] == "salsa" for row in info_ings)
        if not hi_ha_salsa:
            score += 1

    # 2.3) NOUVELLE CUISINE
    elif categoria_tecnica == "nouvelle":
        # Plats lleugers, sovint primers/segons fins
        if curs in ("primer", "segon"):
            score += 2
        # Fons clars / líquids lleugers
        if hi_ha_liquid:
            score += 2
        # Verdures i proteïnes suaus → bons candidats
        if hi_ha_verdura or hi_ha_proteina_o_verdura:
            score += 2
        # Plats no excessivament carregats
        if num_ingredients <= 7:
            score += 1

    # 2.4) CUINA CLÀSSICA FRANCESA
    elif categoria_tecnica == "classica":
        # Segons amb proteïna o verdura → brasejats, glace, napar...
        if hi_ha_proteina_o_verdura and curs in ("segon", "principal"):
            score += 3
        # Presència de líquids → bons candidats per fons, roux, glace
        if hi_ha_liquid:
            score += 2
        # Feculents (gratinats, salses espessides, etc.)
        if hi_ha_feculent:
            score += 1

    # 2.5) NOVA CUINA NÒRDICA
    elif categoria_tecnica == "nordica":
        # Molt centrada en verdura, arrels, cereals, peix
        if hi_ha_verdura:
            score += 2
        if any(row["categoria_macro"] == "peix" for row in info_ings):
            score += 2
        # Plats relativament nets, sense massa ingredients
        if num_ingredients <= 8:
            score += 1

    # 2.6) CUINA FUSIÓ CONTEMPORÀNIA
    elif categoria_tecnica == "fusio":
        # Proteïna + líquid → ideal per lacats, infusions, gelee, etc.
        if hi_ha_proteina_o_verdura and hi_ha_liquid:
            score += 3
        # Fruita o feculent → ajuden a contrast dolç/salat/picant
        if hi_ha_fruta or hi_ha_feculent:
            score += 1

    # 2.7) CUINA CREATIVA
    elif categoria_tecnica == "creativa":
        # Forta orientació a líquids i textura
        if hi_ha_liquid:
            score += 3
        # Plats amb 3–8 ingredients → prou material per jugar, però sense caos
        if 3 <= num_ingredients <= 8:
            score += 2

    # 2.8) CUINA DE MERCAT
    elif categoria_tecnica == "mercat":
        # Pocs ingredients, producte protagonista
        if num_ingredients <= 6:
            score += 2
        # Verdura i proteïna fresques
        if hi_ha_verdura or hi_ha_proteina_o_verdura:
            score += 2
        # Primers/segons senzills
        if curs in ("primer", "segon", "principal"):
            score += 1

    # Altres categories es queden només amb el score base

    return score




# ---------------------------------------------------------------------
#  OPERADOR 2: APLICAR TÈCNIQUES A UN PLAT
# ---------------------------------------------------------------------
def triar_tecniques_per_plat(
    plat,
    nom_estil,
    base_estils,
    base_tecnniques,
    base_ingredients,
    max_tecniques=2,
    min_score=5,
    tecniques_ja_usades=None,
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
    info_ings = _get_info_ingredients_plat(plat, base_ingredients)

    estil_row = base_estils.get(nom_estil)
    if estil_row is None:
        if debug:
            print(f"[TEC] Estil '{nom_estil}' no trobat per al plat '{nom_plat}'.")
        return []

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

    # 3) Assignem ingredient/curs a cada tècnica
    transformacions = []
    ingredients_usats = set()

    for r in seleccionades_raw:
        nom_tecnica = r["nom"]
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        objectiu_frase, obj_ing = _troba_ingredient_aplicable(
            tec_row, plat, info_ings, ingredients_usats
        )

        impacte_textura = [
            t for t in (tec_row.get("impacte_textura") or "").split("|") if t
        ]
        impacte_sabor = [
            s for s in (tec_row.get("impacte_sabor") or "").split("|") if s
        ]

        transformacions.append(
            {
                "nom": nom_tecnica,
                "display": tec_row.get("display_nom", nom_tecnica),
                "objectiu_frase": objectiu_frase,
                "objectiu_ingredient": obj_ing,
                "descripcio": tec_row.get("descripcio", ""),
                "impacte_textura": impacte_textura,
                "impacte_sabor": impacte_sabor,
            }
        )

    return transformacions

#afegit!!!!!!!!!!!!!!!!!!
def triar_tecniques_per_menu(
    plats,              # llista de dicts [primer, segon, postres]
    nom_estil,
    base_estils,
    base_tecnniques,
    base_ingredients,
    max_tecniques_per_plat=2,
    min_score=5,
    debug=False,
):
    """
    Donada una llista de plats i un estil, tria tècniques per a cada plat
    intentant maximitzar la varietat a nivell de menú.
    Retorna una llista paral·lela de llistes de transformacions.
    """
    tecniques_ja_usades = set()
    resultats = []

    for plat in plats:
        transf = triar_tecniques_per_plat(
            plat=plat,
            nom_estil=nom_estil,
            base_estils=base_estils,
            base_tecnniques=base_tecnniques,
            base_ingredients=base_ingredients,
            max_tecniques=max_tecniques_per_plat,
            min_score=min_score,
            tecniques_ja_usades=tecniques_ja_usades,
            debug=debug,
        )
        tecniques_ja_usades.update(t["nom"] for t in transf)
        resultats.append(transf)

    return resultats



def _extreu_camp_resposta(text: str, etiqueta: str) -> str | None:
    """
    Donat un text del LLM, extreu la línia que comença per, per exemple:
       'NOM:' o 'DESCRIPCIO:' ...
    i en retorna el contingut després dels dos punts.
    """
    etiqueta_upper = etiqueta.upper() + ":"
    for linia in text.splitlines():
        linia = linia.strip()
        if linia.upper().startswith(etiqueta_upper):
            return linia.split(":", 1)[1].strip()
    return None



#CANVIADA!!!!!!!!
def _format_transformacions_per_prompt(transformacions: list[dict]) -> str:
    """
    Genera un text breu i estructurat amb les tècniques aplicades,
    de manera que el LLM pugui reutilitzar literalment noms de tècniques i ingredients.
    """
    if not transformacions:
        # Indiquem clarament que NO hi ha tècniques perquè el model no se les inventi
        return (
            "En aquest plat no s'ha aplicat cap tècnica especial; el plat es manté clàssic. "
            "Descriu-lo com un plat ben executat dins l'estil, però sense mencionar tècniques."
        )

    linies = [
        "TÈCNIQUES APLICADES AL PLAT (usa aquests textos exactes quan parlis de tècniques):"
    ]
    for i, t in enumerate(transformacions, start=1):
        nom = t.get("display") or t.get("nom") or ""
        ing_nom = t.get("objectiu_ingredient") or "un ingredient del plat"
        linies.append(f"{i}) {nom} sobre {ing_nom}")

    return "\n".join(linies)


def _descripcio_conte_tecnica(descripcio: str, transformacions: list[dict]) -> bool:
    """
    Comprova si la descripció generada conté almenys un nom de tècnica.
    Serveix per saber si el model ha fet cas a les transformacions.
    """
    if not descripcio or not transformacions:
        return False

    text_lower = descripcio.lower()
    for t in transformacions:
        nom_disp = (t.get("display") or t.get("nom") or "").lower()
        if nom_disp and nom_disp in text_lower:
            return True
    return False


def genera_descripcio_llm(
    plat: dict,
    transformacions: list[dict],
    estil_tecnic: str,
    servei: str,
    estil_row: dict | None = None,
) -> dict:
    """
    Genera NOM, DESCRIPCIO i PRESENTACIO pensats per:
      - sonar bé en carta
      - però sobretot ser útils per generar una imatge acurada
    """

    nom_plat = plat.get("nom", "Plat sense nom")
    ingredients_llista = plat.get("ingredients", []) or []
    curs = (plat.get("curs", "") or "").strip().lower()
    servei_norm = (servei or "indiferent").strip().lower()

    # Ingredients (sense fer-ho etern)
    if not ingredients_llista:
        ingredients_txt = "ingredients diversos"
        ingredients_full = ""
    else:
        ingredients_txt = ", ".join(ingredients_llista[:8])
        ingredients_full = ", ".join(ingredients_llista)

    # Info estil (estils.csv)
    tipus = ""
    sabors_clau = []
    caracteristiques = []
    evita = []
    if isinstance(estil_row, dict):
        tipus = estil_row.get("tipus", "") or ""
        sabors_clau = [x for x in (estil_row.get("sabors_clau") or "").split("|") if x]
        caracteristiques = [x for x in (estil_row.get("caracteristiques") or "").split("|") if x]
        evita = [x for x in (estil_row.get("evita") or "").split("|") if x]

    txt_sabors = ", ".join(sabors_clau) if sabors_clau else "—"
    txt_caract = ", ".join(caracteristiques) if caracteristiques else "—"
    txt_evita = ", ".join(evita) if evita else "—"

    # -------------------------
    # Helpers: format servei
    # -------------------------
    def _guia_servei(s: str) -> tuple[str, str]:
        if s == "assegut":
            return (
                "FORMAT SERVEI: ASSEGUT (plat individual gran, emplatament de restaurant).",
                "Prohibit descriure-ho com mini, mos, cullera, broqueta, vaset, safata o peces repetides."
            )
        if s == "cocktail":
            return (
                "FORMAT SERVEI: COCKTAIL (peça petita 1–2 mossegades, servida en cullera/vaset/mini platet o broqueta).",
                "Prohibit descriure-ho com un plat gran complet o una ració principal."
            )
        if s == "finger_food":
            return (
                "FORMAT SERVEI: FINGER FOOD (unitats que es poden agafar amb la mà: tartaleta, croqueta, mini entrepà, wrap, broqueta...).",
                "Evita formats que requereixin ganivet i forquilla; parla d'unitats petites repetibles."
            )
        return (
            "FORMAT SERVEI: INDIFERENT (sigues coherent amb el plat, sense forçar format).",
            "—"
        )

    guia_servei, prohibit_servei = _guia_servei(servei_norm)

    # -------------------------
    # Helpers: tècniques -> spec
    # -------------------------
    def _resum_transformacions_specs(transformacions: list[dict]) -> str:
        """
        Construeix un bloc curt però molt informatiu:
        - Display + objectiu ingredient
        - Descripcio tècnica
        - Impacte textura i sabor (per convertir-ho en visuals)
        """
        if not transformacions:
            return (
                "TÈCNIQUES:\n"
                "- Cap tècnica especial aplicada (plat clàssic)."
            )

        lines = ["TÈCNIQUES (usa-les i fes-les visibles):"]
        for i, t in enumerate(transformacions, 1):
            disp = (t.get("display") or t.get("nom") or "").strip()
            obj = (t.get("objectiu_ingredient") or "").strip()
            if not obj:
                # si no hi ha ingredient concret, fem servir objectiu_frase curt
                obj = (t.get("objectiu_frase") or "un element del plat").strip()

            desc_tec = (t.get("descripcio") or "").strip()

            tx = t.get("impacte_textura", [])
            sb = t.get("impacte_sabor", [])

            if isinstance(tx, str):
                tx = [x for x in tx.split("|") if x]
            if isinstance(sb, str):
                sb = [x for x in sb.split("|") if x]

            tx_txt = ", ".join(tx) if tx else "—"
            sb_txt = ", ".join(sb) if sb else "—"

            lines.append(
                f"{i}) {disp} sobre {obj} | textura: {tx_txt} | sabor: {sb_txt} | definició: {desc_tec}"
            )
        return "\n".join(lines)

    txt_transformacions_specs = _resum_transformacions_specs(transformacions)

    # També mantenim el teu format curt “Tècnica sobre ingredient” perquè Gemini ho copiï literalment
    txt_transformacions_literal = _format_transformacions_per_prompt(transformacions)

    # -------------------------
    # Prompt nou: més curt però més precís
    # -------------------------
    prompt = f"""
Ets un xef català i també un estilista gastronòmic per fotografia.
Objectiu: descriure el plat perquè algú el pugui EMPLATAR i FOTOGRAFIAR amb precisió.

DADES DEL PLAT
- Nom original: {nom_plat}
- Curs: {curs or "—"}
- Ingredients (llista): {ingredients_txt}
- Ingredients (complet, per no inventar): {ingredients_full or ingredients_txt}

ESTIL
- Estil tècnic: {estil_tecnic}
- Tipus: {tipus or "—"}
- Sabors clau: {txt_sabors}
- Característiques: {txt_caract}
- Evita: {txt_evita}

{guia_servei}

TÈCNIQUES APLICADES (importantíssim per la imatge)
A) LITERAL (copia exactament aquests noms quan les mencionis):
{txt_transformacions_literal}

B) ESPECIFICACIÓ (fes que siguin visibles i coherents amb la definició i impactes):
{txt_transformacions_specs}

TASCA (retorna EXACTAMENT 3 línies, sense text extra):

NOM: (2 a 6 paraules) nom de carta coherent amb el servei "{servei_norm}" i l'estil, sense paraules grandiloqüents.

DESCRIPCIO: (1 frase, màx 38 paraules) descripció de carta PERÒ útil per imatge:
- menciona ingredient principal + 1-2 elements secundaris visibles,
- si hi ha tècniques, HAS D'ESMENTAR TOTES les tècniques del bloc A, indicant sobre quin ingredient s'aplica cadascuna,
- inclou 1 pista visual de textura (p.ex. gel/cubs/escuma/laminat/cruixent/pols) sense inventar ingredients.

PRESENTACIO: (1 frase, màx 55 paraules) instruccions VISUALS per dibuixar el plat:
- posició (centre/anell/línia/racó), alçada (pla/elevat), formes (cubs, làmines, esferes, quenelle, gotes, pols),
- nombre aproximat d'elements repetits (p.ex. 6 gotes, 10 esferes, 3 làmines),
- colors principals i a quin ingredient corresponen,
- coherent amb el servei "{servei_norm}" (assegut = plat gran; cocktail = peça mini; finger = unitats).

RESTRICCIONS DURes
- No inventis ingredients que no siguin a la llista.
- No inventis tècniques ni noms artístics addicionals.
- No parlis d'emocions, viatges, sorpresa, experiència, explosió.
- No posis punts suspensius.
- {prohibit_servei}
""".strip()

    try:
        resp = model_gemini.generate_content(prompt)
        text = resp.text or ""
    except Exception as e:
        print(f"[LLM] Error cridant Gemini: {e}")
        return {
            "nom_nou": f"{nom_plat} (versió {estil_tecnic})",
            "descripcio_carta": f"Versió adaptada del plat en clau {estil_tecnic.replace('_', ' ')}.",
            "proposta_presentacio": "Presentació neta i ordenada, ressaltant el producte principal.",
        }

    nom_nou = _extreu_camp_resposta(text, "NOM")
    descripcio = _extreu_camp_resposta(text, "DESCRIPCIO")
    presentacio = _extreu_camp_resposta(text, "PRESENTACIO")

    # Safety net: si hi ha tècniques però la descripció no en cita cap, forcem la primera
    if transformacions and descripcio:
        if not _descripcio_conte_tecnica(descripcio, transformacions):
            t0 = transformacions[0]
            nom_tecnica = (t0.get("display") or t0.get("nom") or "")
            obj = t0.get("objectiu_ingredient") or t0.get("objectiu_frase") or "un element del plat"
            descripcio = descripcio.rstrip(".")
            descripcio = f"{descripcio}. {nom_tecnica} sobre {obj}."

    if not nom_nou:
        nom_nou = f"{nom_plat} (versió {estil_tecnic})"
    if not descripcio:
        descripcio = f"Versió adaptada del plat en clau {estil_tecnic.replace('_', ' ')}."
    if not presentacio:
        presentacio = "Presentació neta i ordenada, ressaltant el producte principal."

    def _neteja(txt: str) -> str:
        txt = (txt or "").strip()
        txt = txt.replace("...", "")
        return txt

    return {
        "nom_nou": _neteja(nom_nou),
        "descripcio_carta": _neteja(descripcio),
        "proposta_presentacio": _neteja(presentacio),
    }

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
