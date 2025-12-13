import random
import os
from typing import List, Dict
import google.generativeai as genai
import base64
import requests

from operador_ingredients import adaptar_plat_a_estil, adaptar_plat_a_estil_latent
# Crida única de configuració (posa-la al principi del fitxer d'operadors)
API_KEY = os.environ.get("GEMINI_API_KEY")
if not API_KEY:
    raise RuntimeError("Falta la variable d'entorn GEMINI_API_KEY")

genai.configure(api_key=API_KEY)

# Model que farem servir (ràpid i força bo)
GEMINI_MODEL_NAME = "gemini-2.5-flash"
model_gemini = genai.GenerativeModel(GEMINI_MODEL_NAME)

# ---------------------------------------------------------------------
#  FUNCIONS AUXILIARS
# ---------------------------------------------------------------------

def _info_ingredients_plat(plat, base_ingredients):
    """
    Donat un plat amb plat['ingredients'] = ['tomàquet', 'nata', ...]
    retorna la llista de files d'ingredients.csv corresponents.
    """
    index = {row["nom_ingredient"]: row for row in base_ingredients}
    info = []
    for nom in plat.get("ingredients", []):
        fila = index.get(nom)
        if fila:
            info.append(fila)
    return info

# maridatge.py

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


def _troba_ingredient_aplicable(tecnica_row, plat, info_ings, ingredients_usats):
    """
    Retorna una tupla (frase_objectiu, nom_ingredient) on:
      - frase_objectiu: text tipus "el líquid 'salsa de soja'" o "el curs 'postres'"
      - nom_ingredient: nom cru ('salsa de soja') o None si s'ha referenciat el curs.
    Actualitza 'ingredients_usats' per evitar reutilitzar el mateix ingredient.
    """
    tags = set(tecnica_row.get("aplicable_a", "").split("|"))
    curs = plat.get("curs", "").lower()

    # 1. Busquem ingredients no usats, prioritzant categories fortes
    for ing_row in info_ings:
        nom_ing = ing_row["nom_ingredient"]
        if nom_ing in ingredients_usats:
            continue

        categoria = ing_row["categoria_macro"]
        familia = ing_row["familia"]

        # Líquids / salses
        if ("liquids" in tags or "salsa" in tags) and (
            categoria in ("salsa", "altre") or familia in ("aigua", "fons_cuina", "reducció_vi")
        ):
            ingredients_usats.add(nom_ing)
            return f"el líquid '{nom_ing}'", nom_ing

        # Fruita
        if "fruita" in tags and categoria == "fruita":
            ingredients_usats.add(nom_ing)
            return f"la fruita '{nom_ing}'", nom_ing

        # Làctic
        if "lacti" in tags and categoria == "lacti":
            ingredients_usats.add(nom_ing)
            return f"el làctic '{nom_ing}'", nom_ing

        # Feculent (cereals, pasta, arròs…)
        if "cereal_feculent" in tags and categoria == "cereal_feculent":
            ingredients_usats.add(nom_ing)
            return f"el feculent '{nom_ing}'", nom_ing

        # Proteïnes
        if (
            "proteina_animal" in tags or "peix" in tags
        ) and categoria in ("proteina_animal", "peix", "proteina_vegetal"):
            ingredients_usats.add(nom_ing)
            return f"la proteïna '{nom_ing}'", nom_ing

    # 2. Si no hem trobat cap match fort, agafem el primer ingredient no usat
    for ing_row in info_ings:
        nom_ing = ing_row["nom_ingredient"]
        if nom_ing not in ingredients_usats:
            ingredients_usats.add(nom_ing)
            return f"un ingredient com '{nom_ing}'", nom_ing

    # 3. Si no hi ha ingredients (o tots gastats),
    #    i la tècnica és específica de POSTRES, fem servir el CURS
    if "postres" in tags and curs == "postres":
        return "el curs 'postres'", None

    # 4. Fallback final genèric
    return "un element del plat", None


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
#  OPERADOR 1: SUBSTITUCIÓ D'INGREDIENT
# ---------------------------------------------------------------------

def substituir_ingredient(
    plat,
    tipus_cuina,
    base_ingredients,
    base_cuina,
    mode="regles",
    intensitat=0.4,
    ingredients_usats_latent=None,
    perfil_usuari=None,
):
    """
    Substitueix un ingredient d'un plat que NO sigui de l'estil de cuina desitjat
    per un altre ingredient amb el mateix rol i present a la base d'aquell estil.
    """
    if mode == "latent":
        # Mode creatiu basat en vectors de FlavorGraph
        try:
            return adaptar_plat_a_estil_latent(
                plat=plat,
                nom_estil=tipus_cuina,
                base_estils=base_cuina,
                base_ingredients=base_ingredients,
                intensitat=intensitat,
                ingredients_usats_latent=ingredients_usats_latent,
                perfil_usuari=perfil_usuari,
            )
        except Exception as exc:
            print(f"[INGREDIENTS] Error en adaptació latent '{tipus_cuina}': {exc}. S'usa mètode clàssic.")
            # Si hi ha qualsevol problema, fem servir la lògica simple.

    # Ingredients propis de l'estil
    ingredients_estil = set(base_cuina.get(tipus_cuina, {}).get('ingredients', []))

    # Ingredients del plat que NO són de l'estil
    ingredients_a_substituir = [ing for ing in plat['ingredients']
                                if ing not in ingredients_estil]

    if not ingredients_a_substituir:
        # Ja tot és coherent amb l'estil
        return plat

    # Escollim un ingredient a substituir
    ingredient_vell = random.choice(ingredients_a_substituir)

    # Troba el rol de l'ingredient a substituir
    rol = next(
        (ing['rol_tipic'] for ing in base_ingredients
         if ing['nom_ingredient'] == ingredient_vell),
        None
    )
    if not rol:
        return plat

    # Alternatives amb el mateix rol dins els ingredients de l'estil
    alternatives = [
        ing['nom_ingredient'] for ing in base_ingredients
        if ing['rol_tipic'] == rol and ing['nom_ingredient'] in ingredients_estil
    ]

    if not alternatives:
        return plat

    nou_ingredient = random.choice(alternatives)

    # Substituïm a la llista
    nou_plat = plat.copy()
    nou_plat['ingredients'] = [
        nou_ingredient if ing == ingredient_vell else ing
        for ing in plat['ingredients']
    ]

    print(f"[INGREDIENTS] Substituint '{ingredient_vell}' per '{nou_ingredient}' al plat '{plat['nom']}'")
    return nou_plat


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
    info_ings = _info_ingredients_plat(plat, base_ingredients)

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
    estil_row: dict | None = None,
) -> dict:
    """
    Genera nom nou, descripció de carta i proposta de presentació
    per a un plat, utilitzant Gemini.
    """

    nom_plat = plat.get("nom", "Plat sense nom")
    ingredients_llista = plat.get("ingredients", []) or []

    # Resum d'ingredients per no fer el prompt etern
    if not ingredients_llista:
        ingredients_txt = "ingredients diversos"
    elif len(ingredients_llista) <= 6:
        ingredients_txt = ", ".join(ingredients_llista)
    else:
        ingredients_txt = ", ".join(ingredients_llista[:6]) + ", ..."

    curs = plat.get("curs", "")

    # Info de l'estil (si ve de estils.csv)
    tipus = ""
    sabors_clau: list[str] = []
    caracteristiques: list[str] = []
    evita: list[str] = []

    if isinstance(estil_row, dict):
        tipus = estil_row.get("tipus", "")
        sabors_clau = (estil_row.get("sabors_clau") or "").split("|")
        caracteristiques = (estil_row.get("caracteristiques") or "").split("|")
        evita = (estil_row.get("evita") or "").split("|")

    txt_sabors = ", ".join([s for s in sabors_clau if s]) or "no especificats"
    txt_caract = ", ".join([c for c in caracteristiques if c]) or "no especificades"
    txt_evita = ", ".join([e for e in evita if e]) or "—"

    txt_transformacions = _format_transformacions_per_prompt(transformacions)

    prompt = f"""
Ets un xef català amb experiència en cuina d'autor i menús de banquet.
Has de proposar un nom i dues frases per descriure UN SOL PLAT.
Treballes sempre en català estàndard, clar i sense floritures innecessàries.

INFORMACIÓ DEL PLAT
- Nom original del plat: {nom_plat}
- Curs (primer/segon/postres): {curs or "no especificat"}
- Ingredients principals: {ingredients_txt}

INFORMACIÓ DE L'ESTIL
- Nom tècnic de l'estil: {estil_tecnic}
- Tipus d'estil: {tipus or "no especificat"}
- Sabors clau: {txt_sabors}
- Característiques generals de l'estil: {txt_caract}
- Coses que aquest estil acostuma a evitar: {txt_evita}

INFORMACIÓ SOBRE LES TÈCNIQUES APLICADES
{txt_transformacions}

TASCA
Escriu EXACTAMENT tres línies, amb aquest format exacte:

NOM: <nom nou del plat, relacionat amb "{nom_plat}" i l'estil {estil_tecnic}.
      Ha de sonar com un nom de carta de restaurant (per exemple "Tomàquet farcit i Esferificació", "Pinya en gelée fina").
      Pot tenir entre 2 i 6 paraules, sense paraules grandiloqüents com "molecular", "sensorial", "experiència", "viatge".>

DESCRIPCIO: <una sola frase (màxim 30 paraules) que soni com una descripció de carta de restaurant.
             Si hi ha tècniques aplicades, la frase HA D'ESMENTAR TOTES LES TÈCNIQUES de la llista,
             indicant clarament sobre quin ingredient s'aplica cadascuna.
             Fes servir literalment els textos "Tècnica sobre Ingredient" de la secció de tècniques aplicades
             (per exemple "Esferificació sobre salsa de soja", "Puntillisme de salsa sobre pinya natural"),
             integrant-los de manera natural dins la frase.
             Prohibit usar paraules com: explosió, sorpresa, viatge, emoció, experiència.>

PRESENTACIO: <una sola frase (màxim 35 paraules) explicant ÚNICAMENT com es disposa el plat al plat:
              descriu la posició dels elements (centre, línia, racó), l'alçada (pla, elevat), les formes (esferes, cubs, làmines, espirals),
              la presència de salses (en punts, línies, nappé) i els colors principals dels ingredients visibles.
              No parlis del que sentirà el comensal, no facis metàfores, no parlis d'emocions.>

RESTRICCIONS IMPORTANTS
- Català estàndard, sense castellanismes ni paraules inventades.
- No inventis ingredients nous, tècniques noves ni salses o guarnicions que no surtin dels ingredients o de la llista de tècniques aplicades.
- No inventis termes artístics addicionals (com "puntillisme", "fulla cruixent", "pinzellades", "geometries") que no apareguin a la llista de tècniques.
- No acabis cap frase amb punts suspensius "..." ni facis servir punts suspensius.
- Si el text de tècniques diu que el plat es manté clàssic (sense tècniques especials), NO en parlis:
  descriu el plat com una versió ben executada dins l'estil {estil_tecnic}.
- Estil funcional i minimalista: frases descriptives, concretes i objectives, sense poesia.
- Escriu només aquestes tres línies, sense text addicional ni explicacions.
"""

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

    # Safety net: si hi ha tècniques però la descripció no n'esmenta cap,
    # hi afegim una frase senzilla del tipus "S'aplica X sobre Y".
    if transformacions and descripcio:
        if not _descripcio_conte_tecnica(descripcio, transformacions):
            t0 = transformacions[0]
            nom_tecnica = (t0.get("display") or t0.get("nom") or "")
            obj = t0.get("objectiu_ingredient") or t0.get("objectiu_frase") or "un element del plat"
            descripcio = descripcio.rstrip(".")
            descripcio = f"{descripcio}. S'aplica {nom_tecnica} sobre {obj}."

    if not nom_nou:
        nom_nou = f"{nom_plat} (versió {estil_tecnic})"
    if not descripcio:
        descripcio = f"Versió adaptada del plat en clau {estil_tecnic.replace('_', ' ')}."
    if not presentacio:
        presentacio = "Presentació neta i ordenada, ressaltant el producte principal."

    def _neteja(txt: str) -> str:
        txt = txt.strip()
        txt = txt.replace("...", "")
        return txt

    descripcio = _neteja(descripcio)
    presentacio = _neteja(presentacio)

    return {
        "nom_nou": nom_nou,
        "descripcio_carta": descripcio,
        "proposta_presentacio": presentacio,
    }
# ---------------------------------------------------------------------
#  OPERADOR: GENERACIÓ D'IMATGE DEL MENÚ AMB HUGGING FACE (FLUX.1)
# ---------------------------------------------------------------------

def _resum_ambient_esdeveniment(
    tipus_esdeveniment: str,
    temporada: str,
    espai: str,
    formalitat: str,
) -> str:
    """
    Crea una descripció breu de l'ambient (en anglès) per al prompt d'imatge.
    """
    tipus = (tipus_esdeveniment or "").lower()
    temporada = (temporada or "").lower()
    espai = (espai or "").lower()
    formalitat = (formalitat or "").lower()

    # Tipus d'esdeveniment
    if "casament" in tipus:
        base = "an elegant wedding reception"
    elif "aniversari" in tipus:
        base = "a cheerful birthday celebration"
    elif "comunio" in tipus:
        base = "a refined family celebration"
    elif "empresa" in tipus or "congres" in tipus:
        base = "a corporate event dinner"
    else:
        base = "a stylish banquet event"

    # Interior / exterior
    if espai == "exterior":
        lloc = "in an outdoor garden terrace"
    else:
        lloc = "in an indoor dining room"

    # Temporada
    if temporada == "primavera":
        temporada_txt = "with soft daylight, pastel flowers and fresh greenery"
    elif temporada == "estiu":
        temporada_txt = "with warm light, vivid colors and a summery atmosphere"
    elif temporada == "tardor":
        temporada_txt = "with warm amber tones, candles and autumn foliage"
    elif temporada == "hivern":
        temporada_txt = "with soft cool lighting and subtle winter decorations"
    else:
        temporada_txt = "with neutral elegant lighting and simple decorations"

    # Formalitat
    if formalitat == "informal":
        formalitat_txt = "casual table setting, simple plates and a relaxed mood"
    else:
        formalitat_txt = "formal table setting, white tablecloths, polished cutlery and a refined mood"

    return f"{base} {lloc}, {temporada_txt}, {formalitat_txt}"


def construir_prompt_imatge_menu(
    tipus_esdeveniment: str,
    temporada: str,
    espai: str,
    formalitat: str,
    plats_info: list[dict],
) -> str:
    """
    Construeix un prompt perquè FLUX generi una imatge amb EXACTAMENT
    tres plats del menú, en vista zenital (planta), alineats horitzontalment
    i tots amb la mateixa importància visual.
    """

    ambient = _resum_ambient_esdeveniment(
        tipus_esdeveniment, temporada, espai, formalitat
    )

    # Ens assegurem de tenir com a mínim 3 plats
    plats_norm = []
    for plat in plats_info:
        plats_norm.append({
            "curs": plat.get("curs", ""),
            "nom": plat.get("nom", ""),
            "descripcio": plat.get("descripcio", ""),
            "presentacio": plat.get("presentacio", ""),
        })
    while len(plats_norm) < 3:
        plats_norm.append({
            "curs": "Extra",
            "nom": "Neutral dish",
            "descripcio": "",
            "presentacio": "simple round portion in the centre of the plate",
        })

    left, center, right = plats_norm[:3]

    # Text per a cada plat: ingredients/tècniques + presentació
    def _descr(plat):
        desc = plat["descripcio"].strip()
        pres = plat["presentacio"].strip()
        parts = []
        if desc:
            parts.append(f"main elements and techniques: {desc}")
        if pres:
            parts.append(f"plating style: {pres}")
        return " ".join(parts) if parts else "simple clean plating."

    left_desc = _descr(left)
    center_desc = _descr(center)
    right_desc = _descr(right)

    prompt = (
        f"Ultra realistic food photography for {ambient}. "

        # Vista zenital
        "Top-down flat lay view, perfectly overhead, no perspective lines, "
        "as if the camera is exactly above the table. "

        # Composició exacta
        "Exactly THREE plates are visible, arranged in a single straight horizontal row "
        "from left to right. All three plates are the same size and at the same distance "
        "from the camera, with equal visual importance. "
        "No other plates or dishes anywhere in the image. "

        # Assignació de plats
        f"LEFT plate: first course – {left['nom']}, {left_desc} "
        f"CENTER plate: main course – {center['nom']}, {center_desc} "
        f"RIGHT plate: dessert – {right['nom']}, {right_desc} "
        "Each dish must look clearly different from the others. "

        # Entorn de la taula
        "Background: a clean white tablecloth. Minimal fine-dining table elements "
        "(a fork and knife near each plate, maybe one or two simple glasses), "
        "but the three dishes are the focus. "

        # Coses que NO volem
        "No people, no guests, no menu cards, no printed text, no decorations blocking the view. "
        "No depth-of-field blur: everything in the frame is in sharp focus, "
        "all three plates equally sharp. "

        # Detalls tècnics
        "High-quality realistic lighting, neutral tones, 16:9 aspect ratio."
    )

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


def get_ingredient_principal(plat, base_ingredients):
    """Retorna l'ingredient del plat amb typical_role = main."""
    ingredient_principal = None
    llista_ingredients = []
    
    for ing in plat.get("ingredients_en", []):
        print(f"Revisant ingredient del plat: '{ing}'")
        for ing_row in base_ingredients:
            if ing_row['ingredient_name'] == ing:
                print(f"Coincidència trobada a base_ingredients: {ing_row['ingredient_name']}")
                llista_ingredients.append(ing_row)
                if ing_row['typical_role'] == "main":
                    print(f"Ingredient '{ing}' és MAIN!")
                    ingredient_principal = ing_row

    if ingredient_principal:
        print(f"Info ingredient principal del plat: {ingredient_principal}")
    else:
        print("No s'ha trobat ingredient principal")

    return ingredient_principal, llista_ingredients

def passa_filtre_dur(plat, beguda_row):
    curs = plat.get("curs", "")
    ordre = beguda_row["maridatge_ordre"]
    print(ordre)
    
    # Si la beguda és general passa directament
    if beguda_row.get("es_general", "").strip().lower() == "si":
            return True
        
    # ORDRE obligatori
    if curs == "primer":
        if ordre == "ordre-primer":
            return True
        else: False
    elif curs == "segon":
        if ordre == "ordre-segon":
            return True
        else: False
    elif curs == "postres":
        if ordre == "ordre-postres":
            return True
        else: return False
    

def score_beguda_per_plat(beguda_row, ingredient_principal, llista_ingredients):
    total_score = 0

    # ------------------------------
    # Funció interna per puntuar 1 ingredient
    # ------------------------------
    def score_per_ingredient(ingredient):
        if not ingredient:
            return 0
        
        score = 0
        
        # --- Famílies ---
        fam_beguda = set(beguda_row["va_be_amb_familia"].split("|"))
        if ingredient["family"] in fam_beguda:
            score += 2

        # --- Macro categories ---
        macro_beguda = set(beguda_row["va_be_amb_categoria_macro"].split("|"))
        if ingredient["macro_category"] in macro_beguda:
            score += 2

        # --- Sabors ---
        sabors_beguda = set(beguda_row["va_be_amb_sabors"].split("|"))
        evita_sabors = set(beguda_row["evita_sabors"].split("|"))
        sabors_ing = set(ingredient["base_flavors"].split("|"))

        # Suma per coincidència de sabors
        score += len(sabors_ing & sabors_beguda)

        # Resta per sabors conflictius
        score -= len(sabors_ing & evita_sabors)

        return score

    # ---------------------------------------------------------
    # 1) Puntuar ingredients normals
    # ---------------------------------------------------------
    for ing in llista_ingredients:
        total_score += score_per_ingredient(ing)

    # ---------------------------------------------------------
    # 2) Puntuar ingredient principal (DOBLE)
    # ---------------------------------------------------------
    if ingredient_principal:
        total_score += 2 * score_per_ingredient(ingredient_principal)

    return total_score

def recomana_beguda_per_plat(plat, base_begudes, base_ingredients):
    candidates = []

    # 1. Trobar l'ingredient principal del plat 
    ing_main, llista_ing = get_ingredient_principal(plat, base_ingredients)
    print(f"Ingredient principal: {ing_main}")
    
    # 2. FILTRE DUR (amb 'es_general') per cada beguda
    for row in base_begudes:
        if passa_filtre_dur(plat, row):
            print(f"{row['nom']} ha passat el filtre dur")
            candidates.append(row)
    
    if not candidates:
        return None, None

    # 3. Escollir la millor beguda per scoring
    millor = None
    millor_score = -999
    for row in candidates:
        print(f"Provant beguda {row['nom']}")
        sc = score_beguda_per_plat(row, ing_main, llista_ing)
        
        if sc > millor_score:
            millor = row
            millor_score = sc

    return millor, millor_score

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
