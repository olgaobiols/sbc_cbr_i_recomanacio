import random

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


def _troba_ingredient_aplicable(tecnica_row, plat, info_ings, ingredients_usats):
    """
    Retorna una tupla (frase_objectiu, nom_ingredient) on:
      - frase_objectiu: text tipus "el líquid 'salsa de soja'" o "el curs 'postres'"
      - nom_ingredient: nom cru ('salsa de soja') o None si s'ha referenciat el curs.
    Actualitza 'ingredients_usats' per evitar reutilitzar el mateix ingredient.
    """
    tags = set(tecnica_row.get("aplicable_a", "").split("|"))
    curs = plat.get("curs", "").lower()

    # 1. Justificació pel CURS (ex: postres)
    if "postres" in tags and curs == "postres":
        # No gastem cap ingredient concret
        return "el curs 'postres'", None

    # 2. Busquem ingredients no usats, prioritzant categories fortes
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

    # 3. Si no hem trobat cap match fort, agafem el primer ingredient no usat
    for ing_row in info_ings:
        nom_ing = ing_row["nom_ingredient"]
        if nom_ing not in ingredients_usats:
            ingredients_usats.add(nom_ing)
            return f"un ingredient com '{nom_ing}'", nom_ing

    # 4. No hi ha ingredients (o tots gastats)
    return "un element del plat", None


def _score_tecnica_per_plat(tecnica_row, plat, info_ings):
    """
    Dona una puntuació (enter) que indica com bé encaixa aquesta tècnica
    amb el plat i els seus ingredients. (Sense mirar l'estil, només aplicabilitat)
    """
    curs = plat.get("curs", "").lower()  # 'primer', 'segon', 'postres', ...
    tags = set(tecnica_row.get("aplicable_a", "").split("|"))

    score = 0

    # --- 1) Match per CURS (p.ex. 'postres', 'primer', 'segon' dins aplicable_a)
    if curs and curs in tags:
        score += 3

    # --- 2) Match per CATEGORIA i FAMÍLIA d'ingredients (puntuació base)
    cats = {row["categoria_macro"] for row in info_ings}
    fams = {row["familia"] for row in info_ings}

    if tags & cats:
        score += 2
    if tags & fams:
        score += 1

    # --- 3) Heurístiques específiques segons tipus d'ingredient ---

    # Detecció de tipus d'ingredients al plat
    hi_ha_liquid = any(
        row["categoria_macro"] in ("salsa", "altre")
        or row["familia"] in ("aigua", "fons_cuina", "reducció_vi")
        for row in info_ings
    )
    hi_ha_fruta = any(row["categoria_macro"] == "fruita" for row in info_ings)
    hi_ha_lacti = any(row["categoria_macro"] == "lacti" for row in info_ings)
    hi_ha_feculent = any(row["categoria_macro"] == "cereal_feculent" for row in info_ings)

    # LÍQUIDS / SALSES / ALTRES → molt important per tècniques moleculars
    if ("liquids" in tags or "salsa" in tags or "altre" in tags) and hi_ha_liquid:
        score += 3

    # POSTRES
    if "postres" in tags and curs == "postres":
        score += 3

    # FRUITA
    if "fruita" in tags and hi_ha_fruta:
        score += 2

    # LACTIS
    if "lacti" in tags and hi_ha_lacti:
        score += 2

    # FECULENTS
    if "cereal_feculent" in tags and hi_ha_feculent:
        score += 2

    return score


# ---------------------------------------------------------------------
#  OPERADOR 1: SUBSTITUCIÓ D'INGREDIENT
# ---------------------------------------------------------------------

def substituir_ingredient(plat, tipus_cuina, base_ingredients, base_cuina):
    """
    Substitueix un ingredient d'un plat que NO sigui de l'estil de cuina desitjat
    per un altre ingredient amb el mateix rol i present a la base d'aquell estil.
    """
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
    debug=False,
):
    """
    Selecciona fins a `max_tecniques` tècniques de l'estil donat que encaixen
    amb el plat. Per a cada tècnica, associa un ingredient (o el curs) on aplicar-la.

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

        score = _score_tecnica_per_plat(tec_row, plat, info_ings)
        if debug:
            print(f"[SCORE] Plat '{nom_plat}', tècnica '{nom_tecnica}' → {score}")

        if score >= min_score:
            scored.append({"nom": nom_tecnica, "score": score})

    if not scored:
        if debug:
            print(f"[TEC] Cap tècnica de '{nom_estil}' supera el mínim per a '{nom_plat}'.")
        return []

    # 2) Ordenem de millor a pitjor i triem fins a max_tecniques
    scored.sort(key=lambda x: x["score"], reverse=True)
    if max_tecniques is not None:
        scored = scored[:max_tecniques]

    # 3) Assignem ingredient/curs a cada tècnica
    transformacions = []
    ingredients_usats = set()

    for r in scored:
        nom_tecnica = r["nom"]
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        objectiu_frase, obj_ing = _troba_ingredient_aplicable(
            tec_row, plat, info_ings, ingredients_usats
        )

        impacte_textura = [
            t for t in tec_row.get("impacte_textura", "").split("|") if t
        ]
        impacte_sabor = [
            s for s in tec_row.get("impacte_sabor", "").split("|") if s
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

def descriu_transformacions(plat, transformacions):
    """
    Genera un text explicatiu de les transformacions aplicades a un plat.

    Args:
        plat (dict): plat original (amb camp 'nom').
        transformacions (list[dict]): resultat de triar_tecniques_per_plat.

    Retorna:
        str: bloc de text en Markdown.
    """
    nom_plat = plat.get("nom", "<sense_nom>")

    if not transformacions:
        return (
            f"Al plat **{nom_plat}** no s'ha aplicat cap tècnica molecular concreta; "
            "es manté en una presentació més clàssica."
        )

    línies = [f"Al plat **{nom_plat}** es proposen les següents transformacions:"]

    for t in transformacions:
        # Objectiu: ingredient o curs
        obj = t["objectiu_frase"] or "un element del plat"

        # Impactes
        txt_textura = ", ".join(t["impacte_textura"]) if t["impacte_textura"] else "—"
        txt_sabor = ", ".join(t["impacte_sabor"]) if t["impacte_sabor"] else "—"

        línia = (
            f"- **{t['display']}** aplicada sobre {obj}: "
            f"{t['descripcio']}. "
        )
        if txt_textura != "—":
            línia += f"Aporta una textura *{txt_textura}*"
        if txt_sabor != "—":
            if txt_textura != "—":
                línia += " i "
            else:
                línia += "Amb això "
            línia += f"un efecte de sabor *{txt_sabor}*."

        línies.append(línia)

    return "\n".join(línies)



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
