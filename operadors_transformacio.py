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
    Intenta trobar l'ingredient o el curs que justifica aplicar la tècnica.

    IMPORTANT: intenta NO reutilitzar ingredients que ja són a 'ingredients_usats'.
               Si tria un ingredient, l'afegeix al conjunt perquè altres tècniques
               no el tornin a fer servir.
    """
    tags = set(tecnica_row.get("aplicable_a", "").split("|"))
    curs = plat.get("curs", "").lower()

    # 1) Justificació per CURS (ex. postres)
    if "postres" in tags and curs == "postres":
        return "el curs 'postres'"

    # 2) Busquem un ingredient NO utilitzat encara, prioritzant:
    #    - líquids / salses
    #    - fruita
    #    - lactis
    #    - feculents
    #    - proteïnes
    for ing_row in info_ings:
        nom_ing = ing_row["nom_ingredient"]
        if nom_ing in ingredients_usats:
            continue  # ja s'ha fet servir en una altra tècnica

        categoria = ing_row["categoria_macro"]
        familia = ing_row["familia"]

        # 2.1 Líquids / salses / altres
        if ("liquids" in tags or "salsa" in tags or "altre" in tags) and (
            categoria in ("salsa", "altre")
            or familia in ("aigua", "fons_cuina", "reducció_vi")
        ):
            ingredients_usats.add(nom_ing)
            return f"el líquid '{nom_ing}'"

        # 2.2 Fruita
        if "fruita" in tags and categoria == "fruita":
            ingredients_usats.add(nom_ing)
            return f"la fruita '{nom_ing}'"

        # 2.3 Lacti
        if "lacti" in tags and categoria == "lacti":
            ingredients_usats.add(nom_ing)
            return f"el làctic '{nom_ing}'"

        # 2.4 Feculent
        if "cereal_feculent" in tags and categoria == "cereal_feculent":
            ingredients_usats.add(nom_ing)
            return f"el feculent '{nom_ing}'"

        # 2.5 Proteïna (peix / carn / proteïna vegetal)
        if ("peix" in tags or "proteina_animal" in tags) and categoria in (
            "peix",
            "proteina_animal",
            "proteina_vegetal",
        ):
            ingredients_usats.add(nom_ing)
            return f"la proteïna '{nom_ing}'"

    # 3) Si no hem trobat res amb les regles fortes, agafem qualsevol ingredient
    for ing_row in info_ings:
        nom_ing = ing_row["nom_ingredient"]
        if nom_ing not in ingredients_usats:
            ingredients_usats.add(nom_ing)
            return f"un ingredient com '{nom_ing}'"

    # 4) Si ja s'han gastat tots els ingredients (o no n'hi ha)
    return "un element del plat"


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

def aplica_tecniques_al_plat(plat, tecniques_a_aplicar):
    """
    Aplica una llista de tècniques al plat, movent l'antiga tècnica principal
    a secundàries.

    Args:
        plat (dict): Plat a modificar.
        tecniques_a_aplicar (list[str]): Llista de noms de tècniques (ordenades de millor a pitjor).

    Retorna:
        (nou_plat, log_canvis):
            - nou_plat (dict): El plat amb les tècniques actualitzades.
            - log_canvis (str): Resum textual dels canvis.
    """
    if not tecniques_a_aplicar:
        return plat, "Cap tècnica nova aplicada (Score < Mínim)."

    nou_plat = plat.copy()
    log = []

    tecniques_actuals = set(
        t for t in nou_plat.get("tecnniques_secundaries", []) if t
    )
    antiga_principal = nou_plat.get("tecnica_principal")

    # 1) Moure l'antiga principal a secundàries
    if antiga_principal and antiga_principal not in tecniques_actuals:
        tecniques_actuals.add(antiga_principal)
        log.append(f"Antiga tècnica principal '{antiga_principal}' moguda a secundàries.")

    # 2) Nova tècnica principal = primera de la llista
    nova_principal = tecniques_a_aplicar[0]
    nou_plat["tecnica_principal"] = nova_principal
    log.append(f"Tècnica principal assignada: '{nova_principal}' (millor score).")

    # 3) La resta → secundàries
    for tec in tecniques_a_aplicar[1:]:
        if tec not in tecniques_actuals:
            tecniques_actuals.add(tec)
            log.append(f"Afegida a secundàries: '{tec}'.")

    nou_plat["tecnniques_secundaries"] = list(tecniques_actuals)

    return nou_plat, "\n".join(log)


def triar_tecniques_aplicables(plat, nom_estil, base_estils, base_tecnniques, base_ingredients):
    """
    Troba la llista de tècniques clau de l'estil que són aplicables 
    al plat i retorna les que superin un score mínim, ordenades per score.

    Retorna:
        list[(nom_tecnica, raó_aplicació)]
    """
    MIN_SCORE = 5  # llindar per considerar la tècnica "aplicable"
    nom_plat = plat.get("nom", "<sense_nom>")

    estil_row = base_estils.get(nom_estil)
    if estil_row is None:
        print(f"[TRIA TECNICA] No s'ha trobat l'estil '{nom_estil}' a base_estils.")
        return []

    tecnniques_str = estil_row.get("tecnniques_clau", "")
    if not tecnniques_str:
        print(f"[TRIA TECNICA] L'estil '{nom_estil}' no té tecnniques_clau definides.")
        return []

    tecniques_candidats = tecnniques_str.split("|")

    # Info dels ingredients d'aquest plat
    info_ings = _info_ingredients_plat(plat, base_ingredients)

    # 1) Calcul de scores
    resultats = []
    for nom_tecnica in tecniques_candidats:
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        score = _score_tecnica_per_plat(tec_row, plat, info_ings)
        print(f"[SCORE] Plat '{nom_plat}', tècnica '{nom_tecnica}' → score {score}")

        if score >= MIN_SCORE:
            resultats.append({'nom': nom_tecnica, 'score': score})

    if not resultats:
        print(
            f"[TRIA TECNICA] Cap tècnica de l'estil '{nom_estil}' ha superat el mínim "
            f"({MIN_SCORE}) per a '{nom_plat}'."
        )
        return []

    # 2) Ordenem per score
    resultats.sort(key=lambda x: x['score'], reverse=True)

    # 3) Assignem un ingredient (o curs) diferent a cada tècnica si és possible
    tecniques_tria = []
    ingredients_usats = set()

    for r in resultats:
        nom_tecnica = r['nom']
        tec_row = base_tecnniques.get(nom_tecnica)
        if tec_row is None:
            continue

        rao = _troba_ingredient_aplicable(tec_row, plat, info_ings, ingredients_usats)
        tecniques_tria.append((nom_tecnica, rao))

    print(
        f"[TRIA TECNICA] Escollides {len(tecniques_tria)} tècniques per al plat '{nom_plat}'. "
        f"Millor: '{tecniques_tria[0][0]}' (Score: {resultats[0]['score']})."
    )

    return tecniques_tria


def justifica_tecniques_aplicades(plat_modificat, tecniques_aplicades_amb_rao, base_tecnniques):
    """
    Genera el text de justificació dels canvis de tècnica aplicats,
    incloent l'ingredient o curs que ha permès l'aplicació.

    Args:
        plat_modificat (dict): El plat amb les tècniques assignades.
        tecniques_aplicades_amb_rao (list[(nom_tecnica, raó_aplicació)])
        base_tecnniques (dict): Diccionari de tecniques.csv.

    Retorna:
        str: Text detallat de justificació.
    """
    if not tecniques_aplicades_amb_rao:
        return "Cap tècnica nova aplicada (Score < Mínim)."

    log = []

    # 1) Tècnica principal
    nova_principal_nom = plat_modificat.get("tecnica_principal")
    rao_principal = next(
        (r[1] for r in tecniques_aplicades_amb_rao if r[0] == nova_principal_nom),
        'raó_desconeguda'
    )
    tec_row = base_tecnniques.get(nova_principal_nom, {})

    log.append(f"Tècnica principal assignada: **{nova_principal_nom}** (millor score).")
    log.append(
        f"Es justifica en **{rao_principal}**, on s'aplica el concepte: "
        f"'{tec_row.get('descripcio', 'sense descripció')}'"
    )

    # 2) Tècniques secundàries
    secundaries_noms = plat_modificat.get("tecnniques_secundaries", [])
    log.append("\n**Tècniques Secundàries aplicables:**")
    for nom_sec in secundaries_noms:
        if nom_sec == nova_principal_nom:
            continue
        rao_sec = next(
            (r[1] for r in tecniques_aplicades_amb_rao if r[0] == nom_sec),
            'raó_desconeguda'
        )
        log.append(f" - Afegida '{nom_sec}', aplicable a **{rao_sec}**.")

    return "\n".join(log)


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
