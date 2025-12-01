import random
from src.flavorgraph_embeddings import FlavorGraphWrapper
print("Inicialitzant motor FlavorGraph...")
# Assegura't que el path al .pickle és correcte
FLAVOR_ENGINE = FlavorGraphWrapper("models/FlavorGraph_Node_Embedding.pickle")

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

def substituir_avancat(plat, nom_original, base_dades, temperatura=0.2, estil_vector=None):
    
    # 1. Recuperar Dades Ontològiques de l'Original
    info_orig = base_dades.get_info(nom_original)
    rol_orig = info_orig["Rol_Estructural"] # Ex: "Principal"
    cat_orig = info_orig["Categoria"]       # Ex: "Carn"
    
    # 2. Obtenir Candidats "Creatius" del FlavorGraph
    # Si la temperatura és alta (0.8), FlavorGraph ens pot donar "Cigrons" per "Pollastre"
    candidats = FLAVOR_ENGINE.get_creative_candidates(nom_original, temperature=temperatura, style_vector=estil_vector)
    
    millor_opcio = None
    
    for nom_candidat, score in candidats:
        info_cand = base_dades.get_info(nom_candidat)
        if not info_cand: continue
        
        # 3. EL GRAN FILTRE ONTOLÒGIC
        # A. Filtre Crític: EL ROL HA DE SER EL MATEIX
        # No podem canviar un Principal per un Condiment, per molt creatius que siguim
        if info_cand["Rol_Estructural"] != rol_orig:
            continue
            
        # B. Filtre Flexible: LA CATEGORIA
        if temperatura < 0.4:
            # Mode Conservador: Ha de ser la mateixa categoria (Carn -> Carn)
            if info_cand["Categoria"] != cat_orig:
                continue
        else:
            # Mode Creatiu: Acceptem canvi de categoria si el Rol és igual
            # Ex: Carn -> Llegum (els dos són Rol Principal)
            pass 
            
        # Si passa els filtres, tenim guanyador
        millor_opcio = nom_candidat
        break
        
    return millor_opcio

def substituir_ingredient_generatiu(plat, nom_ing_original, base_ingredients, restriccions_usuari, direccio_sabor=None):
    """
    Substitueix un ingredient utilitzant IA (FlavorGraph) per trobar candidats 
    i Regles (Heurístiques) per validar-los.
    
    Args:
        plat (dict): El diccionari del plat.
        nom_ing_original (str): Nom de l'ingredient a treure.
        base_ingredients (list): Llista de diccionaris (el teu CSV d'ingredients).
        restriccions_usuari (list): Llista d'ingredients prohibits o al·lèrgies.
        direccio_sabor (str, opcional): "spicy", "sweet", "fresh", etc. per modificar el perfil.
    """
    
    # 0. Preparació de dades
    # Creem un índex ràpid per buscar info d'ingredients (rols, etc.)
    db_ing_map = {row["nom_ingredient"].lower(): row for row in base_ingredients}
    
    info_original = db_ing_map.get(nom_ing_original.lower())
    if not info_original:
        print(f"  [Error] L'ingredient original '{nom_ing_original}' no és a la BD.")
        return plat # No podem fer res

    rol_original = info_original.get("rol", "Desconegut") # Ex: 'proteina', 'midó'
    print(f"  > Substituint {nom_ing_original} (Rol: {rol_original})...")

    # ---------------------------------------------------------
    # FASE 1: GENERACIÓ DE CANDIDATS (Neuro / Embeddings)
    # ---------------------------------------------------------
    if FLAVOR_ENGINE:
        if direccio_sabor:
            print(f"  > Aplicant direcció semàntica: {direccio_sabor}")
            candidats_raw = FLAVOR_ENGINE.apply_semantic_direction(nom_ing_original, direccio_sabor, intensity=0.6, n=30)
        else:
            # Si no hi ha direcció, busquem els més similars (substitució directa)
            candidats_raw = FLAVOR_ENGINE.find_similar(nom_ing_original, n=30)
    else:
        candidats_raw = [] # Fallback si no tenim IA

    # Extraiem només els noms dels candidats suggerits per la IA
    noms_candidats = [c[0] for c in candidats_raw]
    
    # Si la IA falla o no troba res, afegim ingredients random de la BD com a fallback
    if not noms_candidats:
        noms_candidats = list(db_ing_map.keys())
        random.shuffle(noms_candidats)

    # ---------------------------------------------------------
    # FASE 2: FILTRATGE PER HEURÍSTIQUES DURES (Simbòlic)
    # ---------------------------------------------------------
    millor_candidat = None
    motiu_descart = ""

    for nom_cand in noms_candidats:
        info_cand = db_ing_map.get(nom_cand)
        
        # 1. Existeix a la nostra BD de cuina? (La IA pot proposar coses rares)
        if not info_cand:
            continue 

        # 2. HEURÍSTICA DE ROL (La més important)
        # El substitut ha de complir la mateixa funció estructural (Proteïna per Proteïna)
        if info_cand.get("rol") != rol_original:
            continue 

        # 3. RESTRICCIONS D'USUARI (Al·lèrgies/Gustos)
        # Si és prohibit, fora.
        es_prohibit = False
        for rest in restriccions_usuari:
            if rest.lower() in nom_cand.lower() or rest.lower() in info_cand.get("categoria", "").lower():
                es_prohibit = True
                break
        
        if es_prohibit:
            continue

        # 4. EVITAR REPETICIONS
        if nom_cand in plat["ingredients"]:
            continue

        # Si passem tots els filtres, tenim un guanyador!
        millor_candidat = nom_cand
        print(f"  > Candidat acceptat: {millor_candidat} (Score IA alt i compleix Rols/Restriccions)")
        break
    
    # ---------------------------------------------------------
    # FASE 3: APLICACIÓ DEL CANVI
    # ---------------------------------------------------------
    if millor_candidat:
        # Fem la substitució a la llista d'ingredients del plat
        nous_ingredients = [millor_candidat if x == nom_ing_original else x for x in plat["ingredients"]]
        plat["ingredients"] = nous_ingredients
        
        # Registrem el canvi per l'explicació
        if "transformacions" not in plat: plat["transformacions"] = []
        
        explicacio = f"Substitució de {nom_ing_original} per {millor_candidat}"
        if direccio_sabor: explicacio += f" per aportar un toc {direccio_sabor}"
        
        plat["transformacions"].append(explicacio)
        return plat
    else:
        print("  [Avís] No s'ha trobat cap substitut vàlid que compleixi les regles.")
        return plat













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
