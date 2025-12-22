def _first_present(row, keys):
    for key in keys:
        value = row.get(key)
        if value not in (None, ""):
            return value
    return ""

def _normalize_key(value):
    return str(value or "").strip().lower()

def _normalize_ingredient_row(ing_row):
    out = dict(ing_row)
    out["nom_ingredient"] = _first_present(out, ("nom_ingredient", "ingredient_name", "name"))
    out["rol_tipic"] = _first_present(out, ("rol_tipic", "typical_role", "role"))
    out["familia"] = _first_present(out, ("familia", "family"))
    out["categoria_macro"] = _first_present(out, ("categoria_macro", "macro_category"))
    out["sabors_base"] = _first_present(out, ("sabors_base", "base_flavors"))
    return out

def get_ingredient_principal(plat, base_ingredients):
    """Retorna l'ingredient del plat amb typical_role = main."""
    ingredient_principal = None
    llista_ingredients = []
    
    for ing in plat.get("ingredients", []):
        ing_key = _normalize_key(ing)
        for ing_row in base_ingredients:
            row_name = _normalize_key(
                _first_present(ing_row, ("nom_ingredient", "ingredient_name", "name"))
            )
            if row_name == ing_key:
                norm_row = _normalize_ingredient_row(ing_row)
                llista_ingredients.append(norm_row)
                if norm_row.get('rol_tipic') == "main":
                    ingredient_principal = norm_row
    
    # Fallback: si no hi ha ingredient principal, escollim el primer ingredient reconegut
    if ingredient_principal is None and llista_ingredients:
        ingredient_principal = llista_ingredients[0]

    return ingredient_principal, llista_ingredients

def passa_filtre_dur(plat, beguda_row, begudes_usades):
    curs = plat.get("curs", "")
    ordre = beguda_row.get("maridatge_ordre", "").strip()
    id = beguda_row.get("id", "").strip()
    
    # Si la beguda ja ha estat usada no passa:
    if id in begudes_usades:
        return False
    
    # Si la beguda és general passa directament
    if beguda_row.get("es_general", "").strip().lower() == "si":
        return True
        
    # ORDRE obligatori
    if curs == "primer":
        return ordre == "ordre-primer"
    elif curs == "segon":
        return ordre == "ordre-segon"
    elif curs == "postres":
        return ordre == "ordre-postres"
    else:
        return False

def passa_restriccions(beguda_row, restriccions, alcohol):
    alcohol_beguda = beguda_row.get("alcohol", "").strip()
    alergens_beguda = set(beguda_row["alergen"].split("|"))
    dietes_beguda = set(beguda_row["dietes"].split("|"))
    
     # 1. Comprovació alcohol
    if alcohol.lower() == "no" and alcohol_beguda == "si":
        return False

    # 2. Al·lèrgens: cap restricció d’al·lèrgens ha d’estar present
    for restriccio in restriccions:
        if restriccio.lower() in alergens_beguda:
            return False

    # 3. Dietes: si alguna restricció és tipus dieta, ha d’estar present a la beguda
    restriccions_dieta = {'vegan', 'vegetarian', 'kosher_friendly', 'halal_friendly'}
    for restriccio in restriccions:
        if restriccio.lower() in restriccions_dieta and restriccio.lower() not in dietes_beguda:
            return False

    return True
    
    

def score_beguda_per_plat(beguda_row, ingredient_principal, llista_ingredients):
    total_score = 0
    breakdown = {
        "ingredient_principal": None,
        "ingredients_secundaris": []
    }
    
    # ------------------------------
    # Funció interna per puntuar 1 ingredient
    # ------------------------------
    def score_per_ingredient(ingredient):
        if not ingredient:
            return 0, {}

        score = 0
        detalls = {
            "nom": ingredient.get("nom_ingredient"),
            "familia": None,
            "categoria_macro": None,
            "sabors_match": [],
            "sabors_conflicte": []
        }

        # --- Famílies ---
        fam_beguda = set(beguda_row["va_be_amb_familia"].split("|"))
        if ingredient["familia"] in fam_beguda:
            score += 2
            detalls["familia"] = ingredient["familia"]

        # --- Macro categories ---
        macro_beguda = set(beguda_row["va_be_amb_categoria_macro"].split("|"))
        if ingredient["categoria_macro"] in macro_beguda:
            score += 2
            detalls["categoria_macro"] = ingredient["categoria_macro"]

        # --- Sabors ---
        sabors_beguda = set(beguda_row["va_be_amb_sabors"].split("|"))
        evita_sabors = set(beguda_row["evita_sabors"].split("|"))
        sabors_ing = set(ingredient["sabors_base"].split("|"))

        match = sabors_ing & sabors_beguda
        conflict = sabors_ing & evita_sabors

        score += len(match)
        score -= len(conflict)

        detalls["sabors_match"] = sorted(match)
        detalls["sabors_conflicte"] = sorted(conflict)

        return score, detalls


    # ---------------------------------------------------------
    # 1) Puntuar ingredients normals
    # ---------------------------------------------------------
    for ing in llista_ingredients:
        sc, det = score_per_ingredient(ing)
        total_score += sc
        breakdown["ingredients_secundaris"].append(det)

    # ---------------------------------------------------------
    # 2) Puntuar ingredient principal (DOBLE)
    # ---------------------------------------------------------
    if ingredient_principal:
        sc, det = score_per_ingredient(ingredient_principal)
        det["pes_principal"] = "x2"
        det["subtotal"] = sc * 2
        total_score += sc * 2
        breakdown["ingredient_principal"] = det

    breakdown["total_score"] = total_score

    return total_score, breakdown

def recomana_beguda_per_plat(plat, base_begudes, base_ingredients, restriccions, alcohol, begudes_usades):
    candidates = []
    
    ing_main, llista_ing = get_ingredient_principal(plat, base_ingredients)

    for row in base_begudes:
        if not passa_filtre_dur(plat, row, begudes_usades):
            continue
        if not passa_restriccions(row, restriccions, alcohol):
            continue
        candidates.append(row)

    if not candidates:
        return None, None

    millor = None
    millor_score = float("-inf")

    for row in candidates:
        sc, breakdown = score_beguda_per_plat(row, ing_main, llista_ing)
        if sc > millor_score:
            millor = row
            millor_score = sc
            millor_breakdown = breakdown

    
    if millor is not None:
        begudes_usades.add(millor.get("id"))

    return millor, millor_score, millor_breakdown