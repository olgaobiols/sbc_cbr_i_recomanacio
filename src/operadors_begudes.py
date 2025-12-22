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

def passa_filtre_dur(plat, beguda_row):
    curs = plat.get("curs", "")
    ordre = beguda_row.get("maridatge_ordre", "").strip()
    
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
    if alcohol != alcohol_beguda:
        return False

    # 2. Cap restricció pot estar als al·lèrgens
    for restriccio in restriccions:
        if restriccio in alergens_beguda:
            return False

    # 3. Totes les restriccions han d’estar a dietes
    for restriccio in restriccions:
        if restriccio not in dietes_beguda:
            return False

    return True
    
    

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
        if ingredient["familia"] in fam_beguda:
            score += 2

        # --- Macro categories ---
        macro_beguda = set(beguda_row["va_be_amb_categoria_macro"].split("|"))
        if ingredient["categoria_macro"] in macro_beguda:
            score += 2

        # --- Sabors ---
        sabors_beguda = set(beguda_row["va_be_amb_sabors"].split("|"))
        evita_sabors = set(beguda_row["evita_sabors"].split("|"))
        sabors_ing = set(ingredient["sabors_base"].split("|"))

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

def recomana_beguda_per_plat(plat, base_begudes, base_ingredients, restriccions, alcohol):
    candidates = []
    restriccions_permeses = ['vegan', 'vegetarian', 'kosher_friendly', 'halal_friendly']
    restriccions_beguda = [r for r in restriccions if r in restriccions_permeses]
    
    ing_main, llista_ing = get_ingredient_principal(plat, base_ingredients)

    for row in base_begudes:
        if not passa_filtre_dur(plat, row):
            continue
        if not passa_restriccions(row, restriccions_beguda, alcohol):
            continue
        candidates.append(row)

    if not candidates:
        return None, None

    millor = None
    millor_score = float("-inf")

    for row in candidates:
        sc = score_beguda_per_plat(row, ing_main, llista_ing)
        if sc > millor_score:
            millor = row
            millor_score = sc

    return millor, millor_score
