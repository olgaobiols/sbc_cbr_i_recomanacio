def _get_ing_field(ing_row, *keys):
    for key in keys:
        if key in ing_row:
            return ing_row[key]
    return None

def _norm_text(value):
    return str(value).strip().lower() if value is not None else ""

def get_ingredient_principal(plat, base_ingredients):
    """Retorna l'ingredient del plat amb typical_role = main."""
    ingredient_principal = None
    llista_ingredients = []
    
    for ing in plat.get("ingredients", []):
        ing_norm = _norm_text(ing)
        for ing_row in base_ingredients:
            nom = _get_ing_field(ing_row, "nom_ingredient", "ingredient_name", "name")
            if _norm_text(nom) == ing_norm:
                llista_ingredients.append(ing_row)
                rol = _get_ing_field(ing_row, "rol_tipic", "typical_role")
                if _norm_text(rol) == "main":
                    ingredient_principal = ing_row
    
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
        fam_beguda = set(beguda_row.get("va_be_amb_familia", "").split("|"))
        fam_ing = _get_ing_field(ingredient, "familia", "family")
        if fam_ing and fam_ing in fam_beguda:
            score += 2

        # --- Macro categories ---
        macro_beguda = set(beguda_row.get("va_be_amb_categoria_macro", "").split("|"))
        macro_ing = _get_ing_field(ingredient, "categoria_macro", "macro_category")
        if macro_ing and macro_ing in macro_beguda:
            score += 2

        # --- Sabors ---
        sabors_beguda = set(beguda_row.get("va_be_amb_sabors", "").split("|"))
        evita_sabors = set(beguda_row.get("evita_sabors", "").split("|"))
        sabors_ing_raw = _get_ing_field(ingredient, "sabors_base", "base_flavors")
        sabors_ing = set((sabors_ing_raw or "").split("|"))

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
    
    # 2. FILTRE DUR (amb 'es_general') per cada beguda
    for row in base_begudes:
        if passa_filtre_dur(plat, row):
            candidates.append(row)
    
    if not candidates:
        return None, None

    # 3. Escollir la millor beguda per scoring
    millor = None
    millor_score = -999
    for row in candidates:
        sc = score_beguda_per_plat(row, ing_main, llista_ing)
        
        if sc > millor_score:
            millor = row
            millor_score = sc

    return millor, millor_score
