import random
from typing import List, Dict, Set, Any
from flavorgraph_embeddings import FlavorGraphWrapper

# Inicialitzem el motor d'IA un sol cop (o es pot passar com argument)
# Assegura't que els paths siguin correctes respecte on executes el main
FG_WRAPPER = FlavorGraphWrapper()

# ---------------------------------------------------------------------
# 1. FUNCIONS AUXILIARS (Gestió de Dades)
# ---------------------------------------------------------------------

def _get_info_ingredient(nom_ingredient: str, base_ingredients: List[Dict]) -> Dict:
    """
    Busca la informació completa d'un ingredient (categoria, família, etc.)
    dins de la llista de diccionaris carregada del CSV.
    """
    # Això es podria optimitzar fent un diccionari {nom: info} a l'inici del programa
    # però per simplicitat ho busquem iterativament aquí.
    for ing in base_ingredients:
        if ing['ingredient_name'].lower() == nom_ingredient.lower():
            return ing
    return {}

def _get_candidats_per_categoria(categoria: str, base_ingredients: List[Dict]) -> List[str]:
    """
    Retorna tots els noms d'ingredients que pertanyen a una mateixa categoria macro.
    Ex: Si categoria='proteina_animal', retorna ['pollastre', 'vedella', 'porc'...]
    """
    return [
        ing['ingredient_name'] 
        for ing in base_ingredients 
        if ing.get('macro_category') == categoria
    ]

# ---------------------------------------------------------------------
# 2. NUCLI DE L'OPERADOR (Lògica Híbrida)
# ---------------------------------------------------------------------

def substituir_ingredients_prohibits(
    plat: Dict[str, Any], 
    ingredients_prohibits: Set[str], 
    base_ingredients: List[Dict]
) -> Dict[str, Any]:
    """
    Revisa els ingredients del plat. Si en troba un de prohibit, el substitueix
    fent servir:
    1. ONTOLOGIA: Per mantenir la coherència (Carn per Carn, Fruita per Fruita).
    2. FLAVORGRAPH: Per trobar el més similar sensorialment dins la categoria.
    """
    nou_plat = plat.copy()
    nou_plat['ingredients'] = list(plat['ingredients']) # Copia per no modificar l'original
    log_canvis = []

    for i, ingredient_actual in enumerate(nou_plat['ingredients']):
        
        # Si l'ingredient està a la llista negra
        if ingredient_actual in ingredients_prohibits:
            
            # 1. Recuperem metadades (què és aquest ingredient?)
            info = _get_info_ingredient(ingredient_actual, base_ingredients)
            categoria = info.get('categoria_macro') # Ex: 'proteina_animal', 'verdura'

            if not categoria:
                # Si no sabem què és, no ens arrisquem a canviar-lo malament
                print(f"[AVÍS] No tenim informació per substituir '{ingredient_actual}'. Es manté.")
                continue

            # 2. Generem candidats vàlids (Ontologia)
            possibles_candidats = _get_candidats_per_categoria(categoria, base_ingredients)
            
            # Filtrem: no podem substituir pel mateix que treiem, ni per un altre prohibit!
            candidats_filtrats = [
                c for c in possibles_candidats 
                if c not in ingredients_prohibits and c != ingredient_actual
            ]

            if not candidats_filtrats:
                print(f"[AVÍS] No hi ha substituts disponibles per '{ingredient_actual}' a la categoria '{categoria}'.")
                continue

            # 3. Rànquing intel·ligent (FlavorGraph)
            # Busquem quin dels candidats s'assembla més a l'original
            ranking = FG_WRAPPER._find_nearest_to_vector(
                vector=FG_WRAPPER.get_vector(ingredient_actual),
                n=5, # Mirem els top 5
                exclude_names=ingredients_prohibits # Seguretat extra
            )
            
            # Intersecció: Volem el millor del rànquing que TAMBÉ sigui de la categoria correcta
            millor_substitut = None
            
            # Primer intent: El millor candidat segons FlavorGraph que respecti l'ontologia
            for nom_candidat, score in ranking:
                if nom_candidat in candidats_filtrats:
                    millor_substitut = nom_candidat
                    break
            
            # Segon intent (Fallback): Si FlavorGraph no troba res proper a la categoria,
            # agafem un aleatori de la mateixa categoria (per assegurar estructura del plat).
            if not millor_substitut:
                millor_substitut = random.choice(candidats_filtrats)

            # 4. Aplicar canvi
            nou_plat['ingredients'][i] = millor_substitut
            log_canvis.append(f"Substitució: {ingredient_actual} -> {millor_substitut} (Categoria: {categoria})")

    # Guardem el registre de canvis al plat per explicar-ho a l'usuari després
    nou_plat['log_transformacio'] = log_canvis
    return nou_plat

# ---------------------------------------------------------------------
# 3. CAS D'ÚS ESPECÍFIC: ADAPTACIÓ A ESTIL
# ---------------------------------------------------------------------

def adaptar_plat_a_estil(plat: Dict, nom_estil: str, base_estils: Dict, base_ingredients: List[Dict]):
    """
    Wrapper que utilitza la funció genèrica de dalt.
    Determina quins ingredients sobren segons l'estil i els substitueix.
    """
    info_estil = base_estils.get(nom_estil)
    if not info_estil:
        return plat

    # Obtenim ingredients típics de l'estil
    ingredients_permesos = set(info_estil.get('ingredients', []))
    
    # Identifiquem els del plat que NO encaixen (ingredients prohibits en aquest context)
    # NOTA: Això és estricte. Si un ingredient no està a la definició de l'estil, es canvia.
    # Podries relaxar-ho comprovant només ingredients molt clau.
    prohibits = {
        ing for ing in plat['ingredients'] 
        if ing not in ingredients_permesos
    }
    
    if not prohibits:
        return plat # El plat ja encaixa

    print(f"--- Adaptant '{plat['nom']}' a l'estil {nom_estil} ---")
    return substituir_ingredients_prohibits(plat, prohibits, base_ingredients)