import random

def substituir_ingredient(plat, tipus_cuina, base_ingredients, base_cuina):
    """
    Substitueix un ingredient d'un plat que NO sigui de l'estil de cuina desitjat
    per un altre ingredient amb el mateix rol i disponible a la temporada.
    
    Args:
        plat (dict): {'nom': 'Nom del plat', 'ingredients': [...]}
        tipus_cuina (str): 'japonesa', 'francesa', etc.
        base_ingredients (list): Llista de diccionaris amb nom, categoria, rol, disponibilitat
        base_cuina (dict): Diccionari amb ingredients propis de cada estil
        
    Returns:
        dict: Plat amb un ingredient substituït
    """
    
    # Ingredients propis de l'estil
    ingredients_estil = set(base_cuina.get(tipus_cuina, {}).get('ingredients', []))
    
    # Ingredients del plat que NO són de l'estil
    ingredients_a_substituir = [ing for ing in plat['ingredients'] if ing not in ingredients_estil]
    
    if not ingredients_a_substituir:
        print(f"Tots els ingredients ja són de l'estil {tipus_cuina}.")
        return plat
    
    # Escull un ingredient a substituir
    ingredient_vell = random.choice(ingredients_a_substituir)
    
    # Troba el rol de l'ingredient a substituir
    rol = next((ing['rol_tipic'] for ing in base_ingredients if ing['nom_ingredient'] == ingredient_vell), None)
    if not rol:
        print(f"No s'ha trobat rol_tipic per {ingredient_vell}, no es pot substituir.")
        return plat
    
    alternatives = [ing['nom_ingredient'] for ing in base_ingredients
                    if ing['rol_tipic'] == rol and ing['nom_ingredient'] in ingredients_estil]
    
    if not alternatives:
        print(f"No hi ha alternatives per substituir {ingredient_vell} dins l'estil {tipus_cuina}.")
        return plat
    
    nou_ingredient = random.choice(alternatives)
    
    # Substitueix l'ingredient
    nou_plat = plat.copy()
    nou_plat['ingredients'] = [nou_ingredient if ing == ingredient_vell else ing 
                               for ing in plat['ingredients']]
    
    print(f"Substituint '{ingredient_vell}' per '{nou_ingredient}' al plat '{plat['nom']}'")
    return nou_plat

    









def modifica_tecnica(plat, nova_tecnica):
    """ Modifica la tècnica de cocció d'un plat. """
    pass

def transferir_estil(plat, nou_estil):
    """ Transfereix un plat a un nou estil culinari. """
    pass

def comprova_equilibri(plat): 
    pass