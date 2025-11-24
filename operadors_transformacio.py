import random

def substituir_ingredient(plat, tipus_cuina, base_ingredients, base_cuina, temporada):
    """
    Substitueix un ingredient d'un plat que NO sigui de l'estil de cuina desitjat
    per un altre ingredient amb el mateix rol i disponible a la temporada.
    
    Args:
        plat (dict): {'nom': 'Nom del plat', 'ingredients': [...]}
        tipus_cuina (str): 'japonesa', 'francesa', etc.
        base_ingredients (list): Llista de diccionaris amb nom, categoria, rol, disponibilitat
        base_cuina (dict): Diccionari amb ingredients propis de cada estil
        temporada (str): 'primavera', 'estiu', 'tardor', 'hivern'
        
    Returns:
        dict: Plat amb un ingredient substituït
    """
    
    # Ingredients propis de l'estil
    ingredients_estil = set(base_cuina.get(tipus_cuina, {}).get('ingredients', []))
    
    # Ingredients del plat que NO són de l'estil
    ingredients_a_substituir = [ing for ing in plat['ingredients'] if ing not in ingredients_estil]
    
    if not ingredients_a_substituir:
        return plat
    
    # Escull un ingredient a substituir
    ingredient_vell = random.choice(ingredients_a_substituir)
    
    # Troba el rol de l'ingredient a substituir
    rol = next((ing['rol'] for ing in base_ingredients if ing['nom'] == ingredient_vell), None)
    if not rol:
        return plat
    
    # Busca alternatives amb el mateix rol, del mateix estil i disponibles a la temporada
    alternatives = [
        ing['nom'] for ing in base_ingredients
        if ing['rol'] == rol
        and ing['nom'] in ingredients_estil
        and temporada in ing['disponibilitat']
    ]
    
    if not alternatives:
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