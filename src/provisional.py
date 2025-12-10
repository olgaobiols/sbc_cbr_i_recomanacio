import csv
from operadors_transformacio_realista import substituir_ingredient_amb_pairing
def carregar_base_ingredients(path="data/ingredients_en.csv"):
    base_ingredients = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            base_ingredients.append(row)
    return base_ingredients

if __name__ == "__main__":
    print("\n\n>>> TEST SUBSTITUCIÓ D'INGREDIENT AMB PAIRING <<<")

    # 1. Carreguem la base d'ingredients del domini
    base_ingredients = carregar_base_ingredients("data/ingredients_en.csv")

    plat_amanida = {
        "nom": "Amanida de pollastre amb sèsam",
        "ingredients": [
            "chicken",
            "soy_sauce",
            "ginger",
            "brown_sugar",
            "garlic",
            "mayonnaise",
            "sesame_oil",
            "white_sugar",
            "sesame",
            "cabbage",
            "carrot",
            "crispy_noodles"
        ],
        "estil_tags": [],
        "transformacions": []
    }

    print("\n=== TEST 2: Amanida, substitució vegana de chicken ===")
    print("Original:", plat_amanida["ingredients"])

    restriccions = {"vegetarian", "vegan"}  # ha de ser com a mínim una de les dues

    plat_amanida_veg = substituir_ingredient_amb_pairing(
        plat=plat_amanida.copy(),
        nom_ing_original="chicken",
        base_ingredients=base_ingredients,
        restriccions_usuari=restriccions,
        estil_objectiu=None, 
        temperatura=0.0
    )

    print("Adaptat :", plat_amanida_veg["ingredients"])
    for t in plat_amanida_veg.get("transformacions", []):
        print("  -", t.get("descripcio", ""))
