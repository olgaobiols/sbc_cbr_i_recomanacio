import pickle
import sys

PATH_MODEL = "models/FlavorGraph_Node_Embedding.pickle"

print(f"--- INSPECCIONANT: {PATH_MODEL} ---")

try:
    with open(PATH_MODEL, "rb") as f:
        data = pickle.load(f)
        
    print(f"Tipus de dades carregades: {type(data)}")
    
    if isinstance(data, dict):
        keys = list(data.keys())
        print(f"Total de claus: {len(keys)}")
        print("\n--- EXEMPLE DE LES PRIMERES 20 CLAUS (tal qual estan al fitxer) ---")
        for k in keys[:20]:
            print(f"'{k}'")
            
        print("\n--- BUSCANT 'chicken' ---")
        found = False
        for k in keys:
            if "chicken" in str(k).lower():
                print(f"Trobada coincidència parcial: '{k}'")
                found = True
                # Si en trobem molts, parem als 10 primers
                if found and "chicken" in str(keys[:10]): break
        
        if not found:
            print("NO S'HA TROBAT CAP CLAU QUE CONTINGUI 'chicken'.")
            
    else:
        print("ALERTA: El fitxer no és un diccionari! És un:", type(data))
        print("Això explicaria per què falla la cerca per nom.")

except FileNotFoundError:
    print("Error: No trobo el fitxer. Revisa la ruta.")
except Exception as e:
    print(f"Error obrint el pickle: {e}")