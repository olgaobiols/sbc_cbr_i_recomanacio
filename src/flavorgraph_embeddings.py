import pickle
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
from typing import List, Dict, Tuple, Optional

class FlavorGraphWrapper:
    def __init__(self, 
                 model_path: str = "models/FlavorGraph_Node_Embedding.pickle",
                 nodes_path: str = "models/nodes_191120.csv"):
        
        print(f"--- INICIALITZANT FLAVORGRAPH (AMB FILTRE CULINARI) ---")
        
        # 1. Carreguem Embeddings
        try:
            with open(model_path, "rb") as f:
                self.raw_embeddings = pickle.load(f)
        except FileNotFoundError:
            print(f"ERROR: No trobo {model_path}")
            self.raw_embeddings = {}
            return

        # 2. Carreguem Noms i TIPUS de nodes
        # El fitxer nodes_191120.csv sol tenir: node_id, name, type (o is_hub)
        print(f"2. Carregant metadades de nodes des de {nodes_path}...")
        self.name_to_vector = {}
        self.valid_ingredients = set() # Aquí guardarem només el que sigui menjar real
        
        try:
            df = pd.read_csv(nodes_path)
            df.columns = [c.lower() for c in df.columns] # Normalitzem columnes
            
            # Intentem detectar la columna de tipus
            # FlavorGraph sol marcar ingredients com 'ingredient' i químics com 'compound'
            col_id = next((c for c in df.columns if 'id' in c), None)
            col_name = next((c for c in df.columns if 'name' in c), None)
            col_type = next((c for c in df.columns if 'type' in c or 'category' in c), None)

            for index, row in df.iterrows():
                node_id = str(row[col_id])
                name = str(row[col_name]).lower().replace("_", " ")
                
                # FILTRE DE QUALITAT:
                # 1. Si tenim columna tipus, descartem 'compound'
                is_chemical = False
                if col_type and 'compound' in str(row[col_type]).lower():
                    is_chemical = True
                
                # 2. Heurística per si no trobem la columna tipus:
                # Els químics solen tenir noms llargs, números i guions (ex: 1,2-ethan...)
                if not col_type:
                    if any(char.isdigit() for char in name) and len(name) > 15:
                        is_chemical = True
                    if "," in name and "-" in name: # Ex: 2,4-decadienal
                        is_chemical = True

                # Guardem només si tenim vector i NO és un químic
                if node_id in self.raw_embeddings and not is_chemical:
                    self.name_to_vector[name] = self.raw_embeddings[node_id]
                    self.valid_ingredients.add(name)
                    
            print(f"FET! {len(self.valid_ingredients)} ingredients culinaris carregats (Químics descartats).")

        except Exception as e:
            print(f"ERROR llegint CSV: {e}")

    def get_vector(self, ingredient_name: str) -> Optional[np.ndarray]:
        term = ingredient_name.lower().replace("_", " ").strip()
        if term in self.name_to_vector: return self.name_to_vector[term]
        for name, vector in self.name_to_vector.items():
            if term in name.split(): return vector
        return None

    def find_similar(self, ingredient_name: str, n: int = 10) -> List[Tuple[str, float]]:
        vec = self.get_vector(ingredient_name)
        if vec is None: return []
        
        # Ara busquem només dins dels ingredients vàlids
        return self._find_nearest_to_vector(vec, n, exclude_names=[ingredient_name])

    def apply_semantic_direction(self, ingredient_name: str, direction: str, intensity: float = 0.5, n: int = 10):
        # ... (Codi igual que abans, però defineix self.directions al __init__ si vols usar-lo)
        pass 

    def _find_nearest_to_vector(self, vector: np.ndarray, n: int, exclude_names: List[str] = []) -> List[Tuple[str, float]]:
        # Optimització: Només comparem amb ingredients vàlids
        names = list(self.name_to_vector.keys())
        matrix = np.array(list(self.name_to_vector.values()))
        
        sims = cosine_similarity(vector.reshape(1, -1), matrix)[0]
        sorted_indices = sims.argsort()[::-1]
        
        results = []
        count = 0
        for idx in sorted_indices:
            name = names[idx]
            # Filtre extra: No volem noms massa llargs (sovint són plats processats)
            if name not in exclude_names and sims[idx] < 0.999 and len(name) < 25:
                results.append((name, float(sims[idx])))
                count += 1
                if count >= n: break
        return results

    def get_creative_candidates(self, ingredient_name, n=50, temperature=0.0, style_vector=None):
        """
        Retorna candidats ajustant la 'bogeria' (temperature) i la direcció (style).
        temperature: 0.0 (conservador) a 1.0 (molt creatiu/aleatori)
        """
        vec = self.get_vector(ingredient_name)
        if vec is None: return []

        # 1. APLICAR ESTIL (VECTOR STEERING)
        # Si volem "Japonitzar", sumem el vector mitjà d'ingredients japonesos
        search_vector = vec
        if style_vector is not None:
            # Normalitzem per no perdre la noció de l'ingredient original
            search_vector = (vec * 0.7) + (style_vector * 0.3)

        # 2. BUSCAR ELS N VEÏNS (Ampliem la finestra segons la temperatura)
        # Si temp és alta, necessitem recuperar MÉS veïns per tenir on triar
        window_size = int(n * (1 + temperature * 2)) # Ex: Si n=10, temp=1 -> window=30
        candidates = self._find_nearest_to_vector(search_vector, n=window_size, exclude_names=[ingredient_name])

        # 3. SELECCIÓ ESTOCÀSTICA (TEMPERATURE)
        # En lloc de retornar els top N, retornem N escollits segons la temperatura
        
        if temperature < 0.1:
            return candidates[:n] # Retorna els millors tal qual
        
        # Com més alta la temp, més probabilitat d'agafar els de baix de la llista
        weighted_candidates = []
        import random
        
        # Barregem una mica la llista basada en la temperatura
        # Això és una simulació simple de sampling:
        num_to_pick = min(n, len(candidates))
        
        # Com més temperatura, més permetem agafar índexs llunyans
        indices_possibles = list(range(len(candidates)))
        
        seleccionats = []
        for _ in range(num_to_pick):
            # Amb temp alta, triem índexs més grans (menys similars)
            # Amb temp baixa, ens quedem prop del 0
            if not indices_possibles: break
            
            # Distribució exponencial inversa modificada per temp? 
            # Més fàcil: Random choice amb pesos
            idx = int(random.triangular(0, len(indices_possibles)-1, mode=0 + (temperature * len(indices_possibles))))
            idx = min(idx, len(indices_possibles)-1)
            
            real_idx = indices_possibles.pop(idx)
            seleccionats.append(candidates[real_idx])
            
        return seleccionats
    
    
# --- TEST RÀPID ---
if __name__ == "__main__":
    fg = FlavorGraphWrapper("models/FlavorGraph_Node_Embedding.pickle", "models/nodes_191120.csv")
    print("\n--- Similars a 'chicken' (Netejat) ---")
    for x in fg.find_similar("chicken"): print(x)