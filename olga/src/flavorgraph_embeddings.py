import pickle
import pandas as pd
import numpy as np
import random
from typing import List, Tuple, Optional

"""
La operació de canvi d'ingredients combina coneixement explícit (ontologia / regles)
amb machine learning (embeddings de FlavorGraph). 

Aquest fitxer actua com el cervell sensorial del sistema: sap quins ingredients s'assemblen
basant-se en la seva similaritat (pairing). 

INCLOU:
- Càrrega del model vectorial FlavorGraph (pickle).
- Càrrega del diccionari de noms (CSV).
- Filtratge per eliminar compostos químics i quedar-nos només amb menjar coherent segons ontologia.
- Funció per obtenir el vector d'un ingredient donat el seu nom.
"""

class FlavorGraphWrapper:
    def __init__(self, model_path: str = "models/FlavorGraph_Node_Embedding.pickle", nodes_path: str = "models/nodes_191120.csv"):
        
        # 1. Carreguem Embeddings
        with open(model_path, "rb") as f:
            self.raw_embeddings = pickle.load(f)

        # 2. Carreguem Metadades i Filtrem
        df = pd.read_csv(nodes_path)
        df.columns = [c.lower() for c in df.columns]
        
        self.name_to_vector = {} 
        self.valid_ingredients = set()

        for _, row in df.iterrows():
            node_id = str(row['node_id'])
            name = str(row['name']).lower().replace("_", " ")
            node_type = str(row['type']).lower() if 'type' in df.columns else ""

            # Filtre: descartem químics i noms estranys
            is_chemical = 'compound' in node_type or (any(c.isdigit() for c in name) and len(name) > 10)

            if node_id in self.raw_embeddings and not is_chemical:
                self.name_to_vector[name] = self.raw_embeddings[node_id]
                self.valid_ingredients.add(name)

        # 3. Preparem la cache per a cerques ràpides
        self._prepare_cache()

    def _prepare_cache(self):
        """Optimització: Matriu NumPy estàtica per càlcul vectorial massiu."""
        self.cached_names = list(self.name_to_vector.keys())
        self.cached_matrix = np.array(list(self.name_to_vector.values()))
        self.cached_norms = np.linalg.norm(self.cached_matrix, axis=1)

    def get_vector(self, ingredient_name: str) -> Optional[np.ndarray]:
        """Retorna el vector d'un ingredient (cerca exacta o parcial)."""
        term = ingredient_name.lower().replace("_", " ").strip()
        if term in self.name_to_vector: return self.name_to_vector[term]
        for name, vector in self.name_to_vector.items():
            if term in name.split(): return vector
        return None

    def find_similar(self, ingredient_name: str, n: int = 10) -> List[Tuple[str, float]]:
        """Retorna els N ingredients més similars (mode conservador)."""
        vec = self.get_vector(ingredient_name)
        if vec is None: return []
        return self._find_nearest_to_vector(vec, n, exclude_names=[ingredient_name])

    def get_creative_candidates(self, ingredient_name: str, n: int = 10, temperature: float = 0.0, style_vector: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Retorna candidats ajustant la 'bogeria' (temperature) i la direcció (style).
        - temperature: 0.0 (top N exactes) -> 1.0 (exploració àmplia).
        - style_vector: vector per modificar la direcció de cerca (ex: vector 'picant').
        """
        vec = self.get_vector(ingredient_name)
        if vec is None: return []

        # 1. APLICAR ESTIL (VECTOR STEERING)
        # Si tenim estil, barregem el vector original (70%) amb l'estil (30%)
        search_vector = vec
        if style_vector is not None:
            search_vector = (vec * 0.7) + (style_vector * 0.3)

        # 2. DEFINIR FINESTRA DE CERCA
        # Si temp és alta, recuperem molts més candidats (pool) per triar aleatòriament després
        # Ex: temp 0 -> window 10. Temp 1.0 -> window 30.
        window_size = int(n * (1 + temperature * 2))
        pool = self._find_nearest_to_vector(search_vector, n=window_size, exclude_names=[ingredient_name])

        # 3. SELECCIÓ (SAMPLING)
        if temperature < 0.1:
            return pool[:n] # Retorna els millors tal qual (determinista)

        # Selecció estocàstica basada en índexs
        selected = []
        indices_possibles = list(range(len(pool)))
        
        count = min(n, len(pool))
        for _ in range(count):
            if not indices_possibles: break
            
            # Lògica Triangular:
            # Mode 0 (temp baixa) -> tendeix a agafar els primers índexs (millors scores)
            # Mode alt (temp alta) -> tendeix a dispersar-se cap al final de la llista
            mode_val = temperature * (len(indices_possibles) - 1)
            idx = int(random.triangular(0, len(indices_possibles) - 1, mode=mode_val))
            
            # Protecció de límits i selecció
            idx = min(idx, len(indices_possibles) - 1)
            real_idx = indices_possibles.pop(idx) # Treiem l'índex per no repetir
            
            selected.append(pool[real_idx])
            
        return selected

    def _find_nearest_to_vector(self, target_vector: np.ndarray, n: int, exclude_names: List[str]) -> List[Tuple[str, float]]:
        """Càlcul massiu de similitud cosinus utilitzant la matriu cache."""
        if not hasattr(self, 'cached_matrix'): self._prepare_cache()

        dot_product = np.dot(self.cached_matrix, target_vector)
        target_norm = np.linalg.norm(target_vector)
        similarities = dot_product / (self.cached_norms * target_norm)

        sorted_indices = similarities.argsort()[::-1]
        
        results = []
        count = 0
        for idx in sorted_indices:
            name = self.cached_names[idx]
            score = float(similarities[idx])
            
            if (name not in exclude_names) and (score < 0.999) and (len(name) < 25):
                results.append((name, score))
                count += 1
                if count >= n: break
        return results

# --- TEST ---
if __name__ == "__main__":
    # Ajusta els paths segons on executis això
    try:
        fg = FlavorGraphWrapper("models/FlavorGraph_Node_Embedding.pickle", "models/nodes_191120.csv")
        print("Model carregat correctament.")
        
        ing = "chicken"
        print(f"\n--- Similars a '{ing}' (Conservador) ---")
        for x in fg.find_similar(ing, n=5): print(x)
        
        print(f"\n--- Similars a '{ing}' (Creatiu Temp=0.8) ---")
        for x in fg.get_creative_candidates(ing, n=5, temperature=0.8): print(x)
        
    except FileNotFoundError:
        print("Error: No es troben els fitxers de models. Assegura't de ser a la carpeta arrel.")