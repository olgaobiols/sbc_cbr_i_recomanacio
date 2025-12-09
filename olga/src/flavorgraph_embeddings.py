import pickle
import pandas as pd
import numpy as np
import random
import unicodedata
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

        self.alias_map = {
            "prawns": "shrimp",
            "prawn": "shrimp",
            "garbanzo": "chickpea",
            "garbanzo bean": "chickpea",
            "garbanzo beans": "chickpeas",
            "bell pepper": "red bell pepper",
            "bell peppers": "red bell pepper",
            "maize": "corn",
            "sweetcorn": "corn",
            "courgette": "zucchini",
            "aubergine": "eggplant",
        }

        for _, row in df.iterrows():
            node_id = str(row['node_id'])
            name = str(row['name'])
            node_type = str(row['type']).lower() if 'type' in df.columns else ""

            # Filtre: descartem químics i noms estranys
            is_chemical = 'compound' in node_type or (any(c.isdigit() for c in name) and len(name) > 10)

            if node_id in self.raw_embeddings and not is_chemical:
                normalized_name = self._normalize_term(name)
                if not normalized_name:
                    continue
                self.name_to_vector[normalized_name] = self.raw_embeddings[node_id]
                self.valid_ingredients.add(normalized_name)

        # 3. Preparem la cache per a cerques ràpides
        self._prepare_cache()

    def _prepare_cache(self):
        """Optimització: Matriu NumPy estàtica per càlcul vectorial massiu."""
        self.cached_names = list(self.name_to_vector.keys())
        if not self.cached_names:
            self.cached_matrix = np.empty((0, 0))
            self.cached_norms = np.array([])
            return
        self.cached_matrix = np.array(list(self.name_to_vector.values()))
        self.cached_norms = np.linalg.norm(self.cached_matrix, axis=1)

    def _normalize_term(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", str(text))
        text = text.encode("ascii", "ignore").decode("ascii")
        text = text.replace("-", " ").replace("_", " ").lower().strip()
        text = " ".join(text.split())
        return text

    def get_vector(self, ingredient_name: str) -> Optional[np.ndarray]:
        """Retorna el vector d'un ingredient (cerca exacta o parcial)."""
        term = self._normalize_term(ingredient_name)
        if not term:
            return None

        direct = self.name_to_vector.get(term)
        if direct is not None:
            return direct

        alias = self.alias_map.get(term)
        if alias:
            alias_norm = self._normalize_term(alias)
            if alias_norm in self.name_to_vector:
                return self.name_to_vector[alias_norm]

        # Proves simples: singular/plural i coincidència parcial
        if term.endswith("s"):
            singular = term[:-1]
            if singular in self.name_to_vector:
                return self.name_to_vector[singular]

        for name, vector in self.name_to_vector.items():
            if term == name:
                return vector
            if term in name.split():
                return vector
            if term in name:
                return vector

        return None

    def _normalize_vector(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if vector is None:
            return None
        norm = np.linalg.norm(vector)
        if norm == 0:
            return None
        return vector / norm

    def find_similar(self, ingredient_name: str, n: int = 10) -> List[Tuple[str, float]]:
        """Retorna els N ingredients més similars (mode conservador)."""
        vec = self.get_vector(ingredient_name)
        if vec is None: return []
        return self._find_nearest_to_vector(vec, n, exclude_names=[ingredient_name])

    def get_creative_candidates(
        self,
        ingredient_name: str,
        n: int = 10,
        temperature: float = 0.0,
        style_vector: Optional[np.ndarray] = None
    ) -> List[Tuple[str, float]]:   
        """
        Retorna candidats ajustant la 'bogeria' (temperature) i la direcció (style).
        - temperature: 0.0 (top N exactes) -> 1.0 (exploració àmplia).
        - style_vector: vector per modificar la direcció de cerca (ex: vector 'picant').
        """
        vec = self.get_vector(ingredient_name)
        if vec is None: return []

        base_vec = self._normalize_vector(vec)
        if base_vec is None:
            return []

        # 1. APLICAR ESTIL (VECTOR STEERING)
        search_vector = base_vec.copy()
        if style_vector is not None:
            style_vec = self._normalize_vector(style_vector)
            if style_vec is not None:
                # Factor de barreja depenent de la temperatura (com més alta, més estil)
                steer_strength = min(0.85, 0.25 + temperature * 0.6)
                combined = self._normalize_vector(
                    (1 - steer_strength) * base_vec + steer_strength * style_vec
                )
                search_vector = combined if combined is not None else base_vec

        # 2. DEFINIR FINESTRA DE CERCA
        temperature = max(0.0, min(1.0, temperature))
        window_size = max(n * 2, int(n * (1 + temperature * 3)))

        # 2b. Afegeix soroll per escapar veïnatge local
        noise_level = 0.15 * temperature
        if noise_level > 0:
            noise = np.random.normal(0, noise_level, size=search_vector.shape)
            noisy = self._normalize_vector(search_vector + noise)
            search_vector = noisy if noisy is not None else search_vector

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

    def get_style_representatives(
        self,
        style_vector: Optional[np.ndarray],
        n: int = 5,
        exclude_names: Optional[List[str]] = None,
        candidate_pool: Optional[List[str]] = None
    ) -> List[Tuple[str, float]]:
        """
        Retorna els ingredients més propers al vector d'estil.
        Útil per afegir "tocs" d'un estil concret si el plat encara no hi arriba.
        Si es passa candidate_pool, només es consideren aquests ingredients.
        """
        norm_vec = self._normalize_vector(style_vector)
        if norm_vec is None:
            return []

        exclude_norm = set()
        if exclude_names:
            exclude_norm = {
                self._normalize_term(name)
                for name in exclude_names
                if name
            }

        if candidate_pool:
            scored = []
            seen = set()
            for name in candidate_pool:
                norm_name = self._normalize_term(name)
                if not norm_name or norm_name in exclude_norm or norm_name in seen:
                    continue
                vec = self.get_vector(name)
                vec_norm = self._normalize_vector(vec)
                if vec_norm is None:
                    continue
                sim = float(np.dot(vec_norm, norm_vec))
                scored.append((name, sim))
                seen.add(norm_name)
            scored.sort(key=lambda x: x[1], reverse=True)
            return scored[:n]

        pool = self._find_nearest_to_vector(norm_vec, max(n * 4, 10), list(exclude_norm))
        reps = []
        for name, score in pool:
            if name in exclude_norm:
                continue
            reps.append((name, score))
            exclude_norm.add(name)
            if len(reps) >= n:
                break
        return reps

    def _find_nearest_to_vector(self, target_vector: np.ndarray, n: int, exclude_names: List[str]) -> List[Tuple[str, float]]:
        """Càlcul massiu de similitud cosinus utilitzant la matriu cache."""
        if target_vector is None:
            return []

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

    def similarity_with_vector(self, ingredient_name: str, target_vector: np.ndarray) -> Optional[float]:
        """Retorna la similitud cosinus entre un ingredient i un vector arbitrari."""
        vec = self.get_vector(ingredient_name)
        if vec is None or target_vector is None:
            return None
        vec_norm = self._normalize_vector(vec)
        target_norm = self._normalize_vector(target_vector)
        if vec_norm is None or target_norm is None:
            return None
        return float(np.dot(vec_norm, target_norm))
    
    def compute_concept_vector(self, ingredient_names: List[str]) -> Optional[np.ndarray]:
        """
        Calcula el 'centre de gravetat' (mitjana) d'una llista d'ingredients.
        Això ens dona el vector d'un concepte (ex: vector 'picant', vector 'italia').
        """
        vectors = []
        for name in ingredient_names:
            v = self.get_vector(name)
            if v is not None:
                vectors.append(v)
        
        if not vectors:
            return None
        
        # Calculem la mitjana de tots els vectors (axis=0 vol dir per columnes)
        return np.mean(vectors, axis=0)

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
