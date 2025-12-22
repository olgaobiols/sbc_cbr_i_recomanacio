import pickle
import pandas as pd
import numpy as np
import random
import unicodedata
from typing import List, Tuple, Optional
"""
GESTIÓ DEL NIVELL SUBSIMBÒLIC (FlavorGraph)
Implementa la representació latent del sistema.
Gestiona l'espai vectorial n-dimensional per calcular afinitats químiques
i generar candidats creatius segons Pairing.
"""

class FlavorGraphWrapper:
    def __init__(self, model_path: str = "models/FlavorGraph_Node_Embedding.pickle", nodes_path: str = "models/nodes_191120.csv"):
        # Carreguem Embeddings FlavorGraph
        with open(model_path, "rb") as f:
            self.raw_embeddings = pickle.load(f)
            
        # Carreguem Metadades i mapeig noms -> vectors
        df = pd.read_csv(nodes_path)
        df.columns = [c.lower() for c in df.columns]
        
        self.name_to_vector = {} 
        self.valid_ingredients = set()
        # Normalització manual per sinònims culinaris comuns
        self.alias_map = {
            "prawns": "shrimp", "prawn": "shrimp", "garbanzo": "chickpea",
            "garbanzo bean": "chickpea", "garbanzo beans": "chickpeas",
            "bell pepper": "red bell pepper", "bell peppers": "red bell pepper",
            "maize": "corn", "sweetcorn": "corn", "courgette": "zucchini", "aubergine": "eggplant",
        }

        for _, row in df.iterrows():
            node_id = str(row['node_id'])
            name = str(row['name'])
            node_type = str(row['type']).lower() if 'type' in df.columns else ""

            # Filtre de qualitat: descartem compostos químics purs o IDs numèrics estranys
            is_chemical = 'compound' in node_type or (any(c.isdigit() for c in name) and len(name) > 10)

            if node_id in self.raw_embeddings and not is_chemical:
                normalized_name = self._normalize_term(name)
                if normalized_name:
                    self.name_to_vector[normalized_name] = self.raw_embeddings[node_id]
                    self.valid_ingredients.add(normalized_name)

        # Preparem la cache per a cerques ràpides
        self._prepare_cache()

    def _prepare_cache(self):
            """Crea matrius NumPy estàtiques per càlcul vectorial massiu (eficiència)."""
            self.cached_names = list(self.name_to_vector.keys())
            if not self.cached_names:
                self.cached_matrix, self.cached_norms = np.empty((0, 0)), np.array([])
            else:
                self.cached_matrix = np.array(list(self.name_to_vector.values()))
                self.cached_norms = np.linalg.norm(self.cached_matrix, axis=1)
    
    def _normalize_term(self, text: str):
            """Neteja strings per garantir coincidències (ASCII, lowercase)."""
            if not text: return ""
            text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
            return " ".join(text.replace("-", " ").replace("_", " ").lower().split())

    def get_vector(self, ingredient_name: str) -> Optional[np.ndarray]:
        """Recupera el vector associat a un ingredient (cerca directa, aliàs o parcial)."""
        term = self._normalize_term(ingredient_name)
        if not term: return None
        
        # 1Cerca exacta o per aliàs
        if term in self.name_to_vector: return self.name_to_vector[term]
        if term in self.alias_map:
            alias = self._normalize_term(self.alias_map[term])
            if alias in self.name_to_vector: return self.name_to_vector[alias]
            
        # Heurístiques simples (singulars, substrings)
        if term.endswith("s") and term[:-1] in self.name_to_vector: return self.name_to_vector[term[:-1]]
        for name, vector in self.name_to_vector.items():
            if term == name or term in name.split() or term in name: return vector
        return None

    def _normalize_vector(self, vector: np.ndarray) -> Optional[np.ndarray]:
        if vector is None: return None
        norm = np.linalg.norm(vector)
        return vector / norm if norm != 0 else None

    def find_similar(self, ingredient_name: str, n: int = 10) -> List[Tuple[str, float]]:
        """Recuperació conservadora basada en similitud cosinus."""
        vec = self.get_vector(ingredient_name)
        return self._find_nearest_to_vector(vec, n, exclude_names=[ingredient_name]) if vec is not None else []

    def get_creative_candidates(self, ingredient_name: str, n: int = 10, temperature: float = 0.0, style_vector: Optional[np.ndarray] = None) -> List[Tuple[str, float]]:
        """
        Generació de candidats ajustant 'temperatura' (exploració) i 'estil' (direcció)[cite: 129].
        - Temperature: 0.0 (conservador) -> 1.0 (creatiu/arriscat).
        """
        base_vec = self._normalize_vector(self.get_vector(ingredient_name))
        if base_vec is None: return []

        # Vector Steering: Modificar direcció cap a un estil (ex: "fer-ho picant")
        search_vec = base_vec.copy()
        if style_vector is not None:
            style_vec = self._normalize_vector(style_vector)
            if style_vec is not None:
                steer = min(0.85, 0.25 + temperature * 0.6)
                steered = self._normalize_vector((1 - steer) * base_vec + steer * style_vec)
                if steered is not None:
                    search_vec = steered

        # Soroll Gaussià per escapar òptims locals 
        temperature = np.clip(temperature, 0.0, 1.0)
        if temperature > 0:
            noise = np.random.normal(0, 0.15 * temperature, size=search_vec.shape)
            noised = self._normalize_vector(search_vec + noise)
            if noised is not None:
                search_vec = noised

        # Recuperació i mostreig
        window_size = max(n * 2, int(n * (1 + temperature * 3)))
        pool = self._find_nearest_to_vector(search_vec, n=window_size, exclude_names=[ingredient_name])
        
        if temperature < 0.1: return pool[:n] # Determinista

        # Mostreig estocàstic triangular (prioritza millors scores però permet varietat)
        selected, indices = [], list(range(len(pool)))
        for _ in range(min(n, len(pool))):
            if not indices: break
            idx = int(random.triangular(0, len(indices) - 1, mode=temperature * (len(indices) - 1)))
            selected.append(pool[indices.pop(min(idx, len(indices) - 1))])
        return selected

    def get_style_representatives(self, style_vector: Optional[np.ndarray], n: int = 5, exclude_names: List[str] = None, candidate_pool: List[str] = None) -> List[Tuple[str, float]]:
        """Troba ingredients que millor representen un vector d'estil (ex: 'Italian')."""
        target = self._normalize_vector(style_vector)
        if target is None: return []
        
        excludes = {self._normalize_term(x) for x in (exclude_names or []) if x}

        if candidate_pool: # Filtrar sobre llista tancada
            scored = []
            seen = set()
            for name in candidate_pool:
                norm = self._normalize_term(name)
                if not norm or norm in excludes or norm in seen: continue
                v = self._normalize_vector(self.get_vector(name))
                if v is not None:
                    scored.append((name, float(np.dot(v, target))))
                    seen.add(norm)
            return sorted(scored, key=lambda x: x[1], reverse=True)[:n]

        # Cerca oberta a tota la base
        pool = self._find_nearest_to_vector(target, max(n * 4, 10), list(excludes))
        return [p for p in pool if p[0] not in excludes][:n]

    def _find_nearest_to_vector(self, target_vector: np.ndarray, n: int, exclude_names: List[str]) -> List[Tuple[str, float]]:
        """Nucli matemàtic: Càlcul de similitud cosinus vectoritzat."""
        if target_vector is None: return []
        if not hasattr(self, 'cached_matrix'): self._prepare_cache()

        # Dot product massiu
        sims = np.dot(self.cached_matrix, target_vector) / (self.cached_norms * np.linalg.norm(target_vector))
        sorted_idxs = sims.argsort()[::-1]
        
        results, count = [], 0
        for idx in sorted_idxs:
            name, score = self.cached_names[idx], float(sims[idx])
            if name not in exclude_names and score < 0.999 and len(name) < 25:
                results.append((name, score))
                count += 1
                if count >= n: break
        return results

    def similarity_with_vector(self, ingredient_name: str, target_vector: np.ndarray) -> Optional[float]:
        """Calcula distància semàntica entre un ingredient i un concepte."""
        v1, v2 = self._normalize_vector(self.get_vector(ingredient_name)), self._normalize_vector(target_vector)
        return float(np.dot(v1, v2)) if v1 is not None and v2 is not None else None
    
    def compute_concept_vector(self, ingredient_names: List[str]) -> Optional[np.ndarray]:
        """Genera el vector 'centre de masses' d'una llista d'ingredients (defineix un Estil/Concepte)."""
        vectors = [v for name in ingredient_names if (v := self.get_vector(name)) is not None]
        return np.mean(vectors, axis=0) if vectors else None