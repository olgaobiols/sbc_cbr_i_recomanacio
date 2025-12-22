import csv
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional

"""
BASE DE CONEIXEMENT (Singleton)
-------------------------------
Centralitza l'accés a l'ontologia del sistema (Ingredients, Estils, Tècniques).
Gestiona la càrrega de dades des de CSV/JSON i ofereix mètodes de consulta normalitzats.
Actua com a Single Source of Truth per a tot el sistema CBR.
"""

class KnowledgeBase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance._inicialitzat = False
        return cls._instance

    def __init__(self):
        if self._inicialitzat: return

        self.data_dir = "data"
        self.ingredients: Dict[str, Dict] = {}
        self.estils: Dict[str, Dict] = {}
        self.tecniques: Dict[str, Dict] = {}
        self.begudes: Dict[str, Dict] = {}
        self.estils_latents: Dict = {}

        # Càrrega massiva de dades
        self._carregar_ingredients()
        self._carregar_generic("estils.csv", self.estils, "nom_estil")
        self._carregar_generic("tecniques.csv", self.tecniques, "nom_tecnica")
        self._carregar_generic("begudes_en.csv", self.begudes, ["id", "nom"])
        self._carregar_latents()

        self._inicialitzat = True
        print(f"Ontologia carregada: {len(self.ingredients)} ingredients, {len(self.estils)} estils.")

    def _normalize(self, text: str) -> str:
        """Normalitza text (ASCII, minúscules) per a claus de diccionari robustes."""
        if not text: return ""
        text = unicodedata.normalize("NFKD", str(text)).encode("ascii", "ignore").decode("ascii")
        return text.strip().lower().replace("-", " ").replace("_", " ")

    # GESTIÓ DE DADES (Loaders)
    def _carregar_csv(self, filename: str, target_dict: Dict, key_field: Any, normalize_key: bool = False):
        """Helper genèric per carregar CSVs amb gestió d'errors i claus flexibles."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            print(f"[KnowledgeBase] Arxiu no trobat: {filename}")
            return

        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames: return

                # Detecció de la columna clau (permet llistes de candidats com ['id', 'nom'])
                candidates = [key_field] if isinstance(key_field, str) else key_field
                key_col = next((k for k in candidates if k in reader.fieldnames), None)
                
                if not key_col: return # Si no troba cap clau vàlida, salta

                for row in reader:
                    val = row.get(key_col, "")
                    if val:
                        k = self._normalize(val) if normalize_key else val.strip()
                        target_dict[k] = row
        except Exception as e:
            print(f"[KnowledgeBase] Error llegint {filename}: {e}")

    def _carregar_generic(self, filename: str, target: Dict, key: Any):
        """Wrapper simple pel loader genèric."""
        self._carregar_csv(filename, target, key, normalize_key=False)

    def _carregar_ingredients(self):
        """Càrrega específica d'ingredients (requereix normalització de clau)."""
        keys = ["nom_ingredient", "ingredient_name", "name"]
        self._carregar_csv("ingredients_en.csv", self.ingredients, keys, normalize_key=True)

    def _carregar_latents(self):
        path = os.path.join(self.data_dir, "estils_latents.json")
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                self.estils_latents = json.load(f)

    # RETAIN (Integració CBR)
    def retain_case(self, new_case: Dict, evaluation_result: str, transformation_log: List[str], 
                   user_score: int, retriever_instance: Any) -> bool:
        """
        Delega la lògica de retenció al mòdul Retain.
        Importem dins la funció per evitar Cicles de Dependència (Retain -> KB -> Retain).
        """
        from Retain import retain_case as retain_case_impl
        return retain_case_impl(self, new_case, evaluation_result, transformation_log, user_score, retriever_instance)

    # -------------------------
    # API DE CONSULTA (Getters)
    # -------------------------
    def get_info_ingredient(self, nom: str) -> Optional[Dict]:
        """Retorna metadades normalitzades d'un ingredient."""
        row = self.ingredients.get(self._normalize(nom))
        if not row: return None

        # Normalització de camps per garantir consistència al codi
        out = row.copy()
        mappings = {
            "ingredient_name": ["nom_ingredient", "name"],
            "macro_category": ["categoria_macro"],
            "family": ["familia"]
        }
        for std_key, alt_keys in mappings.items():
            if std_key not in out:
                found = next((out.get(k) for k in alt_keys if out.get(k)), "")
                out[std_key] = found
        return out

    def get_info_estil(self, nom_estil: str) -> Optional[Dict]:
        return self.estils.get(nom_estil)

    def get_info_tecnica(self, nom_tecnica: str) -> Optional[Dict]:
        return self.tecniques.get(nom_tecnica)

    # HELPERS D'ESTILS I LATENTS
    def llista_estils_per_tipus(self, tipus: str) -> List[str]:
        target = (tipus or "").strip().lower()
        return sorted([
            nom for nom, row in self.estils.items() 
            if (row.get("tipus") or "").strip().lower() == target])

    def imprimir_estils_per_tipus(self, tipus: str) -> List[str]:
        estils = self.llista_estils_per_tipus(tipus)
        print(f"\n Estils disponibles ({tipus}):")
        if not estils: print("   (cap)")
        else:
            for i, nom in enumerate(estils, start=1): print(f"   {i}. {nom}")
        return estils

    def get_sabors_estil(self, nom_estil: str) -> List[str]:
        if row := self.estils.get(nom_estil):
            return [x.strip().lower() for x in str(row.get("sabors_clau", "")).split("|") if x.strip()]
        return []

    def suggerir_estils_culturals_per_latent(self, latent: str, top_k: int = 6) -> List[str]:
        """Troba estils culturals que encaixin amb un sabor/concepte latent."""
        latent = (latent or "").strip().lower()
        if not latent: return []

        candidats = []
        for nom, row in self.estils.items():
            if (row.get("tipus") or "").strip().lower() != "cultural": continue
            
            sabors = self.get_sabors_estil(nom)
            score = 0
            if latent in sabors: score = 2
            elif any(latent in s for s in sabors): score = 1
            
            if score > 0: candidats.append((score, nom))

        candidats.sort(key=lambda x: x[0], reverse=True)
        return [nom for _, nom in candidats[:top_k]]