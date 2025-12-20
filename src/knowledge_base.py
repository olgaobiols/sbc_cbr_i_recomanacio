import csv
import json
import os
import unicodedata
from typing import Dict, List, Optional

class KnowledgeBase:
    """
    Implementació del patró Singleton per gestionar l'Ontologia i el Coneixement del Domini
    tal com es defineix al Capítol 3 (Representació del Coneixement).
    Carrega els vocabularis controlats i les jerarquies d'ingredients.
    """
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance._inicialitzat = False
        return cls._instance

    def __init__(self):
        if self._inicialitzat:
            return
        
        # Estructures de dades en memòria
        self.ingredients: Dict[str, Dict] = {} # {nom_normalitzat: fila_csv}
        self.estils: Dict[str, Dict] = {}
        self.tecniques: Dict[str, Dict] = {}
        self.estils_latents: Dict = {}
        
        # Camins als fitxers de dades
        self.data_dir = "data" 
        
        # Càrrega
        self._carregar_ingredients()
        self._carregar_estils()
        self._carregar_tecniques()
        self._carregar_latents()
        
        self._inicialitzat = True
        print(f"✅ [KnowledgeBase] Ontologia carregada: {len(self.ingredients)} ingredients, {len(self.estils)} estils.")

    def _normalize(self, text: str) -> str:
        """Normalització estàndard per a claus de diccionari."""
        if not text: return ""
        text = unicodedata.normalize("NFKD", str(text))
        text = text.encode("ascii", "ignore").decode("ascii")
        return text.strip().lower().replace("-", " ").replace("_", " ")

    def _carregar_ingredients(self):
        path = os.path.join(self.data_dir, "ingredients.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nom = self._normalize(row["ingredient_name"])
                    if nom:
                        self.ingredients[nom] = row
        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")

    def _carregar_estils(self):
        path = os.path.join(self.data_dir, "estils.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.estils[row["nom_estil"]] = row
        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")

    def _carregar_tecniques(self):
        path = os.path.join(self.data_dir, "tecniques.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    self.tecniques[row["nom_tecnica"]] = row
        except FileNotFoundError: pass

    def _carregar_latents(self):
        path = os.path.join(self.data_dir, "estils_latents.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.estils_latents = json.load(f)
        except FileNotFoundError: pass

    # --- API Pública per al Cicle CBR ---
    
    def get_info_ingredient(self, nom: str) -> Optional[Dict]:
        """Retorna la info ontològica (categoria, rol, etc.) d'un ingredient."""
        return self.ingredients.get(self._normalize(nom))

    def get_info_estil(self, nom_estil: str) -> Optional[Dict]:
        return self.estils.get(nom_estil)
    
    def get_info_tecnica(self, nom_tecnica: str) -> Optional[Dict]:
        return self.tecniques.get(nom_tecnica)