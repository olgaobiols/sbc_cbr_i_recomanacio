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
        self.ingredients: Dict[str, Dict] = {}  # {nom_normalitzat: fila_csv}
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
        print(
            f"✅ [KnowledgeBase] Ontologia carregada: "
            f"{len(self.ingredients)} ingredients, {len(self.estils)} estils."
        )

    def _normalize(self, text: str) -> str:
        """Normalització estàndard per a claus de diccionari."""
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", str(text))
        text = text.encode("ascii", "ignore").decode("ascii")
        return text.strip().lower().replace("-", " ").replace("_", " ")

    def _carregar_ingredients(self):
        path = os.path.join(self.data_dir, "ingredients_en.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                nom_keys = ["nom_ingredient", "ingredient_name", "name"]
                # Try to match the column name present in the CSV
                if reader.fieldnames:
                    try:
                        nom_idx = next(
                            key for key in nom_keys if key in reader.fieldnames
                        )
                    except StopIteration:
                        raise KeyError(
                            f"No s'ha trobat cap columna de nom a {path}. "
                            f"Esperat alguna de {', '.join(nom_keys)}"
                        )
                else:
                    raise KeyError(
                        f"No s'han detectat columnes al fitxer {path}. "
                        "Comprova el CSV."
                    )
                for row in reader:
                    nom = self._normalize(row.get(nom_idx, ""))
                    if nom:
                        self.ingredients[nom] = row
        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")
        except KeyError as exc:
            print(f"⚠️ [KnowledgeBase] Error carregant ingredients: {exc}")

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
        except FileNotFoundError:
            pass

    def _carregar_latents(self):
        path = os.path.join(self.data_dir, "estils_latents.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.estils_latents = json.load(f)
        except FileNotFoundError:
            pass

    # --- API Pública per al Cicle CBR ---

    def get_info_ingredient(self, nom: str) -> Optional[Dict]:
        row = self.ingredients.get(self._normalize(nom))
        if not row:
            return None

        # Retornem una còpia + camps "compatibles" amb els operadors antics
        out = dict(row)

        # Alias del nom
        if "ingredient_name" not in out:
            out["ingredient_name"] = out.get("nom_ingredient", "")

        # Alias de categoria macro
        if "macro_category" not in out:
            out["macro_category"] = out.get("categoria_macro", "")

        # Alias de família
        if "family" not in out:
            out["family"] = out.get("familia", "")

        return out

    def get_info_estil(self, nom_estil: str) -> Optional[Dict]:
        """Retorna la informació d'un estil culinari."""
        return self.estils.get(nom_estil)
    
    def get_info_tecnica(self, nom_tecnica: str) -> Optional[Dict]:
        """Retorna la informació d'una tècnica culinària."""
        return self.tecniques.get(nom_tecnica)

    def llista_estils(self) -> List[str]:
        """
        Mostra per pantalla els estils culinaris disponibles, numerats,
        a partir del fitxer estils.csv carregat a la base de coneixement.

        La llista s'actualitza automàticament en afegir nous estils
        al CSV i tornar a executar el programa.
        """
        noms_estils = list(self.estils.keys())

        if not noms_estils:
            print("⚠️ [KnowledgeBase] No hi ha estils disponibles.")
            return []

        noms_estils.sort()

        print("\n Estils culinaris disponibles:")
        for i, nom in enumerate(noms_estils, start=1):
            print(f"  {i:>2}. {nom}")

        return noms_estils
