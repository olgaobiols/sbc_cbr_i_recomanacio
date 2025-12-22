import csv
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional

from Retain import retain_case as retain_case_impl


class KnowledgeBase:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(KnowledgeBase, cls).__new__(cls)
            cls._instance._inicialitzat = False
        return cls._instance

    def __init__(self):
        if self._inicialitzat:
            return

        self.ingredients: Dict[str, Dict] = {}
        self.estils: Dict[str, Dict] = {}
        self.tecniques: Dict[str, Dict] = {}
        self.estils_latents: Dict = {}
        self.begudes: Dict[str, Dict] = {}

        self.data_dir = "data"

        self._carregar_ingredients()
        self._carregar_estils()
        self._carregar_tecniques()
        self._carregar_latents()
        self._carregar_begudes()

        self._inicialitzat = True
        print(
            f"✅ [KnowledgeBase] Ontologia carregada: "
            f"{len(self.ingredients)} ingredients, {len(self.estils)} estils."
        )

    def _normalize(self, text: str) -> str:
        if not text:
            return ""
        text = unicodedata.normalize("NFKD", str(text))
        text = text.encode("ascii", "ignore").decode("ascii")
        return text.strip().lower().replace("-", " ").replace("_", " ")

    # -------------------------
    # Carregues
    # -------------------------
    def _carregar_ingredients(self) -> None:
        path = os.path.join(self.data_dir, "ingredients_en.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)

                nom_keys = ["nom_ingredient", "ingredient_name", "name"]
                if not reader.fieldnames:
                    raise KeyError(f"No s'han detectat columnes a {path}")

                try:
                    nom_col = next(k for k in nom_keys if k in reader.fieldnames)
                except StopIteration:
                    raise KeyError(
                        f"No s'ha trobat columna de nom a {path}. "
                        f"Esperat alguna de {', '.join(nom_keys)}"
                    )

                for row in reader:
                    nom = self._normalize(row.get(nom_col, ""))
                    if nom:
                        self.ingredients[nom] = row

        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")
        except KeyError as exc:
            print(f"⚠️ [KnowledgeBase] Error carregant ingredients: {exc}")

    def _carregar_estils(self) -> None:
        path = os.path.join(self.data_dir, "estils.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nom = row.get("nom_estil")
                    if nom:
                        self.estils[nom] = row
        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")

    def _carregar_tecniques(self) -> None:
        path = os.path.join(self.data_dir, "tecniques.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    nom = row.get("nom_tecnica")
                    if nom:
                        self.tecniques[nom] = row
        except FileNotFoundError:
            pass

    def _carregar_latents(self) -> None:
        path = os.path.join(self.data_dir, "estils_latents.json")
        try:
            with open(path, "r", encoding="utf-8") as f:
                self.estils_latents = json.load(f)
        except FileNotFoundError:
            pass

    def _carregar_begudes(self) -> None:
        path = os.path.join(self.data_dir, "begudes_en.csv")
        try:
            with open(path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    key = (row.get("id") or row.get("nom") or "").strip()
                    if key:
                        self.begudes[key] = row
        except FileNotFoundError:
            print(f"⚠️ [KnowledgeBase] No s'ha trobat {path}")

    # -------------------------
    # RETAIN
    # -------------------------
    def retain_case(
        self,
        new_case: Dict,
        evaluation_result: str,
        transformation_log: List[str],
        user_score: int,
        retriever_instance: Any,
    ) -> bool:
        return retain_case_impl(
            self,
            new_case,
            evaluation_result,
            transformation_log,
            user_score,
            retriever_instance,
        )

    # -------------------------
    # API pública
    # -------------------------
    def get_info_ingredient(self, nom: str) -> Optional[Dict]:
        row = self.ingredients.get(self._normalize(nom))
        if not row:
            return None

        out = dict(row)

        if "ingredient_name" not in out:
            out["ingredient_name"] = out.get("nom_ingredient", "") or out.get("name", "")

        if "macro_category" not in out:
            out["macro_category"] = out.get("categoria_macro", "")

        if "family" not in out:
            out["family"] = out.get("familia", "")

        return out

    def get_info_estil(self, nom_estil: str) -> Optional[Dict]:
        return self.estils.get(nom_estil)

    def get_info_tecnica(self, nom_tecnica: str) -> Optional[Dict]:
        return self.tecniques.get(nom_tecnica)

    # -------------------------
    # Helpers d'estils (els teus)
    # -------------------------
    def llista_estils_per_tipus(self, tipus: str) -> List[str]:
        tipus = (tipus or "").strip().lower()
        result = []
        for nom, row in (self.estils or {}).items():
            t = (row.get("tipus") or "").strip().lower()
            if t == tipus:
                result.append(nom)
        return sorted(result)

    def imprimir_estils_per_tipus(self, tipus: str) -> List[str]:
        estils = self.llista_estils_per_tipus(tipus)
        print(f"\n Estils disponibles ({tipus}):")
        if not estils:
            print("   (cap)")
            return []
        for i, nom in enumerate(estils, start=1):
            print(f"   {i}. {nom}")
        return estils

    def _split_pipe(self, s: str) -> List[str]:
        if not s:
            return []
        return [x.strip().lower() for x in str(s).split("|") if x.strip()]

    def get_sabors_estil(self, nom_estil: str) -> List[str]:
        row = self.estils.get(nom_estil)
        if not row:
            return []
        return self._split_pipe(row.get("sabors_clau", ""))

    def suggerir_estils_culturals_per_latent(self, latent: str, top_k: int = 6) -> List[str]:
        latent = (latent or "").strip().lower()
        if not latent:
            return []

        candidats = []
        for nom_estil, row in (self.estils or {}).items():
            if not isinstance(row, dict):
                continue

            tipus = (row.get("tipus") or "").strip().lower()
            if tipus != "cultural":
                continue

            sabors = self._split_pipe(row.get("sabors_clau", ""))

            score = 0
            for s in sabors:
                if s == latent:
                    score = max(score, 2)
                elif latent in s:
                    score = max(score, 1)

            if score > 0:
                candidats.append((score, nom_estil))

        candidats.sort(key=lambda x: x[0], reverse=True)
        return [nom for _, nom in candidats[:top_k]]
