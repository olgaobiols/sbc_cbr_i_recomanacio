import json
import math
from typing import Dict, Any, List, Optional
from estructura_cas import DescripcioProblema

class Retriever:
    """
    FASE 1: RETRIEVE (RECUPERACIÓ)
    Implementa l'estratègia Adaptation-Guided Retrieval per prioritzar casos
    segons la seva viabilitat d'adaptació i similitud.
    """

    # --- PESOS D'AGREGACIÓ GLOBAL ---
    W = {
        'event': 0.30, 
        'servei': 0.25, 
        'restr': 0.20,
        'temp': 0.10, 
        'formal': 0.10, 
        'pax': 0.03, 
        'preu': 0.02
    }

    # --- GRUPS SEMÀNTICS PER A SIMILITUD ---
    FAMILIARS = {"casament", "comunio", "comunió", "bateig", "aniversari", "reunio_familiar"}
    CORPORATIUS = {"empresa", "congres", "congrés"}
    INFORMALS = {"cocktail", "finger_food", "buffet"}
    SEASONS = ["primavera", "estiu", "tardor", "hivern"]

    def __init__(self, path_base_casos: str):
        self.base_casos = self._carregar_base_casos(path_base_casos)

    def _carregar_base_casos(self, path: str) -> List[Dict]:
        """Carrega la base de casos des del fitxer JSON."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            print(f"[Retriever]: No s'ha pogut carregar {path}")
            return []

    # --- UTILITATS ---

    def _norm(self, x: Any) -> str:
        """Normalitza cadenes de text per a comparacions."""
        return str(x).strip().lower() if x else ""

    # --- MÈTRIQUES DE SIMILITUD LOCAL ---

    def _sim_event(self, a: str, b: str) -> float:
        """Calcula similitud basada en categories d'esdeveniment."""
        a, b = self._norm(a), self._norm(b)
        if a == b: return 1.0
        if a in self.FAMILIARS and b in self.FAMILIARS: return 0.7
        if a in self.CORPORATIUS and b in self.CORPORATIUS: return 0.7
        return 0.2

    def _sim_servei(self, a: str, b: str) -> float:
        """Avalua la compatibilitat del tipus de servei."""
        a, b = self._norm(a), self._norm(b)
        if a == b or "indiferent" in (a, b): return 1.0
        if a in self.INFORMALS and b in self.INFORMALS: return 0.8
        return 0.2

    def _sim_restriccions(self, req_user: List[str], case_restr: List[str]) -> float:
        """Índex de Jaccard: (Intersecció / Unió). Penalitza la manca de seguretat."""
        u = {self._norm(r) for r in req_user if r}
        c = {self._norm(r) for r in case_restr if r}
        if not u: return 1.0  # Si l'usuari no té restriccions, el cas és 100% compatible
        
        unio = len(u | c)
        return len(u & c) / unio if unio > 0 else 1.0

    def _sim_temporada(self, a: str, b: str) -> float:
        """Distància Cíclica: Considera que Primavera i Hivern són adjacents."""
        a, b = self._norm(a), self._norm(b)
        if a == b or "indiferent" in (a, b): return 1.0
        try:
            idx_a, idx_b = self.SEASONS.index(a), self.SEASONS.index(b)
            dist = abs(idx_a - idx_b)
            # El valor màxim de distància en un cercle de 4 és 2
            return 1.0 - (min(dist, 4 - dist) * 0.5)
        except ValueError:
            return 0.4

    def _sim_preu(self, target: Any, actual: Any) -> float:
        """Penalització Asimètrica: Superar el pressupost penalitza exponencialment."""
        try:
            t, a = float(target), float(actual)
            if t <= 0 or a <= t: return 1.0
            # Funció de decaïment si el preu real és superior a l'objectiu
            return max(0.1, math.exp(-4.0 * ((a - t) / t)))
        except (ValueError, TypeError):
            return 0.8

    def _sim_pax(self, target: Any, actual: Any) -> float:
        """Similitud logística basada en la diferència de comensals."""
        try:
            return 1.0 / (1.0 + 0.01 * abs(float(target) - float(actual)))
        except (ValueError, TypeError):
            return 0.5

    # --- AGREGACIÓ I RECUPERACIÓ ---

    def _score(self, req: Any, cas: Dict) -> Dict[str, Any]:
        """Calcula el score global ponderat entre una petició i un cas de la base."""
        p = cas.get("problema", {})
        r_d = req.to_dict() if hasattr(req, 'to_dict') else req
        
        # Càlcul de totes les mètriques locals
        sims = {
            'event': self._sim_event(r_d.get("tipus_esdeveniment"), p.get("tipus_esdeveniment")),
            'servei': self._sim_servei(r_d.get("servei"), p.get("servei")),
            'restr': self._sim_restriccions(r_d.get("restriccions", []), p.get("restriccions", [])),
            'temp': self._sim_temporada(r_d.get("temporada"), p.get("temporada")),
            'formal': 1.0 if self._norm(r_d.get("formalitat")) == self._norm(p.get("formalitat")) else 0.5,
            'pax': self._sim_pax(r_d.get("n_comensals"), p.get("n_comensals")),
            'preu': self._sim_preu(r_d.get("preu_pers_objectiu"), p.get("preu_pers_objectiu", p.get("preu_pers")))
        }

        # Agregació ponderada segons els pesos definits a self.W
        score_total = sum(self.W[k] * sims[k] for k in self.W)
        
        return {
            "score_final": score_total, 
            "detall": sims
        }

    def recuperar_casos_similars(self, peticio: DescripcioProblema, k: int = 3) -> List[Dict]:
        """Retorna els top-k casos (k-NN) ordenats per similitud decreixent."""
        scored = []
        for cas in self.base_casos:
            res = self._score(peticio, cas)
            scored.append({**res, "cas": cas})
        
        # Ordenem per score_final de més a menys similar
        scored.sort(key=lambda x: x["score_final"], reverse=True)
        return scored[:k]