# src/retriever_nuevo.py
import json
import math
from dataclasses import asdict
from typing import Dict, Any, List
from estructura_cas import DescripcioProblema

class Retriever:
    """
    Retriever CBR avançat (Adaptation-Guided Retrieval).
    Implementa la lògica definida a la Secció 4.1 de la memòria:
      - Similitud Semàntica (Matriu d'Esdeveniments)
      - Similitud de Conjunts (Jaccard per Restriccions)
      - Similitud Cíclica (Temporada)
      - Similitud Asimètrica (Preu)
    """

    # --- PESOS DE L'AGREGACIÓ GLOBAL (Total = 1.0) ---
    # Justificació: Prioritzem l'estructura (Event/Servei) i la viabilitat (Restriccions)
    # per sobre dels detalls ambientals o numèrics fàcilment adaptables.
    W_EVENT      = 0.30
    W_SERVEI     = 0.25
    W_RESTRICCIONS = 0.20  # [CLAU] Adaptation-Guided Retrieval
    W_TEMPORADA  = 0.10
    W_FORMALITAT = 0.10
    W_COMENSALS  = 0.03
    W_PREU       = 0.02

    def __init__(self, path_base_casos: str):
        try:
            with open(path_base_casos, "r", encoding="utf-8") as f:
                self.base_casos = json.load(f)
            print(f"✅ [Retriever] Inicialitzat amb {len(self.base_casos)} casos.")
        except FileNotFoundError:
            print(f"❌ [Retriever] Error: No es troba '{path_base_casos}'.")
            self.base_casos = []

    def _norm_txt(self, x: Any) -> str:
        """Normalització bàsica de text per a comparacions."""
        if x is None: return ""
        return str(x).strip().lower()

    # ==========================================
    # 1. MÈTRIQUES DE SIMILITUD LOCAL (sim_i)
    # ==========================================

    def _sim_event(self, a: str, b: str) -> float:
        """Matriu de Semblança per Tipus d'Esdeveniment."""
        a, b = self._norm_txt(a), self._norm_txt(b)
        
        # Coincidència exacta
        if a == b: return 1.0
        
        # Famílies Semàntiques
        familiars = {"casament", "comunio", "comunió", "bateig", "aniversari", "reunio_familiar"}
        corporatius = {"empresa", "congres", "congrés"}
        
        if a in familiars and b in familiars:
            return 0.7
        if a in corporatius and b in corporatius:
            return 0.7
            
        # Casos disjunts o desconeguts
        return 0.2

    def _sim_restriccions(self, req_user: List[str], case_restr: List[str]) -> float:
        """
        Similitud de Jaccard per a conjunts.
        Prioritza casos que estructuralment ja compleixen les restriccions.
        """
        # Normalitzem conjunts
        set_u = set(self._norm_txt(r) for r in req_user if r)
        set_c = set(self._norm_txt(r) for r in case_restr if r)
        
        # Si l'usuari no té restriccions, qualsevol cas serveix (Sim = 1.0)
        # (Tot i que podríem penalitzar lleugerament si el cas és molt restrictiu innecessàriament,
        # simplifiquem assumint que és acceptable).
        if not set_u:
            return 1.0

        # Jaccard = Intersecció / Unió
        interseccio = len(set_u.intersection(set_c))
        unio = len(set_u.union(set_c))
        
        if unio == 0: return 1.0
        
        return float(interseccio / unio)

    def _sim_temporada(self, a: str, b: str) -> float:
        """Distància Cíclica (Primavera <-> Hivern són adjacents)."""
        a, b = self._norm_txt(a), self._norm_txt(b)
        if a == b or "indiferent" in (a, b): return 1.0
        
        ordre = ["primavera", "estiu", "tardor", "hivern"]
        if a in ordre and b in ordre:
            idx_a, idx_b = ordre.index(a), ordre.index(b)
            # Distància mínima al cercle (0, 1 o 2)
            dist = abs(idx_a - idx_b)
            dist_ciclica = min(dist, 4 - dist)
            
            # Mapeig: 0->1.0, 1->0.5, 2->0.0
            return 1.0 - (dist_ciclica * 0.5)
            
        return 0.4 # Valor per defecte si no es reconeix

    def _sim_servei(self, a: str, b: str) -> float:
        a, b = self._norm_txt(a), self._norm_txt(b)
        if a == b or "indiferent" in (a, b): return 1.0
        
        # Semblances parcials
        informals = {"cocktail", "finger_food", "buffet"}
        if a in informals and b in informals: return 0.8
        
        return 0.2

    def _sim_formalitat(self, a: str, b: str) -> float:
        a, b = self._norm_txt(a), self._norm_txt(b)
        if a == b or "indiferent" in (a, b): return 1.0
        return 0.5

    def _sim_comensals(self, n_pet, n_cas) -> float:
        """Diferència relativa inversa."""
        try:
            val_p, val_c = float(n_pet), float(n_cas)
            diff = abs(val_p - val_c)
            # Factor d'esmorteïment (alpha=0.01)
            return 1.0 / (1.0 + 0.01 * diff)
        except: return 0.5

    def _sim_preu_asimetric(self, p_pet, p_cas) -> float:
        """
        Penalització asimètrica:
        - Si el cas és més barat que l'objectiu -> Sim = 1.0 (Bé)
        - Si el cas és més car -> Penalització exponencial (Malament)
        """
        try:
            pp, pc = float(p_pet), float(p_cas)
            if pp <= 0: return 1.0
            
            if pc <= pp:
                return 1.0
            
            # Ratio de sobrecost (ex: 120 vs 100 -> 0.2)
            ratio = (pc - pp) / pp
            # Penalització forta (beta=4)
            return max(0.1, math.exp(-4.0 * ratio))
        except: return 0.8

    # ==========================================
    # 2. AGREGACIÓ GLOBAL
    # ==========================================

    def _score(self, peticio: DescripcioProblema, cas: dict) -> dict:
        # Convertim input a dict si ve com a objecte
        p = peticio.to_dict() if hasattr(peticio, 'to_dict') else peticio
        pr = cas.get("problema", {})

        # --- Càlcul de Similituds Locals ---
        s_ev = self._sim_event(p.get("tipus_esdeveniment"), pr.get("tipus_esdeveniment"))
        s_sv = self._sim_servei(p.get("servei"), pr.get("servei"))
        s_rs = self._sim_restriccions(p.get("restriccions", []), pr.get("restriccions", []))
        s_tm = self._sim_temporada(p.get("temporada"), pr.get("temporada"))
        s_fm = self._sim_formalitat(p.get("formalitat"), pr.get("formalitat"))
        s_cm = self._sim_comensals(p.get("n_comensals"), pr.get("n_comensals"))
        s_pr = self._sim_preu_asimetric(
            p.get("preu_pers_objectiu", p.get("preu_pers")), 
            pr.get("preu_pers_objectiu", pr.get("preu_pers"))
        )

        # --- Agregació Ponderada ---
        score_final = (
            self.W_EVENT * s_ev +
            self.W_SERVEI * s_sv +
            self.W_RESTRICCIONS * s_rs +
            self.W_TEMPORADA * s_tm +
            self.W_FORMALITAT * s_fm +
            self.W_COMENSALS * s_cm +
            self.W_PREU * s_pr
        )

        return {
            "score_final": score_final,
            "detall": {
                "Event": s_ev, 
                "Servei": s_sv,
                "Restriccions": s_rs, 
                "Temp": s_tm, 
                "Formal": s_fm,
                "Preu": s_pr,
                "Pax": s_cm
            }
        }

    def recuperar_casos_similars(self, peticio: DescripcioProblema, k: int = 3):
        """Retorna els top-k casos més similars."""
        resultats = []
        for cas in self.base_casos:
            sc = self._score(peticio, cas)
            resultats.append({
                "cas": cas, 
                "score_final": sc["score_final"], 
                "detall": sc["detall"]
            })
        
        # Ordenar descendentment per score
        resultats.sort(key=lambda x: x["score_final"], reverse=True)
        return resultats[:k]