# retriever.py
import json
import math
from dataclasses import asdict, is_dataclass

from estructura_cas import DescripcioProblema


class Retriever:
    """
    Retriever CBR basat en:
      - similitud categòrica (tipus_esdeveniment, servei, temporada, formalitat)
      - similitud numèrica (n_comensals, preu_pers) amb preu asimètric
    Agregació amb pesos i retorn dels top-k.
    """

    # Pesos (suma = 1.0)
    W_EVENT      = 0.35
    W_SERVEI     = 0.30
    W_TEMPORADA  = 0.20
    W_FORMALITAT = 0.10
    W_COMENSALS  = 0.03
    W_PREU       = 0.02


    def __init__(self, path_base_casos: str):
        try:
            with open(path_base_casos, "r", encoding="utf-8") as f:
                self.base_casos = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: No es troba '{path_base_casos}'.")
            self.base_casos = []
            return

        print(f"✅ Retriever inicialitzat amb {len(self.base_casos)} casos carregats (similaritat estructurada).")

    # -----------------------------
    # Petició -> dict (NOMÉS format nou)
    # -----------------------------
    def _peticio_to_dict(self, peticio: DescripcioProblema) -> dict:
        return asdict(peticio)


    def _norm_txt(self, x: str | None) -> str:
        if x is None:
            return ""
        return str(x).strip().lower()

    # -----------------------------
    # Similaritats locals (0..1)
    # -----------------------------
    def _sim_event(self, a: str, b: str) -> float:
        a = self._norm_txt(a)
        b = self._norm_txt(b)
        if not a or not b:
            return 0.6
        if a == b:
            return 1.0

        familiars = {"casament", "comunio", "comunió", "bateig", "baptisme", "aniversari"}
        institucionals = {"empresa", "congres", "congrés"}

        if a == "altres" or b == "altres":
            return 0.5

        if a in familiars and b in familiars:
            if ("casament" in (a, b)) and (("comunio" in (a, b)) or ("comunió" in (a, b)) or ("bateig" in (a, b)) or ("baptisme" in (a, b))):
                return 0.7
            return 0.6

        if a in institucionals and b in institucionals:
            return 0.7

        return 0.2

    def _sim_servei(self, a: str, b: str) -> float:
        a = self._norm_txt(a) or "indiferent"
        b = self._norm_txt(b) or "indiferent"

        if a == "indiferent" or b == "indiferent":
            return 0.8
        if a == b:
            return 1.0

        cosins = {("cocktail", "finger_food"), ("finger_food", "cocktail")}
        if (a, b) in cosins:
            return 0.8

        if a == "assegut" and b in ("cocktail", "finger_food"):
            return 0.2
        if b == "assegut" and a in ("cocktail", "finger_food"):
            return 0.2

        return 0.3

    def _sim_temporada(self, a: str, b: str) -> float:
        a = self._norm_txt(a)
        b = self._norm_txt(b)
        if not a or not b:
            return 0.7
        if a == "indiferent" or b == "indiferent":
            return 0.8
        if a == b:
            return 1.0

        ordre = ["primavera", "estiu", "tardor", "hivern"]
        if a in ordre and b in ordre:
            d = abs(ordre.index(a) - ordre.index(b))
            return {1: 0.7, 2: 0.4, 3: 0.2}.get(d, 0.4)

        return 0.4

    def _sim_formalitat(self, a: str, b: str) -> float:
        a = self._norm_txt(a) or "indiferent"
        b = self._norm_txt(b) or "indiferent"
        if a == "indiferent" or b == "indiferent":
            return 0.8
        if a == b:
            return 1.0
        if {"formal", "informal"} == {a, b}:
            return 0.2
        return 0.4

    def _sim_comensals(self, n_pet, n_cas) -> float:
        try:
            n_pet = float(n_pet)
            n_cas = float(n_cas)
        except Exception:
            return 0.7

        diff = abs(n_pet - n_cas)
        alpha = 0.01
        return 1.0 / (1.0 + alpha * diff)

    def _sim_preu_asimetric(self, preu_pet, preu_cas) -> float:
        try:
            preu_pet = float(preu_pet)
            preu_cas = float(preu_cas)
        except Exception:
            return 0.8

        if preu_pet <= 0:
            return 0.8

        if preu_cas <= preu_pet:
            return 1.0

        ratio = (preu_cas / preu_pet) - 1.0
        beta = 4.0
        sim = math.exp(-beta * ratio)
        return max(0.1, min(1.0, sim))

    # -----------------------------
    # Similaritat global + retorn
    # -----------------------------
    def _score(self, peticio: DescripcioProblema, cas: dict) -> dict:
        p = self._peticio_to_dict(peticio)
        pr = cas.get("problema", {})

        sim_event = self._sim_event(p["tipus_esdeveniment"], pr.get("tipus_esdeveniment"))
        sim_serv = self._sim_servei(p["servei"], pr.get("servei"))
        sim_temp = self._sim_temporada(p["temporada"], pr.get("temporada"))
        sim_form = self._sim_formalitat(p["formalitat"], pr.get("formalitat"))
        sim_com = self._sim_comensals(p["n_comensals"], pr.get("n_comensals"))
        sim_preu = self._sim_preu_asimetric(p["preu_pers"], pr.get("preu_pers"))

        w_sem = self.W_EVENT + self.W_SERVEI + self.W_TEMPORADA + self.W_FORMALITAT
        sim_sem = (
            self.W_EVENT * sim_event
            + self.W_SERVEI * sim_serv
            + self.W_TEMPORADA * sim_temp
            + self.W_FORMALITAT * sim_form
        ) / (w_sem if w_sem > 0 else 1.0)

        w_num = self.W_COMENSALS + self.W_PREU
        sim_num = (
            self.W_COMENSALS * sim_com
            + self.W_PREU * sim_preu
        ) / (w_num if w_num > 0 else 1.0)

        score_final = (
            self.W_EVENT * sim_event
            + self.W_SERVEI * sim_serv
            + self.W_TEMPORADA * sim_temp
            + self.W_FORMALITAT * sim_form
            + self.W_COMENSALS * sim_com
            + self.W_PREU * sim_preu
        )

        return {
            "score_final": float(score_final),
            "sim_semantica": float(sim_sem),
            "sim_numerica": float(sim_num),
            "breakdown": {
                "sim_event": float(sim_event),
                "sim_servei": float(sim_serv),
                "sim_temporada": float(sim_temp),
                "sim_formalitat": float(sim_form),
                "sim_comensals": float(sim_com),
                "sim_preu": float(sim_preu),
            },
        }

    def recuperar_casos_similars(self, peticio: DescripcioProblema, k: int = 3):
        if not self.base_casos:
            return []

        resultats = []
        for cas in self.base_casos:
            s = self._score(peticio, cas)
            resultats.append(
                {
                    "cas": cas,
                    "score_final": s["score_final"],
                    "detall": {
                        "sim_semantica": round(s["sim_semantica"], 4),
                        "sim_numerica": round(s["sim_numerica"], 4),
                        **{kk: round(vv, 4) for kk, vv in s["breakdown"].items()},
                    },
                }
            )

        resultats.sort(key=lambda x: x["score_final"], reverse=True)
        return resultats[:k]
