import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from estructura_cas import DescripcioProblema

class Retriever:
    def __init__(self, path_base_casos):

        # ----------------------------
        # 1. Carregar base
        # ----------------------------
        try:
            with open(path_base_casos, 'r', encoding="utf-8") as f:
                self.base_casos = json.load(f)
        except FileNotFoundError:
            print(f"❌ No es troba: {path_base_casos}")
            self.base_casos = []
            return

        print("🧠 Carregant MiniLM per a embeddings enriquits...")
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

        # ----------------------------
        # 2. Crear signatura rica per cas
        # ----------------------------
        self.corpus_text = [self._construir_signatura_cas(c) for c in self.base_casos]
        self.corpus_embeddings = self.model.encode(self.corpus_text, convert_to_tensor=True)

        print(f"✅ Retriever indexat amb {len(self.base_casos)} casos.")

    # ---------------------------------------------------------
    # Construcció de SIGNATURA: la clau de la millora
    # ---------------------------------------------------------
    def _construir_signatura_cas(self, cas):
        p = cas["problema"]
        s = cas["solucio"]

        # 1. Ingredients concatenats
        ing = (
            s["primer_plat"]["ingredients"] +
            s["segon_plat"]["ingredients"] +
            s["postres"]["ingredients"]
        )
        ing_text = " ".join(ing)

        # 2. Text ric i estructurat
        text = (
            f"Estil: {p['estil_culinari']}. "
            f"Esdeveniment: {p['tipus_esdeveniment']}. "
            f"Temporada: {p['temporada']}. "
            f"Formalitat: {p['formalitat']}. "
            f"Restriccions: {' '.join(p['restriccions']) if p['restriccions'] else 'cap'}. "
            f"Ingredients: {ing_text}."
        )
        return text

    # ---------------------------------------------------------
    # Similitud numèrica millorada
    # ---------------------------------------------------------
    def _similitud_numerica(self, peticio: DescripcioProblema, cas_dict):

        # Pressupost → hard constraint suavitzat
        preu_cas = cas_dict["solucio"]["preu_total"]
        if preu_cas <= peticio.pressupost_max:
            s_preu = 1.0
        else:
            excedent = preu_cas - peticio.pressupost_max
            s_preu = max(0.0, 1 - excedent / 50)

        # Comensals (gaussiana suau)
        diff = abs(peticio.n_comensals - cas_dict["problema"]["n_comensals"])
        s_com = np.exp(-diff / 80)

        return 0.6 * s_preu + 0.4 * s_com

    # ---------------------------------------------------------
    # Similitud semàntica per atributs (pesada)
    # ---------------------------------------------------------
    def _similitud_textual_per_atribut(self, valor_query, valor_cas):
        t = f"{valor_query}"
        c = f"{valor_cas}"
        emb_t = self.model.encode(t, convert_to_tensor=True)
        emb_c = self.model.encode(c, convert_to_tensor=True)
        return float(util.cos_sim(emb_t, emb_c)[0][0])

    # ---------------------------------------------------------
    # Recuperació principal
    # ---------------------------------------------------------
    def recuperar_casos_similars(self, peticio: DescripcioProblema, k=3):

        if not self.base_casos:
            return []

        # -----------------------------------------------------
        # 1. Embedding de signatura rica de la petició
        # -----------------------------------------------------
        query_text = (
            f"Estil: {peticio.estil_culinari}. "
            f"Esdeveniment: {peticio.tipus_esdeveniment}. "
            f"Temporada: {peticio.temporada}. "
            f"Formalitat: {peticio.formalitat}. "
            f"Restriccions: {' '.join(peticio.restriccions) if peticio.restriccions else 'cap'}."
        )
        query_emb = self.model.encode(query_text, convert_to_tensor=True)

        # -----------------------------------------------------
        # 2. Semàntica global
        # -----------------------------------------------------
        cos_scores = util.cos_sim(query_emb, self.corpus_embeddings)[0]

        resultats = []

        # -----------------------------------------------------
        # 3. Combinació multi-similarity (PESADA)
        # -----------------------------------------------------
        for idx, sem_score in enumerate(cos_scores):
            cas = self.base_casos[idx]
            p = cas["problema"]

            sim_sem_global = float(sem_score)

            # Atributs textuals amb pesos separats
            sim_estil = self._similitud_textual_per_atribut(peticio.estil_culinari, p["estil_culinari"])
            sim_event = self._similitud_textual_per_atribut(peticio.tipus_esdeveniment, p["tipus_esdeveniment"])
            sim_temp = self._similitud_textual_per_atribut(peticio.temporada, p["temporada"])
            sim_form = self._similitud_textual_per_atribut(peticio.formalitat, p["formalitat"])

            sim_num = self._similitud_numerica(peticio, cas)

            # PESOS recomanats
            score_final = (
                0.30 * sim_sem_global +
                0.20 * sim_estil +
                0.10 * sim_event +
                0.05 * sim_temp +
                0.05 * sim_form +
                0.30 * sim_num
            )

            resultats.append({
                "cas": cas,
                "score_final": score_final,
                "detall": {
                    "sim_semantica_global": round(sim_sem_global, 3),
                    "sim_estil": round(sim_estil, 3),
                    "sim_event": round(sim_event, 3),
                    "sim_temp": round(sim_temp, 3),
                    "sim_form": round(sim_form, 3),
                    "sim_num": round(sim_num, 3)
                }
            })

        resultats.sort(key=lambda x: x["score_final"], reverse=True)
        return resultats[:k]
