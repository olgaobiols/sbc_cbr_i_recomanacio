import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from estructura_cas import DescripcioProblema

class Retriever:
    def __init__(self, path_base_casos):
        # 1. Carregar la base de casos
        try:
            with open(path_base_casos, 'r', encoding='utf-8') as f:
                self.base_casos = json.load(f)
        except FileNotFoundError:
            print(f"❌ Error: No es troba '{path_base_casos}'. Has executat el generador?")
            self.base_casos = []
            return
        
        # 2. Carregar el model d'Embeddings (Petit i ràpid)
        print("🧠 Carregant model de llenguatge (MiniLM) per a la similitud semàntica...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # 3. Pre-calcular els embeddings dels casos existents (Indexació)
        # Concatenem estil i tipus d'event per crear la "signatura" del cas
        self.corpus_text = [
            f"{c['problema']['estil_culinari']} {c['problema']['tipus_esdeveniment']} {c['problema']['formalitat']}" 
            for c in self.base_casos
        ]
        # Això converteix el text en vectors numèrics
        self.corpus_embeddings = self.model.encode(self.corpus_text, convert_to_tensor=True)
        print(f"✅ Retriever inicialitzat amb {len(self.base_casos)} casos indexats.")

    def _similitud_numerica(self, peticio: DescripcioProblema, cas_dict):
        """Calcula una puntuació (0-1) basada en restriccions dures."""
        
        # 1. Pressupost (Factor crític)
        preu_cas = cas_dict['solucio']['preu_total']
        if preu_cas > peticio.pressupost_max:
            # Si es passa de pressupost, penalitzem molt (però no descartem del tot per si l'estil és perfecte)
            sim_preu = 0.2
        else:
            sim_preu = 1.0
            
        # 2. Comensals (Factor logístic)
        # Utilitzem una funció gaussiana simple: com més lluny, menys similitud
        diff = abs(peticio.n_comensals - cas_dict['problema']['n_comensals'])
        sim_comensals = 1 / (1 + 0.01 * diff) 

        return (sim_preu * 0.6) + (sim_comensals * 0.4)

    def recuperar_casos_similars(self, peticio: DescripcioProblema, k=3):
        """
        Retorna els k casos més similars combinant Semàntica + Numèrica.
        """
        if not self.base_casos:
            return []

        # 1. Crear embedding de la PETICIÓ de l'usuari
        query_text = f"{peticio.estil_culinari} {peticio.tipus_esdeveniment} {peticio.formalitat}"
        query_embedding = self.model.encode(query_text, convert_to_tensor=True)

        # 2. Calcular Similitud Cosinus (Semàntica)
        # Això ens diu com d'aprop estan els conceptes (ex: "Japonès" vs "Oriental")
        cos_scores = util.cos_sim(query_embedding, self.corpus_embeddings)[0]

        resultats = []

        # 3. Combinar amb Similitud Numèrica i crear llista de candidats
        for idx, score_sem_tensor in enumerate(cos_scores):
            cas = self.base_casos[idx]
            score_sem = float(score_sem_tensor)
            
            score_num = self._similitud_numerica(peticio, cas)
            
            # PONDERACIÓ FINAL: 
            # 70% Estil (Semàntica) + 30% Restriccions (Numèrica)
            # Això prioritza que el menú "inspiri" l'estil, encara que haguem d'adaptar preu després.
            score_final = (score_sem * 0.7) + (score_num * 0.3)
            
            resultats.append({
                'cas': cas,
                'score_final': score_final,
                'detall': {
                    'sim_semantica': round(score_sem, 4),
                    'sim_numerica': round(score_num, 4)
                }
            })

        # 4. Ordenar per millor puntuació i retornar els top K
        resultats.sort(key=lambda x: x['score_final'], reverse=True)
        return resultats[:k]