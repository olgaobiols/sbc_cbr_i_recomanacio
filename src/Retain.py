import json
import math
import os
from typing import Any, Dict, List


def retain_case(
    kb_instance: Any,
    new_case: Dict,
    evaluation_result: str,
    transformation_log: List[str],
    user_score: int,
    retriever_instance: Any,
) -> bool:
    """
    Retain policy segons teoria: Seguretat -> Cost d'adaptació -> Utilitat -> Redundància.
    Retorna True si el cas s'ha guardat, False si es descarta.
    """
    # 1) Filtre de seguretat
    if evaluation_result == "fracas_critic":
        print("❌ [RETAIN] Cas descartat per Fracàs Crític (violació de restriccions).")
        return False

    # 2) Cost d'adaptació K_adapt
    k_adapt = 0
    for entry in transformation_log or []:
        txt = (entry or "").lower()
        txt_norm = txt.replace("ó", "o").replace("è", "e").replace("à", "a")
        if "estil" in txt_norm or "latent" in txt_norm or "tecnica" in txt_norm:
            k_adapt += 3
        elif "substitucio" in txt_norm:
            k_adapt += 1
        else:
            k_adapt += 1


    # 3) Utilitat U
    q_user = max(0.0, min(1.0, float(user_score) / 5.0))
    alpha = 0.5
    utilitat = q_user * (1 + alpha * math.log(1 + k_adapt))
    print(f"ℹ️ [RETAIN] Cost d'adaptació K_adapt={k_adapt} | Utilitat U={utilitat:.2f}")


    # 4) Redundància (d_min)
    path_fixe = os.path.join("base_de_casos.json")
    if not hasattr(kb_instance, "base_casos") or kb_instance.base_casos is None:
        try:
            with open(path_fixe, "r", encoding="utf-8") as f:
                content = f.read().strip()
                kb_instance.base_casos = json.loads(content) if content else []
        except:
            kb_instance.base_casos = []

    sim_max = 0.0
    for existing_case in kb_instance.base_casos:
        try:
            sc = retriever_instance._score(new_case["problema"], existing_case)
            sim = sc.get("score_final", 0.0)
            if sim > sim_max:
                sim_max = sim
        except Exception:
            continue

    d_min = 1.0 - sim_max
    gamma = 0.01
    if d_min < gamma:
        print(f"❌ [RETAIN] Cas descartat per Redundància. Distància {d_min:.2f} < Gamma.")
        return False

    # 5) Persistència segons utilitat
    utility_threshold = 0.6
    
    if utilitat > utility_threshold:
        prob_obj = new_case["problema"]
        solu_obj = new_case.get("solucio", {})

        # --- PROCESSAMENT DELS PLATS (Adaptat al teu main) ---
        plats_per_json = []
        
        # El teu main envia un dict amb claus: "primer", "segon", "postres"
        # Iterem sobre aquestes claus per extreure els objectes plat
        for clau_curs in ["primer", "segon", "postres"]:
            p = solu_obj.get(clau_curs)
            if p:
                # Si p és un objecte o diccionari, extraiem la info
                # Fem servir una funció helper per seguretat
                def get_v(obj, attr, default=""):
                    if isinstance(obj, dict):
                        return obj.get(attr, default)
                    return getattr(obj, attr, default)

                plats_per_json.append({
                    "curs": clau_curs,
                    "nom": get_v(p, "nom", "Sense nom"),
                    "ingredients": list(get_v(p, "ingredients", [])),
                    "preu": float(get_v(p, "preu", 0.0))
                })

        # --- ESTRUCTURA DEL PROBLEMA ---
        # Convertim a llista per evitar l'error del JSON amb els 'set'
        restr_raw = getattr(prob_obj, "restriccions", [])
        restr_list = list(restr_raw) if isinstance(restr_raw, (set, list)) else []

        problema_dict = {
            "tipus_esdeveniment": getattr(prob_obj, "tipus_esdeveniment", ""),
            "estil_culinari": getattr(prob_obj, "estil_culinari", ""),
            "temporada": getattr(prob_obj, "temporada", ""),
            "n_comensals": getattr(prob_obj, "n_comensals", 0),
            "preu_pers_objectiu": getattr(prob_obj, "preu_pers_objectiu", 0.0),
            "servei": getattr(prob_obj, "servei", ""),
            "formalitat": getattr(prob_obj, "formalitat", "indiferent"),
            "restriccions": restr_list
        }

        # --- ESTRUCTURA FINAL DEL CAS ---
        final_case = {
            "id_cas": len(kb_instance.base_casos) + 1,
            "problema": problema_dict,
            "solucio": {
                "plats": plats_per_json,
                "begudes": [], # Pots extreure-les si les afegeixes al cas_proposat del main
                "preu_total_real": sum(p["preu"] for p in plats_per_json)
            }
        }

        # --- GUARDAT AL FITXER ---
        kb_instance.base_casos.append(final_case)
        try:
            with open(path_fixe, "w", encoding="utf-8") as f:
                json.dump(kb_instance.base_casos, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            print(f"✅ [RETAIN] Cas {final_case['id_cas']} guardat amb èxit!")
            return True
        except Exception as e:
            print(f"❌ [RETAIN] Error en l'escriptura JSON: {e}")
            return False

    print(f"❌ [RETAIN] Cas descartat per Baixa Utilitat ({utilitat:.2f}).")
    return False