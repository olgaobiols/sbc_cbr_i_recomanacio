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
    if not hasattr(kb_instance, "base_casos") or kb_instance.base_casos is None:
        base_path = os.path.join(kb_instance.data_dir, "base_de_casos.json")
        alt_path = os.path.join("src", "base_de_casos.json")
        if os.path.exists(base_path):
            with open(base_path, "r", encoding="utf-8") as f:
                kb_instance.base_casos = json.load(f)
        elif os.path.exists(alt_path):
            with open(alt_path, "r", encoding="utf-8") as f:
                kb_instance.base_casos = json.load(f)
        else:
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
    gamma = 0.1
    if d_min < gamma:
        print(f"❌ [RETAIN] Cas descartat per Redundància. Distància {d_min:.2f} < Gamma.")
        return False

    # 5) Persistència segons utilitat
    utility_threshold = 0.6
    if utilitat > utility_threshold:
        final_case = {
            "problema": new_case["problema"],
            "solucio": new_case["solucio"],
            "metadades": {
                "score_usuari": user_score,
                "traça_transformacions": transformation_log,
                "cost_adaptacio": k_adapt,
                "utilitat_calculada": utilitat,
            },
        }
        kb_instance.base_casos.append(final_case)
        out_path = os.path.join(kb_instance.data_dir, "base_de_casos.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(kb_instance.base_casos, f, indent=4, ensure_ascii=False)
        print(f"✅ [RETAIN] Cas APRÈS i guardat! (Utilitat {utilitat:.2f} > Llindar).")
        return True

    print(f"❌ [RETAIN] Cas descartat per Baixa Utilitat ({utilitat:.2f}).")
    return False
