import json
import math
import os
from typing import Any, Dict, List


LLINDAR_UTILITAT = 0.6
GAMMA = 0.01

def _normalize_cost_text(value: str) -> str:
    if not value:
        return ""
    txt = str(value).lower()
    return (
        txt.replace("ó", "o")
        .replace("è", "e")
        .replace("à", "a")
        .replace("í", "i")
        .replace("ï", "i")
        .replace("ú", "u")
    )

def _calcular_cost_adaptacio(transformation_log: List[str]) -> int:
    complex_kw = {"estil", "latent", "tecnica", "estructural", "cultural", "toc magic"}
    simple_kw = {"substitucio", "maridatge", "fallback"}
    excluded_kw = {"imatge", "descripcio", "llm", "presentacio"}

    k_adapt = 0
    for entry in transformation_log or []:
        txt_norm = _normalize_cost_text(entry)
        if not txt_norm:
            continue
        if any(kw in txt_norm for kw in excluded_kw):
            continue
        if any(kw in txt_norm for kw in complex_kw):
            k_adapt += 3
        elif any(kw in txt_norm for kw in simple_kw):
            k_adapt += 1
        else:
            k_adapt += 1
    return k_adapt

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
        print("[DECISIÓ: DESCARTAT PER SEGURETAT]")
        print("El cas ha estat rebutjat degut a un Fracàs Crític (violació de restriccions dures o al·lèrgies).")
        print("Segons la política de retenció, el sistema no pot interioritzar coneixement insegur.\n")
        return False

    # 2) Cost d'adaptació K_adapt
    k_adapt = _calcular_cost_adaptacio(transformation_log)
    print(f"Cost d'adaptació calculat: K_adapt={k_adapt}\n")


    # 3) Utilitat U
    q_user = max(0.0, min(1.0, float(user_score) / 5.0))
    alpha = 0.5
    utilitat = q_user * (1 + alpha * math.log(1 + k_adapt))


    # 4) Redundància (d_min)
    path_fixe = os.path.join("data", "base_de_casos.json")
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
    if d_min < GAMMA:
        print("[DECISIÓ: DESCARTAT PER REDUNDÀNCIA]")
        print("    • El cas no aporta prou novetat a la Base de Casos.")
        print(f"    • Distància al veí més proper (d_min={d_min:.2f}) < radi d'exclusió (gamma={GAMMA}).\n")
        return False

    # 5) Persistència segons utilitat i estructura final detallada
    utility_threshold = LLINDAR_UTILITAT
    
    if utilitat > utility_threshold:
        prob_obj = new_case["problema"]
        solu_obj = new_case.get("solucio", {})

        # --- 1. PROCESSAMENT DELS PLATS (amb rols i tags) ---
        plats_per_json = []
        for clau_curs in ["primer", "segon", "postres"]:
            p = solu_obj.get(clau_curs)
            if p:
                def get_v(obj, attr, default):
                    if isinstance(obj, dict):
                        return obj.get(attr, default)
                    return getattr(obj, attr, default)

                plats_per_json.append({
                    "curs": clau_curs,
                    "nom": get_v(p, "nom", "Sense nom"),
                    "ingredients": list(get_v(p, "ingredients", [])),
                    "rols_principals": list(get_v(p, "rols_principals", [])),
                    "estil_tags": list(get_v(p, "estil_tags", [])),
                    "preu": float(get_v(p, "preu", 0.0))
                })

        # --- 2. ESTRUCTURA DEL PROBLEMA ---
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

        # --- 3. ESTRUCTURA FINAL DEL CAS (Problema + Solució + Avaluació) ---
        final_case = {
            "id_cas": len(kb_instance.base_casos) + 1,
            "problema": problema_dict,
            "solucio": {
                "plats": plats_per_json,
                "begudes": list(solu_obj.get("begudes", [])),
                "preu_total_real": sum(p["preu"] for p in plats_per_json),
                "logs_transformacio": transformation_log if transformation_log else []
            },
            "avaluacio": {
                "puntuacio_global": user_score,
                "validat": (evaluation_result == "exit"),
                "cost_adaptacio": k_adapt,
                "ingredients_rebutjats": [], # Es podria omplir si el log ho indica
                "utilitat": round(utilitat, 2)
            }
        }

        # --- 4. GUARDAT AL FITXER JSON ---
        kb_instance.base_casos.append(final_case)
        try:
            with open(path_fixe, "w", encoding="utf-8") as f:
                json.dump(kb_instance.base_casos, f, indent=4, ensure_ascii=False)
                f.flush()
                os.fsync(f.fileno())
            print("[DECISIÓ: APRÈS I RETINGUT]")
            print("El cas s'ha incorporat exitosament a la memòria a llarg termini pels següents motius:")
            print("    • Seguretat validada.")
            print(f"    • Alta Utilitat calculada (U={utilitat:.2f}): Esforç d'adaptació (K={k_adapt}) vs Satisfacció.")
            print(f"    • Novetat confirmada (d_min >= {GAMMA}).\n")
            return True
        except Exception as e:
            print(f"Error al guardar: {e}")
            return False

    print("[DECISIÓ: DESCARTAT PER BAIXA UTILITAT]")
    print(f" • Utilitat calculada (U={utilitat:.2f}) inferior al llindar ({LLINDAR_UTILITAT}).")
    print(f" • Motiu: El Cost d'Adaptació ha estat baix (K_adapt={k_adapt}).")
    print(" •  Etiquetat com a solució trivial. El cost de manteniment supera el cost de regeneració futura.\n")
    return False
