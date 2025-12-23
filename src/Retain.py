import json
import math
import os
import unicodedata
from typing import Any, Dict, List, Optional

"""
GESTOR DE LA FASE RETAIN (Aprenentatge)
--------------------------------------
Implementa la política de retenció de casos a la base de coneixement (BC).
Segueix el criteri de decisió: Seguretat -> Cost d'Adaptació -> Utilitat -> Redundància.
Calcula la utilitat $U$ basant-se en la satisfacció de l'usuari i l'esforç d'adaptació.
"""

# Paràmetres de la política de retenció
LLINDAR_UTILITAT = 0.6  # Tau_u: Mínim per considerar que el cas val la pena guardar-lo
GAMMA = 0.01            # Radi d'exclusió: Evita guardar casos gairebé idèntics
PATH_BC = os.path.join("data", "base_de_casos.json")

def _normalize_text(text: str) -> str:
    """Elimina accents i normalitza a minúscules per a comparacions robustes."""
    if not text: return ""
    return "".join(
        c for c in unicodedata.normalize('NFD', str(text))
        if unicodedata.category(c) != 'Mn'
    ).lower()

def _calcular_cost_adaptacio(transformation_log: List[str]) -> int:
    """
    Heurística per quantificar l'esforç d'adaptació (K_adapt).
    Assigna pesos segons la complexitat de la transformació realitzada.
    """
    complex_kw = {"estil", "latent", "tecnica", "estructural", "cultural", "toc magic"}
    simple_kw = {"substitucio", "maridatge", "fallback"}
    excluded_kw = {"imatge", "descripcio", "llm", "presentacio"}

    k_total = 0
    for entry in (transformation_log or []):
        norm = _normalize_text(entry)
        if any(kw in norm for kw in excluded_kw): continue
        
        if any(kw in norm for kw in complex_kw):
            k_total += 3  # Transformacions d'alt valor cognitiu
        elif any(kw in norm for kw in simple_kw):
            k_total += 1  # Substitucions directes
        else:
            k_total += 1
    return k_total

def retain_case(
    kb_instance: Any,
    new_case: Dict,
    evaluation_result: str,
    transformation_log: List[str],
    user_score: int,
    retriever_instance: Any,
) -> bool:
    """
    Avalua si un cas nou ha de ser persistit a la Base de Casos.
    Retorna True si el cas s'ha après, False si s'ha descartat.
    """
    
    # 1. FILTRE DE SEGURETAT: No s'aprenen errors crítics (al·lèrgies, etc.)
    if evaluation_result == "CRITICAL_FAILURE":
        print("[RETAIN] Descartat: El cas conté errors de seguretat o restriccions violades.")
        return False

    # 2. CÀLCUL DEL COST D'ADAPTACIÓ (K)
    k_adapt = _calcular_cost_adaptacio(transformation_log)

    # 3. CÀLCUL D'UTILITAT (U)
    # Fórmula: $U = Q_{user} \times (1 + \alpha \cdot \ln(1 + K_{adapt}))$
    # On Q_user és la nota normalitzada (0-1) i alpha és el pes de la novetat estructural.
    q_user = max(0.0, min(1.0, float(user_score) / 5.0))
    alpha = 0.5
    utilitat = q_user * (1 + alpha * math.log(1 + k_adapt))

    # 4. FILTRE DE REDUNDÀNCIA (Diversitat)
    # Cerquem la similitud amb el cas més proper de la BC
    if not hasattr(kb_instance, "base_casos") or not kb_instance.base_casos:
        kb_instance.base_casos = _carregar_bc_existent()

    sim_max = 0.0
    for existing in kb_instance.base_casos:
        try:
            res = retriever_instance._score(new_case["problema"], existing)
            sim = res.get("score_final", 0.0)
            if sim > sim_max: sim_max = sim
        except: continue

    d_min = 1.0 - sim_max
    if d_min < GAMMA:
        print(f"[RETAIN] Descartat: Cas redundant (d_min {d_min:.4f} < {GAMMA}).")
        return False

    # 5. DECISIÓ FINAL I PERSISTÈNCIA
    if utilitat > LLINDAR_UTILITAT:
        return _persistir_cas(kb_instance, new_case, k_adapt, utilitat, user_score, transformation_log)

    print(f"[RETAIN] Descartat: Baixa utilitat (U={utilitat:.2f}). Solució massa trivial.")
    return False

# --- AUXILIARS DE PERSISTÈNCIA ---

def _carregar_bc_existent() -> List[Dict]:
    """Carrega la base de casos de disc si no està en memòria."""
    if os.path.exists(PATH_BC):
        with open(PATH_BC, "r", encoding="utf-8") as f:
            try: return json.load(f)
            except: return []
    return []

def _persistir_cas(kb, case, k_adapt, utilitat, score, logs) -> bool:
    """Serialitza i guarda el cas amb l'estructura canònica."""
    prob = case["problema"]
    solu = case.get("solucio", {})
    
    # Helper per extreure atributs de dataclasses o dicts indistintament
    def get_val(obj, attr, default=None):
        return getattr(obj, attr, default) if hasattr(obj, attr) else obj.get(attr, default)

    # Reconstrucció de plats amb metadades
    plats_json = []
    for curs in ["primer_plat", "segon_plat", "postres"]:
        p = get_val(solu, curs)
        if p:
            plats_json.append({
                "curs": curs.replace("_plat", ""),
                "nom": get_val(p, "nom"),
                "ingredients": list(get_val(p, "ingredients", [])),
                "rols": list(get_val(p, "rols_ingredients", [])),
                "tags": list(get_val(p, "estil_tags", [])),
                "preu": float(get_val(p, "preu", 0.0))
            })

    # Estructura final del cas per a la BC
    final_entry = {
        "id_cas": len(kb.base_casos) + 1,
        "problema": {
            "tipus_esdeveniment": get_val(prob, "tipus_esdeveniment"),
            "estil_culinari": get_val(prob, "estil_culinari"),
            "temporada": get_val(prob, "temporada"),
            "n_comensals": get_val(prob, "n_comensals"),
            "preu_pers_objectiu": get_val(prob, "preu_pers_objectiu"),
            "restriccions": list(get_val(prob, "restriccions", []))
        },
        "solucio": {
            "plats": plats_json,
            "begudes": [get_val(b, "nom") for b in get_val(solu, "begudes", [])],
            "preu_total_real": sum(p["preu"] for p in plats_json),
            "logs_transformacio": logs
        },
        "avaluacio": {
            "puntuacio_global": score,
            "cost_adaptacio": k_adapt,
            "utilitat": round(utilitat, 2),
            "data_aprenentatge": str(os.path.getmtime(PATH_BC)) if os.path.exists(PATH_BC) else "avui"
        }
    }

    kb.base_casos.append(final_entry)
    with open(PATH_BC, "w", encoding="utf-8") as f:
        json.dump(kb.base_casos, f, indent=4, ensure_ascii=False)
    
    print(f"[RETAIN] ÈXIT: Cas après amb utilitat {utilitat:.2f} i cost d'adaptació {k_adapt}.")
    return True