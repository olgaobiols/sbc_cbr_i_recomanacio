import json
import os
from typing import List, Set, Dict, Any, Optional, Tuple
import numpy as np

from estructura_cas import DescripcioProblema
from retriever_nuevo import Retriever
from knowledge_base import KnowledgeBase
from gestor_feedback import GestorRevise
from operadors_transformacio_realista import (
    substituir_ingredient, 
    triar_tecniques_per_plat, 
    genera_descripcio_llm, 
    construir_prompt_imatge_menu, 
    genera_imatge_menu_hf
)
from operador_ingredients import (
    FG_WRAPPER,
    ingredients_incompatibles,
    substituir_ingredients_prohibits,
)

from operadors_begudes import recomana_beguda_per_plat, get_ingredient_principal, passa_filtre_dur, score_beguda_per_plat

# =========================
#   INICIALITZACIÓ GLOBAL
# =========================

# 1. Carreguem el cervell (Ontologia + Estils)
kb = KnowledgeBase()

print("\n[KB] Estils latents disponibles:")
print(" - " + "\n - ".join(sorted(kb.estils_latents.keys())))

# Com que els operadors antics esperen llistes de diccionaris, 
# creem referències compatibles per no trencar res:
base_ingredients_list = list(kb.ingredients.values())

# =========================
#   FUNCIONS AUXILIARS CLI
# =========================

def input_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default

def input_choice(prompt, options, default, indiferent_value=None):
    opts = list(options)
    if "indiferent" not in opts:
        opts.append("indiferent")
    opts_txt = "/".join(opts)
    while True:
        txt = input(f"{prompt} ({opts_txt}) [{default}]: ").strip().lower()
        if not txt:
            return default
        if txt == "indiferent":
            return indiferent_value if indiferent_value is not None else "indiferent"
        if txt in opts:
            return txt
        print(f"  Valor no vàlid. Opcions: {opts_txt}")

def input_int_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt: return default
    try:
        return int(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return default

def input_float_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt: return float(default)
    try:
        return float(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return float(default)

def parse_list_input(txt: str) -> Set[str]:
    """Converteix 'gluten, vegan' en {'gluten', 'vegan'} normalitzat."""
    if not txt: return set()
    return {x.strip().lower() for x in txt.split(",") if x.strip()}

EU_ALLERGENS = [
    ("gluten", "Cereals amb gluten"),
    ("crustaceans", "Crustacis"),
    ("egg", "Ous"),
    ("fish", "Peix"),
    ("peanuts", "Cacauets"),
    ("soy", "Soja"),
    ("milk", "Llet"),
    ("nuts", "Fruits secs"),
    ("celery", "Api"),
    ("mustard", "Mostassa"),
    ("sesame", "Sesam"),
    ("sulfites", "Sulfitos"),
    ("lupin", "Lupi"),
    ("molluscs", "Moluscs"),
]

def seleccionar_alergens() -> List[str]:
    print("\nALLERGENS UE (14)")
    for i, (key, label) in enumerate(EU_ALLERGENS, start=1):
        print(f"  {i:>2}. {label} [{key}]")
    txt = input_default("Selecciona allergens (numeros separats per comes, Enter si cap)", "")
    if not txt:
        return []
    seleccionats = []
    for part in txt.split(","):
        token = part.strip().lower()
        if not token:
            continue
        if token.isdigit():
            idx = int(token)
            if 1 <= idx <= len(EU_ALLERGENS):
                seleccionats.append(EU_ALLERGENS[idx - 1][0])
            continue
        for key, _ in EU_ALLERGENS:
            if token == key:
                seleccionats.append(key)
                break
    vistos = set()
    resultat = []
    for key in seleccionats:
        if key not in vistos:
            resultat.append(key)
            vistos.add(key)
    return resultat

PATH_USER_PROFILES = "data/user_profiles.json"

def _load_user_profiles(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        return {}

def _save_user_profiles(path: str, data: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    os.replace(tmp_path, path)

def _get_user_profile(data: Dict[str, Any], user_id: str) -> Dict[str, Any]:
    uid = str(user_id)
    perfil = data.get(uid)
    if perfil is None:
        perfil = {}
        data[uid] = perfil
    return perfil

def _store_user_alergies(data: Dict[str, Any], user_id: str, alergies: List[str]) -> None:
    perfil = _get_user_profile(data, user_id)
    perfil["alergies"] = list(alergies)


def _dedup_preserve_order(items: List[str]) -> List[str]:
    vistos = set()
    resultat = []
    for x in items:
        if x not in vistos:
            resultat.append(x)
            vistos.add(x)
    return resultat

def _vector_mitja(ingredients: List[str]) -> Optional[np.ndarray]:
    vectors = []
    for ing in ingredients:
        vec = FG_WRAPPER.get_vector(ing)
        if vec is not None:
            vectors.append(vec)
    if not vectors:
        return None
    return np.mean(vectors, axis=0)

def _similitud_plat_estil(ingredients: List[str], estils_latents: Dict, nom_estil: str) -> float:
    estil_data = estils_latents.get(nom_estil, {})
    ings_estil = estil_data.get("ingredients", [])
    vec_estil = FG_WRAPPER.compute_concept_vector(ings_estil)
    vec_plat = _vector_mitja(ingredients)
    if vec_estil is None or vec_plat is None:
        return 0.0
    norm_a = np.linalg.norm(vec_plat)
    norm_b = np.linalg.norm(vec_estil)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(vec_plat, vec_estil) / (norm_a * norm_b))

def imprimir_resum_adaptacio(
    etiqueta_plat: str,
    plat: dict,
    ingredients_abans: List[str],
    nom_estil: str,
    intensitat: float,
    kb: Any,
) -> Tuple[List[str], str]:
    """
    Mostra un resum compacte de l'adaptació i retorna un resum per al final.
    """
    ingredients_despres = plat.get("ingredients", []) or []
    ingredients_despres_unics = _dedup_preserve_order(ingredients_despres)

    set_abans = set(ingredients_abans)
    set_despres = set(ingredients_despres_unics)
    afegits = [x for x in ingredients_despres_unics if x not in set_abans]
    trets = [x for x in ingredients_abans if x not in set_despres]

    logs = plat.get("log_transformacio", []) or []
    log_text = " ".join(logs).lower()

    sim_abans = _similitud_plat_estil(ingredients_abans, kb.estils_latents, nom_estil)
    sim_despres = _similitud_plat_estil(ingredients_despres_unics, kb.estils_latents, nom_estil)
    delta = sim_despres - sim_abans

    print(f"\n[{etiqueta_plat}] {plat.get('nom','—')}")
    print(f"- Ingredients: {len(ingredients_abans)} -> {len(ingredients_despres_unics)}")
    print(f"- Abans: {', '.join(ingredients_abans) if ingredients_abans else '—'}")
    print(f"- Despres: {', '.join(ingredients_despres_unics) if ingredients_despres_unics else '—'}")

    duplicat_proposat = None
    if "afegit" in log_text:
        for ing in ingredients_abans:
            if f"afegit {ing}" in log_text:
                duplicat_proposat = ing
                break

    canvi_text = "CAP CANVI"
    motiu = "ja coherent amb l'estil / no millora pairing"
    if trets and afegits:
        canvi_text = "SUBSTITUCIO"
        motiu = "substitució vectorial compatible amb ontologia"
    elif afegits:
        canvi_text = "INSERCIO"
        if "fallback simbòlic" in log_text:
            motiu = "similitud latent insuficient, inserció simbòlica vàlida amb millor pairing"
        else:
            motiu = "inserció vectorial compatible amb ontologia"
    elif duplicat_proposat:
        print(f"- Nota: ingredient proposat ja present ({duplicat_proposat}) -> ignorat per duplicat")
        canvi_text = "CAP CANVI"
        motiu = "inserció descartada (duplicat)"

    if afegits:
        print(f"- Canvi: +{', '.join(afegits)}")
    elif trets:
        print(f"- Canvi: -{', '.join(trets)}")

    print(f"- Pairing (plat): {sim_abans:.2f} -> {sim_despres:.2f} ({delta:+.2f})")
    print(f"- Decisio: {canvi_text}")
    print(f"- Motiu: {motiu}")
    print("")

    if canvi_text == "INSERCIO" and afegits:
        resum = f"+{', '.join(afegits)}"
    elif canvi_text == "SUBSTITUCIO" and afegits and trets:
        resum = f"{trets[0]} -> {afegits[0]}"
    else:
        resum = "cap"

    return afegits, resum


def imprimir_tecnniques_proposades(etiqueta_plat: str, plat: dict, transf: list[dict]):
    nom_plat = plat.get("nom", "—")
    print(f"\n🧪 TÈCNIQUES PROPOSADES — {etiqueta_plat}: {nom_plat}")

    if not transf:
        print("   (Cap tècnica aplicada)")
        return

    for i, t in enumerate(transf, start=1):
        display = t.get("display") or t.get("nom") or "tècnica"
        obj_frase = t.get("objectiu_frase") or "un element del plat"
        desc = (t.get("descripcio") or "").strip()

        tx = t.get("impacte_textura", [])
        sb = t.get("impacte_sabor", [])
        tx_txt = ", ".join(tx) if isinstance(tx, list) and tx else ""
        sb_txt = ", ".join(sb) if isinstance(sb, list) and sb else ""

        print(f"   {i}) {display} → {obj_frase}")
        if desc:
            print(f"      - què és: {desc}")
        if tx_txt:
            print(f"      - textura: {tx_txt}")
        if sb_txt:
            print(f"      - sabor:   {sb_txt}")





def imprimir_casos(candidats, top_k=5):
    """Mostra els resultats del Retriever de forma ordenada."""
    if not candidats:
        print("\n❌ No s'ha trobat cap cas similar.")
        return

    print(f"\n--- {len(candidats)} CASOS TROBATS (Top {min(top_k, len(candidats))}) ---")

    for i, c in enumerate(candidats[:top_k], start=1):
        cas = c["cas"]
        score = c["score_final"]
        detall = c["detall"]
        sol = cas.get("solucio", {})
        pr = cas.get("problema", {})

        etiqueta = "⭐ RECOMANAT" if i == 1 else f"#{i}"
        
        # Dades clau
        event = pr.get("tipus_esdeveniment", "?")
        restr = pr.get("restriccions", [])
        if restr:
            str_restr = f" (Restr: {', '.join(restr)})"
        else:
            str_restr = ""

        # Menú resumit
        plats = sol.get("plats", []) or []

        def _nom_plat(curs: str) -> str:
            curs = str(curs).lower()
            for p in plats:
                if str(p.get("curs", "")).lower() == curs:
                    return p.get("nom", "—")
            return "—"

        p1 = _nom_plat("primer")
        p2 = _nom_plat("segon")
        p3 = _nom_plat("postres")

        print(f"\n{etiqueta} [Similitud: {score:.1%}] - ID: {cas.get('id_cas', '?')}")
        print(f"   Context:  {event} | {pr.get('temporada','?')} | {pr.get('servei','?')}{str_restr}")
        print(f"   Menú:     1. {p1} | 2. {p2} | 3. {p3}")

        # Detall complet dels plats
        def _ordre_plat(p: dict) -> int:
            curs = str(p.get("curs", "")).lower()
            ordre = {"primer": 0, "segon": 1, "postres": 2}
            return ordre.get(curs, 99)

        plats_ordenats = sorted(plats, key=_ordre_plat)
        print("   Plats detallats:")
        for p in plats_ordenats:
            curs = p.get("curs", "?")
            nom = p.get("nom", "—")
            ings = ", ".join(p.get("ingredients", []) or [])
            rols = ", ".join(p.get("rols_principals", []) or [])
            preu = p.get("preu", None)
            print(f"     - {curs}: {nom}")
            print(f"       Ingredients: {ings if ings else '—'}")
            if rols:
                print(f"       Rols principals: {rols}")
            if preu not in ("", None):
                print(f"       Preu: {preu}")

        # Detall de puntuació (útil per debug/demo)
        parts = []
        if "Restriccions" in detall: parts.append(f"Restr={detall['Restriccions']:.2f}")
        if "Event" in detall: parts.append(f"Event={detall['Event']:.2f}")
        print(f"   Detall:   {' | '.join(parts)}")

def imprimir_menu_final(
    plat1, transf_1, info_llm_1, beguda1, score1,
    plat2, transf_2, info_llm_2, beguda2, score2,
    postres, transf_post, info_llm_post, beguda_postres, score_postres
):
    print("\n" + "="*40)
    print("      🍽️  MENÚ ADAPTAT FINAL  🍽️")
    print("="*40)

    for etiqueta, plat, info_llm, beguda, score in [
        ("PRIMER PLAT", plat1, info_llm_1, beguda1, score1),
        ("SEGON PLAT",  plat2, info_llm_2, beguda2, score2),
        ("POSTRES",     postres, info_llm_post, beguda_postres, score_postres),
    ]:
        nom = info_llm.get("nom_nou", plat.get("nom", "—")) if info_llm else plat.get("nom", "—")
        desc = info_llm.get("descripcio_carta", "") if info_llm else "Plat clàssic."
        
        print(f"\n🔹 {etiqueta}: {nom}")
        ings = ", ".join(plat.get("ingredients", []))
        print(f"   Ingredients: {ings}")
        if desc:
            print(f"   Carta: {desc}")
        
        if beguda is None:
            print("   🍷 Beguda recomanada: ❌ No s'ha trobat cap beguda adequada")
        else:
            print(f"   🍷 Beguda recomanada: {beguda.get('nom', '—')} (score {score:.2f})")

        # Si hi ha logs de canvis, els mostrem (Explicabilitat XCBR)
        logs = plat.get("log_transformacio", [])
        if logs:
            print("   🛠️  Adaptacions realitzades:")
            for log in logs:
                print(f"      - {log}")


def debug_kb_match(plat, kb, etiqueta=""):
    print(f"\n[KB CHECK] {etiqueta} — {plat.get('nom','—')}")
    for ing in plat.get("ingredients", []):
        ok = kb.get_info_ingredient(ing) is not None
        print(f"  - {ing}  ->  {'✅' if ok else '❌ NO A KB'}")



# =========================
#   MAIN INTERACTIU
# =========================

def main():
    print("===========================================")
    print("   RECOMANADOR DE MENÚS RicoRico 3.0")
    print("   (CBR Híbrid: Ontologia + FlavorGraph)")
    print("===========================================\n")

    user_id = input_default("Identificador d'usuari (per guardar preferencies)?", "guest")
    user_profiles = _load_user_profiles(PATH_USER_PROFILES)
    stored_alergies = []
    perfil_guardat = user_profiles.get(str(user_id), {})
    if isinstance(perfil_guardat, dict):
        stored_alergies = list(perfil_guardat.get("alergies", []) or [])
    usar_alergies_guardades = False
    if stored_alergies:
        msg = f"Hola {user_id}! Recordem que ets alergica a: {', '.join(stored_alergies)}."
        msg += " Vols mantenir aquesta restriccio? (s/n/c per esborrar)"
        keep = input_default(msg, "s").strip().lower()
        if keep == "s":
            usar_alergies_guardades = True
        elif keep == "c":
            _store_user_alergies(user_profiles, user_id, [])
            _save_user_profiles(PATH_USER_PROFILES, user_profiles)
            stored_alergies = []

    # 1) Inicialitzem el Retriever
    retriever = Retriever("src/base_de_casos.json")

    while True:
        print("\n📝 --- NOVA PETICIÓ ---")

        # 2) Recollida de Dades (Inputs)
        tipus_esdeveniment = input_choice(
            "Tipus d'esdeveniment?",
            ["casament", "aniversari", "empresa", "congres", "comunio"],
            "casament"
        )
        temporada = input_choice(
            "Temporada?",
            ["primavera", "estiu", "tardor", "hivern"],
            "estiu"
        )
        servei = input_choice(
            "Servei?",
            ["assegut", "cocktail", "finger_food"],
            "assegut"
        )
        n_comensals = input_int_default("Nombre de comensals?", 80)
        preu_pers = input_float_default("Pressupost per persona (€)?", 50.0)
        
        # [NOU] Restriccions
        restr_input = input_default("Tens restriccions? (ex: celiac, vegan) [separat per comes]", "")
        restriccions = parse_list_input(restr_input)
        if usar_alergies_guardades:
            alergies = list(stored_alergies)
        else:
            alergies = seleccionar_alergens()

        if alergies != stored_alergies:
            _store_user_alergies(user_profiles, user_id, alergies)
            _save_user_profiles(PATH_USER_PROFILES, user_profiles)
            stored_alergies = list(alergies)
            if alergies:
                usar_alergies_guardades = True
        perfil_usuari = {"alergies": alergies} if alergies else None
        
        alcohol = input_choice("Voldràs begudes amb alcohol?", ["si", "no"],"si")
        
        estil_culinari = ""

        # 3) Construcció del Problema
        problema = DescripcioProblema(
            tipus_esdeveniment=tipus_esdeveniment,
            temporada=temporada,
            n_comensals=n_comensals,
            preu_pers_objectiu=preu_pers, # Compte amb el nom del camp a la dataclass
            servei=servei,
            restriccions=restriccions,
            alcohol=alcohol,
            estil_culinari=estil_culinari
        )

        # 4) Recuperació (Retrieve)
        print(f"\n🔍 Cercant casos similars (amb èmfasi en {', '.join(restriccions) if restriccions else 'estructura'})...")
        resultats = retriever.recuperar_casos_similars(problema, k=5)
        imprimir_casos(resultats, top_k=3)

        if not resultats:
            if input_default("Vols provar de nou? (s/n)", "s").lower() != 's': break
            continue

        # 5) Selecció del Cas
        idx = input_int_default("\nTria el número del cas base (1-3)", 1)
        cas_seleccionat = resultats[idx-1]["cas"]
        sol = cas_seleccionat["solucio"]

        plats = sol.get("plats", []) or []

        def _agafa_plat(curs: str) -> dict:
            curs = str(curs).lower()
            for p in plats:
                if str(p.get("curs", "")).lower() == curs:
                    return p.copy()
            # fallback perquè no peti si falta algun curs
            return {"curs": curs, "nom": "—", "ingredients": []}

        plat1 = _agafa_plat("primer")
        plat2 = _agafa_plat("segon")
        postres = _agafa_plat("postres")

        # 5.5) Substitucio previa d'ingredients prohibits (al.lergens)
        if perfil_usuari:
            print("\nFASE SUBSTITUCIO D'INGREDIENTS PROHIBITS")
            ingredients_usats = set()
            resums_prohibits = []
            plats_pre = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]
            for etiqueta, p in plats_pre:
                ingredients = list(p.get("ingredients", []) or [])
                prohibits = ingredients_incompatibles(ingredients, kb, perfil_usuari)
                if not prohibits:
                    continue
                plat_tmp = {"nom": p.get("nom", ""), "ingredients": ingredients}
                adaptat = substituir_ingredients_prohibits(
                    plat_tmp,
                    prohibits,
                    kb,
                    perfil_usuari=perfil_usuari,
                    ingredients_usats=ingredients_usats,
                )
                if isinstance(adaptat, dict):
                    p["ingredients"] = adaptat.get("ingredients", ingredients)
                    logs = list(p.get("log_transformacio", []) or [])
                    logs_sub = adaptat.get("log_transformacio", []) or []
                    logs.extend(logs_sub)
                    if logs:
                        p["log_transformacio"] = logs
                    if logs_sub:
                        resums_prohibits.append((etiqueta, logs_sub))

            if resums_prohibits:
                print("\nRESUM SUBSTITUCIONS (ALERGENS)")
                for etiqueta, logs_sub in resums_prohibits:
                    print(f"- {etiqueta}:")
                    for log in logs_sub:
                        print(f"  {log}")

        # 6) Adaptació d'ingredients
        print("\nFASE ADAPTACIO INGREDIENTS")
        print("Estils latents disponibles:", ", ".join(sorted(kb.estils_latents.keys())))
        suggeriment = estil_culinari if estil_culinari in kb.estils_latents else ""
        estil_latent = input_default(
            f"Vols aplicar un 'toc' d'estil latent? (ex: picant, thai...) [{suggeriment}]",
            suggeriment
        )

        if estil_latent:
            if estil_latent not in kb.estils_latents:
                print(f"\n⚠️  AVÍS: l'estil latent '{estil_latent}' no existeix a la KB.")
                print("   Estils latents disponibles:", ", ".join(sorted(kb.estils_latents.keys())))

            intensitat = float(input_default("Intensitat de l'adaptació (0.1 - 0.9)?", "0.5"))
            print(f"\nEstil latent: {estil_latent} | Intensitat: {intensitat}")

            plats = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]
            ingredients_estil_usats = set()

            resums = []
            for etiqueta, p in plats:
                etiqueta_short = etiqueta.split()[0]
                ingredients_abans = list(p.get("ingredients", []) or [])

                resultat = substituir_ingredient(
                    p,
                    estil_latent,
                    kb,
                    mode="latent",
                    intensitat=intensitat,
                    ingredients_estil_usats=ingredients_estil_usats,
                    perfil_usuari=perfil_usuari,
                )

                # Si l’operador retorna un plat nou, enganxem resultats al dict original
                if isinstance(resultat, dict) and resultat is not p:
                    p.clear()
                    p.update(resultat)

                _, resum = imprimir_resum_adaptacio(
                    etiqueta_short,
                    p,
                    ingredients_abans,
                    estil_latent,
                    intensitat,
                    kb,
                )
                resums.append((etiqueta_short, resum))

            print(f"\nRESUM CANVIS ({estil_latent})")
            for etiqueta, resum in resums:
                print(f"{etiqueta.capitalize()}: {resum}")
        # debug_kb_match(plat1, kb, "PRIMER")
        # debug_kb_match(plat2, kb, "SEGON")
        # debug_kb_match(postres, kb, "POSTRES")

        # 7) Adaptació 2: Tècniques i Presentació
        print("\n✨ === FASE ADAPTACIÓ: TÈCNIQUES ===")
        kb.llista_estils() # Podries imprimir-los
        estil_tecnic = input_default("Vols aplicar un estil tècnic? (ex: cuina_molecular, rustica) [Enter per saltar]", "")
        
        transf_1, transf_2, transf_post = [], [], []
        info_llm_1, info_llm_2, info_llm_post = None, None, None
        
        if estil_tecnic and estil_tecnic in kb.estils:
            print(f"⚙️  Aplicant tècniques de '{estil_tecnic}'...")
            transf_1 = triar_tecniques_per_plat(plat1, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list, kb=kb)
            transf_2 = triar_tecniques_per_plat(plat2, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list, kb=kb)
            transf_post = triar_tecniques_per_plat(postres, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list, kb=kb)

                # --- NOU: imprimir resum de canvis proposats abans del Gemini ---
            print("\n🧾 RESUM D'ADAPTACIÓ (TÈCNIQUES)")
            imprimir_tecnniques_proposades("PRIMER PLAT", plat1, transf_1)
            imprimir_tecnniques_proposades("SEGON PLAT",  plat2, transf_2)
            imprimir_tecnniques_proposades("POSTRES",     postres, transf_post)

        # Generació de Text (Gemini)
        if input_default("Generar nous noms i descripcions amb Gemini? (s/n)", "n").lower() == 's':
            estil_tecnic_llm = estil_tecnic if estil_tecnic else "classic"
            estil_row = kb.estils.get(estil_tecnic)
            info_llm_1 = genera_descripcio_llm(plat1, transf_1, estil_tecnic_llm, servei, estil_row)
            info_llm_2 = genera_descripcio_llm(plat2, transf_2, estil_tecnic_llm, servei, estil_row)
            info_llm_post = genera_descripcio_llm(postres, transf_post, estil_tecnic_llm, servei, estil_row)


        # 8) Afegir begudes
        print("\n✨ === FASE ADAPTACIÓ: BEGUDES ===")
        beguda1, score1 = recomana_beguda_per_plat(plat1, list(kb.begudes.values()), base_ingredients_list, restriccions, alcohol)
        beguda2, score2 = recomana_beguda_per_plat(plat2, list(kb.begudes.values()), base_ingredients_list, restriccions, alcohol)
        beguda_postres, score_postres = recomana_beguda_per_plat(postres, list(kb.begudes.values()), base_ingredients_list, restriccions, alcohol)

        
        # 9) Resultat Final
        imprimir_menu_final(plat1, transf_1, info_llm_1, beguda1, score1, plat2, transf_2, info_llm_2, beguda2, score2, postres, transf_post, info_llm_post, beguda_postres, score_postres)

        # 9.1) Imatge del menú (opcional)
        if input_default("Generar imatge detallada del menú? (s/n)", "n").lower() == 's':
            plats_info = []
            for plat, info in [
                (plat1, info_llm_1),
                (plat2, info_llm_2),
                (postres, info_llm_post),
            ]:
                plats_info.append({
                    "curs": plat.get("curs", ""),
                    "nom": info.get("nom_nou", plat.get("nom", "—")) if info else plat.get("nom", "—"),
                    "ingredients": plat.get("ingredients", []) or [],
                    "descripcio": info.get("descripcio_carta", "") if info else "",
                    "presentacio": info.get("presentacio", "") if info else "",
                })

            prompt_imatge = construir_prompt_imatge_menu(
                tipus_esdeveniment=tipus_esdeveniment,
                temporada=temporada,
                espai="interior",
                formalitat=problema.formalitat,
                plats_info=plats_info,
            )
            genera_imatge_menu_hf(prompt_imatge, output_path="menu_event.png")
        
        # 10) FASE REVISE (Dual Memory)
        gestor_revise = GestorRevise()
        cas_proposat = {
            "problema": problema,
            "solucio": {"primer": plat1, "segon": plat2, "postres": postres}
        }
        resultat_avaluacio = gestor_revise.avaluar_proposta(cas_proposat, user_id)
        print(f"\nResultat de la revisió: {resultat_avaluacio['tipus_resultat']}")

        # 11) FASE RETAIN (Política de memòria)
        print("\n🧠 --- FASE RETAIN ---")
        print("   [Retain] Criteris: Seguretat -> Utilitat -> Redundància.")
        map_resultat = {
            "CRITICAL_FAILURE": "fracas_critic",
            "SOFT_FAILURE": "fracas_suau",
            "SUCCESS": "exit",
        }
        resultat_retain = map_resultat.get(resultat_avaluacio["tipus_resultat"], "fracas_suau")

        transformation_log = []
        for p in [plat1, plat2, postres]:
            transformation_log.extend(p.get("log_transformacio", []) or [])
        for t in (transf_1 or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")
        for t in (transf_2 or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")
        for t in (transf_post or []):
            transformation_log.append(f"Tècnica: {t.get('nom') or t.get('display') or t}")

        saved = kb.retain_case(
            new_case=cas_proposat,
            evaluation_result=resultat_retain,
            transformation_log=transformation_log,
            user_score=resultat_avaluacio["puntuacio_global"],
            retriever_instance=retriever,
        )
        if saved:
            print("✅ [RETAIN] Decisió final: el cas s'ha guardat a la memòria.")
        else:
            print("❌ [RETAIN] Decisió final: el cas NO s'ha guardat a la memòria.")

        if input_default("\nSortir? (s/n)", "n").lower() == 's':
            print("Bon profit! 👋")
            break

if __name__ == "__main__":
    main()
