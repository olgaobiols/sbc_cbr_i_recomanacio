import os
from typing import List, Set

from estructura_cas import DescripcioProblema
from retriever_nuevo import Retriever
from knowledge_base import KnowledgeBase
from operadors_transformacio_realista import (
    substituir_ingredient, 
    triar_tecniques_per_plat, 
    genera_descripcio_llm, 
    construir_prompt_imatge_menu, 
    genera_imatge_menu_hf
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


#ELIMINAR EN UN FUTUR 
def debug_ingredients_abans_despres(etiqueta_plat: str, plat: dict, ingredients_abans: list[str]):
    """
    Mostra diferències d'ingredients i també el log_transformacio si existeix.
    """
    ingredients_despres = plat.get("ingredients", []) or []

    set_abans = set(ingredients_abans)
    set_despres = set(ingredients_despres)

    afegits = sorted(list(set_despres - set_abans))
    trets = sorted(list(set_abans - set_despres))

    print(f"\n🧩 DEBUG INGREDIENTS — {etiqueta_plat}: {plat.get('nom','—')}")
    print(f"   Abans ({len(ingredients_abans)}):  {', '.join(ingredients_abans) if ingredients_abans else '—'}")
    print(f"   Després ({len(ingredients_despres)}): {', '.join(ingredients_despres) if ingredients_despres else '—'}")

    if not afegits and not trets:
        print("   ✅ Sense canvis d'ingredients (o canvis només interns).")
    else:
        if trets:
            print(f"   ➖ Tret:   {', '.join(trets)}")
        if afegits:
            print(f"   ➕ Afegit: {', '.join(afegits)}")

    logs = plat.get("log_transformacio", []) or []
    if logs:
        print("   🧾 log_transformacio:")
        for l in logs:
            print(f"      - {l}")


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
    plat1, transf_1, info_llm_1, beguda1,
    plat2, transf_2, info_llm_2, beguda2,
    postres, transf_post, info_llm_post, beguda_postres
):
    print("\n" + "="*40)
    print("      🍽️  MENÚ ADAPTAT FINAL  🍽️")
    print("="*40)

    for etiqueta, plat, info_llm, beguda in [
        ("PRIMER PLAT", plat1, info_llm_1, beguda1),
        ("SEGON PLAT",  plat2, info_llm_2, beguda2),
        ("POSTRES",     postres, info_llm_post, beguda_postres),
    ]:
        nom = info_llm.get("nom_nou", plat.get("nom", "—")) if info_llm else plat.get("nom", "—")
        desc = info_llm.get("descripcio_carta", "") if info_llm else "Plat clàssic."
        
        print(f"\n🔹 {etiqueta}: {nom}")
        ings = ", ".join(plat.get("ingredients", []))
        print(f"   Ingredients: {ings}")
        if desc:
            print(f"   Carta: {desc}")
        
        if beguda:
            print(f"   🍷 Beguda recomanada: {beguda.get('nom', '—')}")

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

    # 1) Inicialitzem el Retriever
    retriever = Retriever("src/base_de_casos.json")

    while True:
        print("\n📝 --- NOVA PETICIÓ ---")

        # 2) Recollida de Dades (Inputs)
        tipus_esdeveniment = input_default("Tipus d'esdeveniment? (casament/aniversari/empresa...)", "casament")
        temporada = input_default("Temporada? (primavera/estiu/tardor/hivern)", "estiu")
        servei = input_default("Servei? (assegut/cocktail)", "assegut")
        n_comensals = input_int_default("Nombre de comensals?", 80)
        preu_pers = input_float_default("Pressupost per persona (€)?", 50.0)
        
        # [NOU] Restriccions
        restr_input = input_default("Tens restriccions? (ex: celiac, vegan) [separat per comes]", "")
        restriccions = parse_list_input(restr_input)
        
        # [NOU] Estil (Opcional)
        estil_culinari = input_default("Estil culinari preferit? (ex: japonès, mediterrani) [opcional]", "")

        # 3) Construcció del Problema
        problema = DescripcioProblema(
            tipus_esdeveniment=tipus_esdeveniment,
            temporada=temporada,
            n_comensals=n_comensals,
            preu_pers_objectiu=preu_pers, # Compte amb el nom del camp a la dataclass
            servei=servei,
            restriccions=restriccions,
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

        # 6) Adaptació d'ingredients
        print("\n🎨 === FASE ADAPTACIÓ: INGREDIENTS ===")
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
            print(f"🔄 Adaptant ingredients cap a '{estil_latent}'...")

            plats = [
                ("PRIMER PLAT", plat1),
                ("SEGON PLAT", plat2),
                ("POSTRES", postres),
            ]

            for etiqueta, p in plats:
                ingredients_abans = list(p.get("ingredients", []) or [])

                resultat = substituir_ingredient(p, estil_latent, kb, mode="latent", intensitat=intensitat)

                # Si l’operador retorna un plat nou, enganxem resultats al dict original
                if isinstance(resultat, dict) and resultat is not p:
                    p.clear()
                    p.update(resultat)

                debug_ingredients_abans_despres(etiqueta, p, ingredients_abans)
        debug_kb_match(plat1, kb, "PRIMER")
        debug_kb_match(plat2, kb, "SEGON")
        debug_kb_match(postres, kb, "POSTRES")

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
                estil_row = kb.estils[estil_tecnic]
                info_llm_1 = genera_descripcio_llm(plat1, transf_1, estil_tecnic, servei, estil_row)
                info_llm_2 = genera_descripcio_llm(plat2, transf_2, estil_tecnic, servei, estil_row)
                info_llm_post = genera_descripcio_llm(postres, transf_post, estil_tecnic, servei, estil_row)


        # 8) Afegir begudes
        print("\n✨ === FASE ADAPTACIÓ: BEGUDES ===")
        beguda1, score1 = recomana_beguda_per_plat(plat1, list(kb.begudes.values()), base_ingredients_list)
        print(f"Beguda per al primer plat:  {beguda1['nom']} (score {score1})")
        beguda2, score2 = recomana_beguda_per_plat(plat2, list(kb.begudes.values()), base_ingredients_list)
        print(f"Beguda per al segon plat:  {beguda2['nom']} (score {score2})")
        beguda_postres, score_postres = recomana_beguda_per_plat(postres, list(kb.begudes.values()), base_ingredients_list)
        print(f"Beguda per al segon plat:  {beguda_postres['nom']} (score {score_postres})")

        
        # 9) Resultat Final
        imprimir_menu_final(plat1, transf_1, info_llm_1, beguda1, plat2, transf_2, info_llm_2, beguda2, postres, transf_post, info_llm_post, beguda_postres)
        
        # 10) Feedback (Revise simple)
        print("\n⭐ FASE REVISE")
        try:
            nota = int(input("Puntua aquest menú (1-5): "))
            if nota >= 4:
                print("✅ M'alegro que t'agradi! (Aquí aniria el RETAIN)")
            else:
                print("📝 Prenem nota per millorar.")
        except: pass

        if input_default("\nSortir? (s/n)", "n").lower() == 's':
            print("Bon profit! 👋")
            break

if __name__ == "__main__":
    main()
