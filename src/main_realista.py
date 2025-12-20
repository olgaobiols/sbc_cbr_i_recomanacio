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

# =========================
#   INICIALITZACIÓ GLOBAL
# =========================

# 1. Carreguem el cervell (Ontologia + Estils)
kb = KnowledgeBase()

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
        p1 = sol.get("primer_plat", {}).get("nom", "—")
        p2 = sol.get("segon_plat", {}).get("nom", "—")
        p3 = sol.get("postres", {}).get("nom", "—")

        print(f"\n{etiqueta} [Similitud: {score:.1%}] - ID: {cas.get('id_cas', '?')}")
        print(f"   Context:  {event} | {pr.get('temporada','?')} | {pr.get('servei','?')}{str_restr}")
        print(f"   Menú:     1. {p1} | 2. {p2} | 3. {p3}")
        
        # Detall de puntuació (útil per debug/demo)
        parts = []
        if "Restriccions" in detall: parts.append(f"Restr={detall['Restriccions']:.2f}")
        if "Event" in detall: parts.append(f"Event={detall['Event']:.2f}")
        print(f"   Detall:   {' | '.join(parts)}")

def imprimir_menu_final(plat1, transf_1, info_llm_1, plat2, transf_2, info_llm_2, postres, transf_post, info_llm_post):
    print("\n" + "="*40)
    print("      🍽️  MENÚ ADAPTAT FINAL  🍽️")
    print("="*40)

    for etiqueta, plat, info_llm in [
        ("PRIMER PLAT", plat1, info_llm_1),
        ("SEGON PLAT",  plat2, info_llm_2),
        ("POSTRES",     postres, info_llm_post),
    ]:
        nom = info_llm.get("nom_nou", plat.get("nom", "—")) if info_llm else plat.get("nom", "—")
        desc = info_llm.get("descripcio_carta", "") if info_llm else "Plat clàssic."
        
        print(f"\n🔹 {etiqueta}: {nom}")
        ings = ", ".join(plat.get("ingredients", []))
        print(f"   Ingredients: {ings}")
        if desc:
            print(f"   Carta: {desc}")
        
        # Si hi ha logs de canvis, els mostrem (Explicabilitat XCBR)
        logs = plat.get("log_transformacio", [])
        if logs:
            print("   🛠️  Adaptacions realitzades:")
            for log in logs:
                print(f"      - {log}")

# =========================
#   MAIN INTERACTIU
# =========================

def main():
    print("===========================================")
    print("   RECOMANADOR DE MENÚS RicoRico 3.0")
    print("   (CBR Híbrid: Ontologia + FlavorGraph)")
    print("===========================================\n")

    # 1) Inicialitzem el Retriever
    retriever = Retriever("data/base_de_casos.json")

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
        
        # Creem còpies de treball dels plats
        plat1 = sol["primer_plat"].copy()
        plat2 = sol["segon_plat"].copy()
        postres = sol["postres"].copy()
        
        # 6) Adaptació 1: Ingredients i Estil Latent (FlavorGraph)
        print("\n🎨 === FASE ADAPTACIÓ: INGREDIENTS ===")
        # Si l'usuari ha demanat un estil al principi, el suggerim aquí
        suggeriment = estil_culinari if estil_culinari in kb.estils_latents else ""
        estil_latent = input_default(f"Vols aplicar un 'toc' d'estil latent? (ex: picant, thai...) [{suggeriment}]", suggeriment)
        
        if estil_latent:
            intensitat = float(input_default("Intensitat de l'adaptació (0.1 - 0.9)?", "0.5"))
            print(f"🔄 Adaptant ingredients cap a '{estil_latent}'...")
            
            # Nota: Usem 'base_ingredients_list' per compatibilitat amb funcions antigues
            for p in [plat1, plat2, postres]:
                # Hack: si el plat ve de JSON, potser no té 'ingredients' com a llista neta
                substituir_ingredient(p, estil_latent, base_ingredients_list, kb.estils_latents, mode="latent", intensitat=intensitat)

        # 7) Adaptació 2: Tècniques i Presentació
        print("\n✨ === FASE ADAPTACIÓ: TÈCNIQUES ===")
        kb.llista_estils() # Podries imprimir-los
        estil_tecnic = input_default("Vols aplicar un estil tècnic? (ex: cuina_molecular, rustica) [Enter per saltar]", "")
        
        transf_1, transf_2, transf_post = [], [], []
        info_llm_1, info_llm_2, info_llm_post = None, None, None
        
        if estil_tecnic and estil_tecnic in kb.estils:
            print(f"⚙️  Aplicant tècniques de '{estil_tecnic}'...")
            transf_1 = triar_tecniques_per_plat(plat1, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list)
            transf_2 = triar_tecniques_per_plat(plat2, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list)
            transf_post = triar_tecniques_per_plat(postres, estil_tecnic, kb.estils, kb.tecniques, base_ingredients_list)
            
            # Generació de Text (Gemini)
            if input_default("Generar nous noms i descripcions amb Gemini? (s/n)", "n").lower() == 's':
                estil_row = kb.estils[estil_tecnic]
                info_llm_1 = genera_descripcio_llm(plat1, transf_1, estil_tecnic, servei, estil_row)
                info_llm_2 = genera_descripcio_llm(plat2, transf_2, estil_tecnic, servei, estil_row)
                info_llm_post = genera_descripcio_llm(postres, transf_post, estil_tecnic, servei, estil_row)

        # 8) Resultat Final
        imprimir_menu_final(plat1, transf_1, info_llm_1, plat2, transf_2, info_llm_2, postres, transf_post, info_llm_post)
        
        # 9) Feedback (Revise simple)
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