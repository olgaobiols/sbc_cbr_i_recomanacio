# main.py
import json
import csv

from estructura_cas import DescripcioProblema
from retriever_nuevo import Retriever
from operadors_transformacio_realista import *


# =========================
#   CARREGA DE BASES
# =========================

# Base d'ingredients
base_ingredients = []
with open("ingredients.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_ingredients.append(row)

# Base d'estils culinaris (per tècniques)
base_estils = {}
with open("estils.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_estils[row["nom_estil"]] = row

# Base de tècniques
base_tecnniques = {}
with open("tecniques.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_tecnniques[row["nom_tecnica"]] = row

# Estils latents (FlavorGraph)
with open("estils_latents.json", "r", encoding="utf-8") as f:
    base_estils_latents = json.load(f)


# =========================
#   FUNCIONS AUXILIARS CLI
# =========================

def input_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    return txt if txt else default


def input_int_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt:
        return default
    try:
        return int(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return default


def input_float_default(prompt, default):
    txt = input(f"{prompt} [{default}]: ").strip()
    if not txt:
        return float(default)
    try:
        return float(txt)
    except ValueError:
        print("  Valor no vàlid, es fa servir el per defecte.")
        return float(default)


def imprimir_casos(candidats, top_k=5):
    """Mostra per pantalla els millors casos recuperats (adaptat al Retriever nou)."""
    if not candidats:
        print("\nNo s'ha trobat cap cas similar.")
        return

    print(f"\n--- {len(candidats)} CASOS TROBATS (es mostren els {min(top_k, len(candidats))} primers) ---")

    for i, c in enumerate(candidats[:top_k], start=1):
        cas = c["cas"]
        score = c["score_final"]
        detall = c["detall"]
        sol = cas.get("solucio", {})
        pr = cas.get("problema", {})

        etiqueta = "#1 (RECOMANAT)" if i == 1 else f"#{i}"

        # Camps del problema del cas
        tipus = pr.get("tipus_esdeveniment", "—")
        temporada = pr.get("temporada", "—")
        formalitat = pr.get("formalitat", "—")
        servei = pr.get("servei", "—")
        n_com = pr.get("n_comensals", "—")
        preu_pers = pr.get("preu_pers", "—")

        # Menú
        p1 = sol.get("primer_plat", {}).get("nom", "—")
        p2 = sol.get("segon_plat", {}).get("nom", "—")
        p3 = sol.get("postres", {}).get("nom", "—")

        print(f"\n{etiqueta} [Similitud global: {score:.1%}] - ID Cas: {cas.get('id_cas', '—')}")
        print(f"   Problema cas -> Event: {tipus} | Servei: {servei} | Temporada: {temporada} | Formalitat: {formalitat}")
        print(f"   Numèrics     -> Comensals: {n_com} | Preu/pers: {preu_pers}€")
        print(f"   Menú         -> {p1} + {p2} + {p3}")

        # Semàntica / Numèrica (agregades)
        sim_sem = detall.get("sim_semantica", None)
        sim_num = detall.get("sim_numerica", None)
        if sim_sem is not None and sim_num is not None:
            print(f"   Detall global -> Semàntica: {sim_sem:.4f} | Numèrica: {sim_num:.4f}")

        # Breakdown complet (per validar que el retriever va bé)
        # Noms segons el retriever: sim_event, sim_servei, sim_temporada, sim_formalitat, sim_comensals, sim_preu
        bd_keys = ["sim_event", "sim_servei", "sim_temporada", "sim_formalitat", "sim_comensals", "sim_preu"]
        bd_parts = []
        for k in bd_keys:
            if k in detall:
                bd_parts.append(f"{k}={detall[k]:.4f}")
        if bd_parts:
            print("   Breakdown     -> " + " | ".join(bd_parts))


def imprimir_menu_final(
    plat1, transf_1, info_llm_1,
    plat2, transf_2, info_llm_2,
    postres, transf_post, info_llm_post
):
    print("\n============================")
    print("   MENÚ ADAPTAT FINAL")
    print("============================")

    for etiqueta, plat, transf, info_llm in [
        ("Primer plat", plat1, transf_1, info_llm_1),
        ("Segon plat",  plat2, transf_2, info_llm_2),
        ("Postres",     postres, transf_post, info_llm_post),
    ]:
        if info_llm is not None:
            nom = info_llm.get("nom_nou", plat.get("nom", "—"))
            desc = info_llm.get("descripcio_carta", "Versió adaptada del plat.")
            proposta = info_llm.get(
                "proposta_presentacio",
                "Presentació cuidada i coherent amb l'estil, ressaltant el producte principal."
            )
        else:
            nom = plat.get("nom", "—")
            desc = "Versió sense transformacions tècniques especials."
            proposta = "Presentació clàssica i ordenada, ressaltant els ingredients principals."

        ingredients = plat.get("ingredients", []) or []
        print(f"\n{etiqueta}: {nom}")
        if ingredients:
            print(f"  Base del plat: {', '.join(ingredients)}")
        else:
            print("  Base del plat: —")
        print(f"  Descripció de carta: {desc}")
        print(f"  Presentació del plat: {proposta}")


# =========================
#   MAIN INTERACTIU
# =========================

def main():
    print("===========================================")
    print("   RECOMANADOR DE MENÚS RicoRico 2.0")
    print("   (CBR + adaptació d'ingredients i tècniques)")
    print("===========================================\n")

    # 1) Inicialitzem el retriever nou
    retriever = Retriever("base_de_casos.json")

    # 2) Intro
    print("Benvingut/da al recomanador de menús!")
    print("T’ajudaré a trobar un menú semblant als que tenim a la base de casos,")
    print("i després el podem adaptar amb canvi d’ingredients i tècniques.\n")
    print("Si alguna resposta no la tens clara, pots posar 'indiferent' quan surti.\n")

    while True:
        print("\n--- Nova petició ---")

        # 3) Preguntes alineades amb el Retriever nou
        tipus_esdeveniment = input_default(
            "Quin tipus d’esdeveniment? (casament/aniversari/comunio/empresa/congres/altres)",
            "casament"
        )
        temporada = input_default(
            "En quina època de l’any? (primavera/estiu/tardor/hivern/indiferent)",
            "primavera"
        )
        espai = input_default(
            "Espai interior o exterior? (interior/exterior)",
            "interior"
        )
        servei = input_default(
            "Tipus de servei? (assegut/cocktail/finger_food/indiferent)",
            "assegut"
        )
        n_comensals = input_int_default(
            "Quants comensals aproximadament? (enter)",
            80
        )
        formalitat = input_default(
            "Grau de formalitat? (formal/informal/indiferent)",
            "formal"
        )
        preu_pers = input_float_default(
            "Pressupost aproximat per persona (€)? (nombre)",
            40.0
        )

        # 4) Construïm la petició per al Retriever nou
        # IMPORTANT: aquí sí passem 'servei', 'formalitat' real i 'preu_pers'
        problema = DescripcioProblema(
            tipus_esdeveniment=tipus_esdeveniment,
            temporada=temporada,
            formalitat=formalitat,
            n_comensals=n_comensals,
            preu_pers = preu_pers,
            servei=servei,
        )

        # 5) Recuperem casos similars
        resultats = retriever.recuperar_casos_similars(problema, k=5)
        imprimir_casos(resultats, top_k=5)

        if not resultats:
            tornar = input_default("\nNo s'ha trobat res gaire similar. Vols provar una altra petició? (s/n)", "s")
            if not tornar.lower().startswith("s"):
                print("\nGràcies per fer servir el recomanador! 👋")
                break
            continue

        # 6) Escollir cas base
        idx_txt = input_default("\nTria un cas per adaptar (número de la llista, 1..N)", "1")
        try:
            idx = int(idx_txt)
        except ValueError:
            idx = 1
        idx = max(1, min(idx, len(resultats)))

        cas_seleccionat = resultats[idx - 1]["cas"]
        sol = cas_seleccionat["solucio"]

        plat1, plat2, postres = sol["primer_plat"], sol["segon_plat"], sol["postres"]

        print("\nHas triat el menú base:")
        print(f"  - Primer plat: {plat1.get('nom','—')}")
        print(f"  - Segon plat:  {plat2.get('nom','—')}")
        print(f"  - Postres:     {postres.get('nom','—')}")

        # ---------------------------
        # ADAPTACIÓ LATENT (FlavorGraph)
        # ---------------------------
        print("\n=== ADAPTACIÓ D'INGREDIENTS (LATENT) ===")
        print("Conceptes disponibles: " + ", ".join(sorted(base_estils_latents.keys())))

        def preparar_plat_per_transformacio(plat):
            plat_prep = plat.copy()
            if 'ingredients_en' in plat and plat['ingredients_en']:
                plat_prep['ingredients'] = list(plat['ingredients_en'])
            return plat_prep

        estil_latent = input_default("\nConcepte a aplicar (buit per saltar)", "").strip()

        plat1_mod, plat2_mod, postres_mod = plat1, plat2, postres

        if estil_latent and estil_latent in base_estils_latents:
            intensitat = float(input_default("Intensitat (0.1 subtil - 0.9 radical)", "0.5"))

            print(f"\n🔄 Transformant menú cap a '{estil_latent}' (usant ingredients en anglès)...")

            p1_prep = preparar_plat_per_transformacio(plat1)
            p2_prep = preparar_plat_per_transformacio(plat2)
            pp_prep = preparar_plat_per_transformacio(postres)

            plat1_mod = substituir_ingredient(p1_prep, estil_latent, base_ingredients, base_estils_latents,
                                             mode="latent", intensitat=intensitat)
            plat2_mod = substituir_ingredient(p2_prep, estil_latent, base_ingredients, base_estils_latents,
                                             mode="latent", intensitat=intensitat)
            postres_mod = substituir_ingredient(pp_prep, estil_latent, base_ingredients, base_estils_latents,
                                                mode="latent", intensitat=intensitat)

            print("\n[DEBUG] Resultat de la transformació latent:")
            for etiqueta, plat_mod in [
                ("Primer plat", plat1_mod),
                ("Segon plat", plat2_mod),
                ("Postres", postres_mod),
            ]:
                nom = plat_mod.get("nom", "Sense nom")
                ingredients = ", ".join(plat_mod.get("ingredients", [])) or "sense ingredients"
                print(f"  - {etiqueta}: {nom} -> {ingredients}")

        # ------------------------
        # ADAPTACIÓ DE TÈCNIQUES
        # ------------------------
        print("\nAra podem adaptar les TÈCNIQUES culinàries (plating / cuina molecular, etc.).")

        estils_tecnics_keys = sorted(base_estils.keys())
        if estils_tecnics_keys:
            print("Estils tècnics disponibles a estils.csv:")
            for i, key in enumerate(estils_tecnics_keys, start=1):
                display = key.replace("_", " ")
                display = display[0].upper() + display[1:]
                print(f"  {i}. {display} ({key})")
        else:
            print("  [AVÍS] No hi ha estils tècnics definits a estils.csv.")

        estil_tecnic = None
        if estils_tecnics_keys:
            resposta_estil = input_default(
                "\nTria un estil tècnic pel NÚMERO (1..N) o prem Enter per NO aplicar tècniques",
                ""
            ).strip()

            if resposta_estil:
                try:
                    idx_et = int(resposta_estil)
                    if 1 <= idx_et <= len(estils_tecnics_keys):
                        estil_tecnic = estils_tecnics_keys[idx_et - 1]
                    else:
                        print("  [AVÍS] Número fora de rang. No s'aplicaran tècniques noves.")
                except ValueError:
                    print("  [AVÍS] Entrada no vàlida. No s'aplicaran tècniques noves.")

        transf_1, transf_2, transf_post = [], [], []
        info_llm_1 = info_llm_2 = info_llm_post = None

        plat1_final, plat2_final, postres_final = plat1_mod, plat2_mod, postres_mod

        if estil_tecnic:
            print(f"\n### ADAPTACIÓ DE TÈCNIQUES AL NOU ESTIL: '{estil_tecnic}' ###")

            transf_1 = triar_tecniques_per_plat(
                plat1_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients
            )
            transf_2 = triar_tecniques_per_plat(
                plat2_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients
            )
            transf_post = triar_tecniques_per_plat(
                postres_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients
            )

            estil_row = base_estils.get(estil_tecnic)
            usar_llm = input_default(
                "\nVols que Gemini generi noms nous i descripcions? (s/n)",
                "n"
            ).strip().lower().startswith("s")

            if usar_llm:
                info_llm_1 = genera_descripcio_llm(plat1_mod, transf_1, estil_tecnic, servei, estil_row)
                info_llm_2 = genera_descripcio_llm(plat2_mod, transf_2, estil_tecnic, servei, estil_row)
                info_llm_post = genera_descripcio_llm(postres_mod, transf_post, estil_tecnic, servei, estil_row)

                plat1_final, plat2_final, postres_final = plat1_mod.copy(), plat2_mod.copy(), postres_mod.copy()
                plat1_final["nom"] = info_llm_1["nom_nou"]
                plat2_final["nom"] = info_llm_2["nom_nou"]
                postres_final["nom"] = info_llm_post["nom_nou"]
            else:
                print("[DEBUG] Saltant la generació amb Gemini; es mantenen els noms originals.")
        else:
            print("\nNo s'apliquen tècniques noves (es manté el cas base / adaptat d'ingredients).")

        # ------------------------
        # MENÚ FINAL
        # ------------------------
        imprimir_menu_final(
            plat1_final, transf_1, info_llm_1,
            plat2_final, transf_2, info_llm_2,
            postres_final, transf_post, info_llm_post
        )

        # ------------------------
        # IMATGE (opcional)
        # ------------------------
        resp_img = input_default("\nVols generar una imatge realista del menú? (s/n)", "n")
        if resp_img.lower().startswith("s"):
            try:
                plats_info = [
                    {
                        "curs": "First course",
                        "nom": info_llm_1["nom_nou"] if info_llm_1 else plat1_final.get("nom", ""),
                        "descripcio": info_llm_1.get("descripcio_carta", "") if info_llm_1 else "",
                        "presentacio": info_llm_1.get("proposta_presentacio", "") if info_llm_1 else "",
                        "ingredients": plat1_final.get("ingredients", []),
                    },
                    {
                        "curs": "Main course",
                        "nom": info_llm_2["nom_nou"] if info_llm_2 else plat2_final.get("nom", ""),
                        "descripcio": info_llm_2.get("descripcio_carta", "") if info_llm_2 else "",
                        "presentacio": info_llm_2.get("proposta_presentacio", "") if info_llm_2 else "",
                        "ingredients": plat2_final.get("ingredients", []),
                    },
                    {
                        "curs": "Dessert",
                        "nom": info_llm_post["nom_nou"] if info_llm_post else postres_final.get("nom", ""),
                        "descripcio": info_llm_post.get("descripcio_carta", "") if info_llm_post else "",
                        "presentacio": info_llm_post.get("proposta_presentacio", "") if info_llm_post else "",
                        "ingredients": postres_final.get("ingredients", []),
                    },
                ]

                prompt_imatge = construir_prompt_imatge_menu(
                    tipus_esdeveniment=tipus_esdeveniment,
                    temporada=temporada,
                    espai=espai,
                    formalitat=formalitat,
                    plats_info=plats_info,
                )
                genera_imatge_menu_hf(prompt_imatge, output_path="menu_event_actual.png")

            except Exception as e:
                print(f"[IMATGE] No s'ha pogut generar la imatge: {e}")

        # Tornar a començar?
        continuar = input_default("\nVols demanar una altra recomanació? (s/n)", "s")
        if not continuar.lower().startswith("s"):
            print("\nGràcies per fer servir el recomanador! 👋")
            break


if __name__ == "__main__":
    main()
