import json
import csv
from estructura_cas import DescripcioProblema
from retriever import Retriever
from operadors_transformacio_realista import *

# =========================
#   CARREGA DE BASES
# =========================

# Base d'ingredients
base_ingredients = []
with open("data/ingredients.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_ingredients.append(row)

# Base d'estils culinaris (per tècniques)
base_estils = {}
with open("data/estils.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_estils[row["nom_estil"]] = row

# Base de tècniques
base_tecnniques = {}
with open("data/tecniques.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_tecnniques[row["nom_tecnica"]] = row

# Carreguem els nous estils latents des del JSON (substitueix provisional.py)
with open("data/estils_latents.json", "r", encoding="utf-8") as f:
    base_estils_latents = json.load(f)


# =========================
#   FUNCIONS AUXILIARS CLI
# =========================

def input_default(prompt, default):
    """Demana un input amb valor per defecte."""
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


def imprimir_casos(candidats, top_k=5):
    """Mostra per pantalla els millors casos recuperats."""
    if not candidats:
        print("\nNo s'ha trobat cap cas similar")
        return

    print(f"\n--- {len(candidats)} CASOS TROBATS (es mostren els {min(top_k, len(candidats))} primers) ---")
    for i, c in enumerate(candidats[:top_k], start=1):
        cas = c["cas"]
        score = c["score_final"]
        detall = c["detall"]
        sol = cas["solucio"]
        etiqueta = "#1 (RECOMANAT)" if i == 1 else f"#{i}"
        print(f"\n{etiqueta} [Similitud: {score:.1%}] - ID Cas: {cas['id_cas']}")
        print(f"   Estil original: {cas['problema']['estil_culinari']} ({cas['problema']['tipus_esdeveniment']})")
        print(f"   Preu total: {sol['preu_total']}€  |  Comensals: {cas['problema']['n_comensals']}")
        print(f"   Menú: {sol['primer_plat']['nom']} + {sol['segon_plat']['nom']} + {sol['postres']['nom']}")
        print(f"   Detall similitud -> Semàntica: {detall['sim_semantica_global']:.4f} | Numèrica: {detall['sim_num']:.4f}")

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
            nom = info_llm.get("nom_nou", plat["nom"])
            desc = info_llm.get("descripcio_carta", "Versió adaptada del plat.")
            proposta = info_llm.get(
                "proposta_presentacio",
                "Presentació cuidada i coherent amb l'estil, ressaltant el producte principal."
            )
        else:
            nom = plat["nom"]
            desc = "Versió sense transformacions tècniques especials."
            proposta = "Presentació clàssica i ordenada, ressaltant els ingredients principals."

        print(f"\n{etiqueta}: {nom}")
        print(f"  Base del plat: {', '.join(plat['ingredients'])}")
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

    # 1) Inicialitzem primer el retriever (carrega model + embeddings)
    retriever = Retriever("data/base_de_casos.json")

    # 2) Després fem la intro "humana"
    print("Benvingut/da al recomanador de menús!")
    print("T’ajudaré a trobar un menú semblant als que tenim a la base de casos,")
    print("i després el podem 'tocar' amb canvi d’ingredients i tècniques.\n")
    print("Si alguna resposta no la tens clara, pots posar una opció aproximada.\n")

    # Només info dels estils que ja tens a la base (per context, però no ho preguntem)
    try:
        estils_disponibles = sorted({c["problema"]["estil_culinari"] for c in retriever.base_casos})
        print("Alguns estils culinaris presents a la base de casos:")
        print("  - " + ", ".join(estils_disponibles))
    except Exception:
        print("Alguns estils culinaris típics: mediterrani_fresc, oriental_fusio, tradicional_espanyol, confort_food...")
    print("\n(No cal que triïs estil ara; el sistema buscarà casos similars de forma global.)\n")

    while True:
        print("\n--- Nova petició ---")

        # 3) Preguntes tipus RicoRico (sense preu)
        tipus_esdeveniment = input_default("Quin tipus d’esdeveniment estàs organitzant? (casament/aniversari/comunio/empresa/congres/altres)", "casament")
        temporada = input_default("En quina època de l’any se celebrarà? (primavera/estiu/tardor/hivern)", "primavera")
        espai = input_default("Es farà en un espai interior o exterior? (interior/exterior)", "interior")
        n_comensals = input_int_default("Quants comensals assistiran aproximadament? (nombre enter)", 80)
        formalitat = input_default("Quin grau de formalitat busques? (formal/informal)", "formal")

        # No preguntem ja l'estil de cuina: el marquem com "indiferent"
        problema = DescripcioProblema(
                    tipus_esdeveniment=tipus_esdeveniment,
                    estil_culinari=f"indiferent",
                    n_comensals=n_comensals,
                    temporada=temporada,
                    pressupost_max=999.0,
                    restriccions=[],
                    formalitat="formal",
                )
        
        # 4) Recuperem casos similars
        resultats = retriever.recuperar_casos_similars(problema)
        imprimir_casos(resultats, top_k=5)

        if not resultats:
            tornar = input_default("\nNo s'ha trobat res gaire similar. Vols provar una altra petició? (s/n)", "s")
            if not tornar.lower().startswith("s"):
                print("\nGràcies per fer servir el recomanador! 👋")
                break
            else:
                continue

        # 5) Escollir cas base (per defecte el #1, que ja és la recomanació)
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
        print(f"  - Primer plat: {plat1['nom']}")
        print(f"  - Segon plat:  {plat2['nom']}")
        print(f"  - Postres:     {postres['nom']}")

        # ---------------------------
        # ADAPTACIÓ LATENT (FlavorGraph)
        # ---------------------------
        print("\n=== ADAPTACIÓ D'INGREDIENTS (LATENT) ===")
        print("Conceptes disponibles: " + ", ".join(sorted(base_estils_latents.keys())))
        
        def preparar_plat_per_transformacio(plat):
            """
            Prepara el plat per a l'adaptació latent.
            Si el plat té 'ingredients_en', els posa com a 'ingredients' principals
            perquè FlavorGraph treballi amb ells.
            """
            plat_prep = plat.copy()
            if 'ingredients_en' in plat and plat['ingredients_en']:
                # Usem la llista en anglès per a la transformació
                plat_prep['ingredients'] = list(plat['ingredients_en'])
            else:
                # Fallback: si no hi ha anglès, usem el que hi hagi, però avisem
                # (FlavorGraph pot fallar si rep català)
                pass 
            return plat_prep

        estil_latent = input_default("\nConcepte a aplicar (buit per saltar)", "").strip()

        plat1_mod, plat2_mod, postres_mod = plat1, plat2, postres

        if estil_latent and estil_latent in base_estils_latents:
            intensitat = float(input_default("Intensitat (0.1 subtil - 0.9 radical)", "0.5"))
            
            print(f"\n🔄 Transformant menú cap a '{estil_latent}' (usant ingredients en anglès)...")
            
            # PREPARACIÓ: Passem a anglès perquè FlavorGraph entengui els ingredients
            p1_prep = preparar_plat_per_transformacio(plat1)
            p2_prep = preparar_plat_per_transformacio(plat2)
            pp_prep = preparar_plat_per_transformacio(postres)
            
            # TRANSFORMACIÓ
            # Mode 'latent' fixat, usant la base d'estils latents
            plat1_mod = substituir_ingredient(p1_prep, estil_latent, base_ingredients, base_estils_latents, mode="latent", intensitat=intensitat)
            plat2_mod = substituir_ingredient(p2_prep, estil_latent, base_ingredients, base_estils_latents, mode="latent", intensitat=intensitat)
            postres_mod = substituir_ingredient(pp_prep, estil_latent, base_ingredients, base_estils_latents, mode="latent", intensitat=intensitat)

            # Debug: mostra els plats resultants i ingredients en anglès
            print("\n[DEBUG] Resultat de la transformació latent:")
            for etiqueta, plat_mod in [
                ("Primer plat", plat1_mod),
                ("Segon plat", plat2_mod),
                ("Postres", postres_mod),
            ]:
                nom = plat_mod.get("nom", "Sense nom")
                ingredients = ", ".join(plat_mod.get("ingredients", [])) or "sense ingredients"
                print(f"  - {etiqueta}: {nom} -> {ingredients}")
            
            # NOTA: Els plats modificats ara tenen 'ingredients' en anglès (fruit del FlavorGraph)
            # Això està bé per mostrar el resultat "creatiu".


        # ------------------------
        # 7) ADAPTACIÓ DE TÈCNIQUES
        # ------------------------
        print("\nAra podem adaptar les TÈCNIQUES culinàries (plating / cuina molecular, etc.).")

        # Llista ordenada d'estils tècnics disponibles
        estils_tecnics_keys = sorted(base_estils.keys())
        if estils_tecnics_keys:
            print("Estils tècnics disponibles a estils.csv:")
            for i, key in enumerate(estils_tecnics_keys, start=1):
                # nom presentable: "cuina_molecular" -> "Cuina molecular"
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

        MAX_TEC_PER_PLAT = 2  # màxim 2 tècniques per plat per fer-ho creïble

        # Inicialitzem llistes de transformacions i info del LLM
        transf_1, transf_2, transf_post = [], [], []
        info_llm_1 = info_llm_2 = info_llm_post = None

        if estil_tecnic:
            print(f"\n### ADAPTACIÓ DE TÈCNIQUES AL NOU ESTIL: '{estil_tecnic}' ###")

            # 1) Triem tècniques
            transf_1 = triar_tecniques_per_plat(
                plat1_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients,
                max_tecniques=MAX_TEC_PER_PLAT
            )
            transf_2 = triar_tecniques_per_plat(
                plat2_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients,
                max_tecniques=MAX_TEC_PER_PLAT
            )
            transf_post = triar_tecniques_per_plat(
                postres_mod, estil_tecnic, base_estils, base_tecnniques, base_ingredients,
                max_tecniques=MAX_TEC_PER_PLAT
            )

            # 2) LLM: nom nou, descripció i justificació per cada plat
            estil_row = base_estils.get(estil_tecnic)
            info_llm_1 = genera_descripcio_llm(plat1_mod, transf_1, estil_tecnic, estil_row)
            info_llm_2 = genera_descripcio_llm(plat2_mod, transf_2, estil_tecnic, estil_row)
            info_llm_post = genera_descripcio_llm(postres_mod, transf_post, estil_tecnic, estil_row)

            # Fem servir versions "modificades" per al menú final (el nom del LLM)
            plat1_final, plat2_final, postres_final = plat1_mod.copy(), plat2_mod.copy(), postres_mod.copy()
            plat1_final["nom"] = info_llm_1["nom_nou"]
            plat2_final["nom"] = info_llm_2["nom_nou"]
            postres_final["nom"] = info_llm_post["nom_nou"]

        else:
            print("\nNo s'apliquen tècniques noves (es manté el cas base / adaptat d'ingredients).")
            plat1_final, plat2_final, postres_final = plat1_mod, plat2_mod, postres_mod
            # transf_1, transf_2, transf_post ja són [] per defecte

        # ------------------------
        # 8) MENÚ FINAL RESUMIT
        # ------------------------
        imprimir_menu_final(
            plat1_final, transf_1, info_llm_1,
            plat2_final, transf_2, info_llm_2,
            postres_final, transf_post, info_llm_post
        )

        # 9) Tornar a començar?
        continuar = input_default("\nVols demanar una altra recomanació? (s/n)", "s")
        if not continuar.lower().startswith("s"):
            print("\nGràcies per fer servir el recomanador! 👋")
            break


if __name__ == "__main__":
    main()
