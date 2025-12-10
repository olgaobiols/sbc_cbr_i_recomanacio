import json
import csv
import unicodedata
from typing import Optional, Set
from estructura_cas import DescripcioProblema
from retriever import Retriever
from operadors_transformacio_realista import *
from operador_ingredients import substituir_ingredients_prohibits, _get_info_ingredient, _check_compatibilitat


def _normalize_restriccio_text(value: str) -> str:
    if not value:
        return ""
    text = unicodedata.normalize("NFKD", str(value))
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    for ch in ("-", "_"):
        text = text.replace(ch, " ")
    text = " ".join(text.split())
    return text


_ALERGENS_ALIAS_RAW = {
    "fruits secs": "nuts",
    "frutos secos": "nuts",
    "fruit secs": "nuts",
    "fruit sec": "nuts",
    "fruitos secs": "nuts",
    "cacauet": "peanuts",
    "cacahuet": "peanuts",
    "cacahuete": "peanuts",
    "cacahuetes": "peanuts",
    "cacahuate": "peanuts",
    "cacauets": "peanuts",
    "marisc": "crustaceans",
    "mariscos": "crustaceans",
    "crustacis": "crustaceans",
    "crustaceos": "crustaceans",
    "molusc": "molluscs",
    "moluscs": "molluscs",
    "moluscos": "molluscs",
    "peix": "fish",
    "pescado": "fish",
    "peixos": "fish",
    "lactosa": "milk",
    "lacti": "milk",
    "llet": "milk",
    "dairy": "milk",
    "caseina": "milk",
    "ou": "egg",
    "ous": "egg",
    "huevo": "egg",
    "huevos": "egg",
    "soja": "soy",
    "sesam": "sesame",
    "sesamo": "sesame",
    "fructe sec": "nuts",
    "gluten": "gluten",
}
ALERGENS_ALIAS = {
    _normalize_restriccio_text(k): _normalize_restriccio_text(v)
    for k, v in _ALERGENS_ALIAS_RAW.items()
}

DIETA_PATTERNS = [
    ("vega", "vegan"),
    ("vegetar", "vegetarian"),
    ("plant based", "vegan"),
    ("sense carn", "vegetarian"),
    ("halal", "halal friendly"),
    ("kosher", "kosher friendly"),
]

IGNORE_RESTRICCIO_TOKENS = {
    "",
    "cap",
    "cap alergia",
    "cap allergia",
    "cap al.lergia",
    "sense",
    "res",
    "no",
    "ninguna",
    "ningun",
    "cap dieta",
}

# =========================
#   CARREGA DE BASES
# =========================

# Base d'ingredients (Català, per heurístiques clàssiques)
base_ingredients = []
with open("data/ingredients.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_ingredients.append(row)

# Base d'ingredients en anglès (per FlavorGraph i adaptació latent)
base_ingredients_en = []
with open("data/ingredients_en.csv", "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        base_ingredients_en.append(row)

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

def clonar_plat(plat: dict) -> dict:
    nou = plat.copy()
    if "ingredients" in plat:
        nou["ingredients"] = list(plat["ingredients"])
    if "ingredients_en" in plat:
        nou["ingredients_en"] = list(plat["ingredients_en"])
    return nou


def _map_dieta_token(token: str) -> str | None:
    for prefix, canonical in DIETA_PATTERNS:
        if token.startswith(prefix):
            return canonical
    return None


def _construir_perfil_restriccions(text: str) -> dict | None:
    if not text:
        return None
    tokens = [t for t in text.split(",") if t.strip()]
    alergies = []
    dieta = None
    for raw in tokens:
        token = _normalize_restriccio_text(raw)
        if not token or token in IGNORE_RESTRICCIO_TOKENS:
            continue
        if token.startswith("sense "):
            token = token.replace("sense ", "", 1).strip()
        if token.startswith("no "):
            token = token.replace("no ", "", 1).strip()
        if token.startswith("alergia "):
            token = token.replace("alergia ", "", 1).strip()
        if token.startswith("allergia "):
            token = token.replace("allergia ", "", 1).strip()
        diet = _map_dieta_token(token)
        if diet:
            dieta = diet
            continue
        alergia = ALERGENS_ALIAS.get(token, token)
        if alergia:
            alergies.append(alergia)
    perfil = {}
    if alergies:
        perfil["alergies"] = set(alergies)
    if dieta:
        perfil["dieta"] = dieta
    return perfil if perfil else None


def _ingredients_incompatibles_perfil(ingredients: list[str], perfil: dict | None) -> set[str]:
    if not perfil:
        return set()
    prohibits = set()
    for ing in ingredients:
        info = _get_info_ingredient(ing, base_ingredients_en)
        if not info:
            continue
        if not _check_compatibilitat(info, perfil):
            prohibits.add(info.get("ingredient_name", ing))
    return prohibits


def adaptar_plat_a_restriccions(plat: dict, perfil: dict | None, ingredients_usats: Optional[set[str]] = None) -> tuple[dict, list[str]]:
    if not perfil:
        return plat, []
    ingredients_en = list(plat.get("ingredients_en") or [])
    if not ingredients_en:
        return plat, []
    prohibits = _ingredients_incompatibles_perfil(ingredients_en, perfil)
    if not prohibits:
        return plat, []
    plat_tmp = {"nom": plat.get("nom", ""), "ingredients": ingredients_en}
    adaptat = substituir_ingredients_prohibits(
        plat_tmp,
        prohibits,
        base_ingredients_en,
        perfil_usuari=perfil,
        ingredients_usats=ingredients_usats
    )
    nou_plat = plat.copy()
    nou_plat["ingredients"] = list(adaptat["ingredients"])
    nou_plat["ingredients_en"] = list(adaptat["ingredients"])
    return nou_plat, adaptat.get("log_transformacio", [])


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


def demanar_restriccions_usuari() -> dict | None:
    resposta = input_default(
        "Hi ha alguna al·lèrgia, ingredient prohibit o dieta específica? (separa per comes)",
        ""
    ).strip()
    perfil = _construir_perfil_restriccions(resposta)
    if perfil:
        print(f"[INFO] Perfil dietètic detectat: {perfil}")
    return perfil



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
        
        plat1 = clonar_plat(sol["primer_plat"])
        plat2 = clonar_plat(sol["segon_plat"])
        postres = clonar_plat(sol["postres"])

        print("\nHas triat el menú base:")
        print(f"  - Primer plat: {plat1['nom']}")
        print(f"  - Segon plat:  {plat2['nom']}")
        print(f"  - Postres:     {postres['nom']}")

        perfil_restriccions = demanar_restriccions_usuari()
        if perfil_restriccions:
            print("\n=== ADAPTACIÓ PER RESTRICCIONS D'AL·LÈRGIES/DIETES ===")
            plats_actualitzats = []
            hi_ha_canvis = False
            ingredients_usats_restriccions: Set[str] = set()
            for etiqueta, plat in [
                ("Primer plat", plat1),
                ("Segon plat", plat2),
                ("Postres", postres),
            ]:
                plat_adaptat, logs = adaptar_plat_a_restriccions(
                    plat, perfil_restriccions, ingredients_usats_restriccions
                )
                if logs:
                    hi_ha_canvis = True
                    print(f"  {etiqueta} ({plat['nom']}):")
                    for lin in logs:
                        print(f"    - {lin}")
                plats_actualitzats.append(plat_adaptat)
            if not hi_ha_canvis:
                print("  Cap ingredient requeria canvis; el menú ja complia les restriccions.")
            plat1, plat2, postres = plats_actualitzats
        else:
            perfil_restriccions = None

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
            ingredients_usats_latent = set()
            plat1_mod = substituir_ingredient(
                p1_prep, estil_latent, base_ingredients_en, base_estils_latents,
                mode="latent", intensitat=intensitat,
                ingredients_usats_latent=ingredients_usats_latent,
                perfil_usuari=perfil_restriccions
            )
            plat2_mod = substituir_ingredient(
                p2_prep, estil_latent, base_ingredients_en, base_estils_latents,
                mode="latent", intensitat=intensitat,
                ingredients_usats_latent=ingredients_usats_latent,
                perfil_usuari=perfil_restriccions
            )
            postres_mod = substituir_ingredient(
                pp_prep, estil_latent, base_ingredients_en, base_estils_latents,
                mode="latent", intensitat=intensitat,
                ingredients_usats_latent=ingredients_usats_latent,
                perfil_usuari=perfil_restriccions
            )

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

        # Per defecte, el menú final usa els plats adaptats sense tocar els noms
        plat1_final, plat2_final, postres_final = plat1_mod, plat2_mod, postres_mod

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
            usar_llm = input_default(
                "\nVols que Gemini generi noms nous i descripcions? (s/n)", "n"
            ).strip().lower().startswith("s")

            if usar_llm:
                info_llm_1 = genera_descripcio_llm(plat1_mod, transf_1, estil_tecnic, estil_row)
                info_llm_2 = genera_descripcio_llm(plat2_mod, transf_2, estil_tecnic, estil_row)
                info_llm_post = genera_descripcio_llm(postres_mod, transf_post, estil_tecnic, estil_row)

                # Fem servir versions "modificades" per al menú final (el nom del LLM)
                plat1_final, plat2_final, postres_final = plat1_mod.copy(), plat2_mod.copy(), postres_mod.copy()
                plat1_final["nom"] = info_llm_1["nom_nou"]
                plat2_final["nom"] = info_llm_2["nom_nou"]
                postres_final["nom"] = info_llm_post["nom_nou"]
            else:
                print("[DEBUG] Saltant la generació amb Gemini; es mantenen els noms originals.")

        else:
            print("\nNo s'apliquen tècniques noves (es manté el cas base / adaptat d'ingredients).")
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
