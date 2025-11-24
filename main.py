from estructura_cas import DescripcioProblema
from retriever import Retriever

def imprimir_resultat(candidats):
    print(f"\n--- {len(candidats)} CASOS TROBATS ---")
    for i, c in enumerate(candidats):
        cas = c['cas']
        score = c['score_final']
        detall = c['detall']
        
        print(f"\n#{i+1} [Similitud: {score:.1%}] - ID Cas: {cas['id_cas']}")
        print(f"   Estil Original: {cas['problema']['estil_culinari']} ({cas['problema']['tipus_esdeveniment']})")
        print(f"   Preu: {cas['solucio']['preu_total']}€ (vs. Comensals: {cas['problema']['n_comensals']})")
        print(f"   Menú: {cas['solucio']['primer_plat']['nom']} + {cas['solucio']['segon_plat']['nom']}")
        print(f"   Detall: Semàntica {detall['sim_semantica']} | Numèrica {detall['sim_numerica']}")

def main():
    # 1. Inicialitzem el motor de recuperació
    # Assegura't que existeix 'base_de_casos.json' (executa generador_base_casos.py abans)
    retriever = Retriever("base_de_casos.json")

    # --- ESCENARI DE PROVA 1: Cuina Asiàtica ---
    print("\n\nESCENARI 1: Usuari demana 'Sopar empresa asiàtic'")
    # Nota: Al CSV tenim 'oriental_fusio' i 'japones', però l'usuari escriu 'asiàtic'.
    # El model de llenguatge haurà de fer la connexió.
    peticio_1 = DescripcioProblema(
        tipus_esdeveniment="sopar empresa",
        estil_culinari="asiàtic modern",
        n_comensals=50,
        temporada="primavera",
        pressupost_max=40.0,
        restriccions=["cap"],
        formalitat="informal"
    )
    
    resultats_1 = retriever.recuperar_casos_similars(peticio_1)
    imprimir_resultat(resultats_1)

    # --- ESCENARI DE PROVA 2: Boda Tradicional ---
    print("\n\nESCENARI 2: Usuari demana 'Boda clàssica'")
    # Provem si troba els casos 'mediterrani_fresc' o 'tradicional_espanyol'
    peticio_2 = DescripcioProblema(
        tipus_esdeveniment="boda",
        estil_culinari="clàssic tradicional",
        n_comensals=90,
        temporada="estiu",
        pressupost_max=100.0,
        restriccions=["cap"],
        formalitat="formal"
    )

    resultats_2 = retriever.recuperar_casos_similars(peticio_2)
    imprimir_resultat(resultats_2)

if __name__ == "__main__":
    main()