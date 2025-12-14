from dataclasses import dataclass, asdict, field

@dataclass 
class Plat: 
    nom: str
    ingredients: list          # llista d'ingredients
    curs: str                # 'primer', 'segon', 'postres'
    estil_tags: list         # llista d'estils culinàries
    preu: float  = 0.0             # preu del plat
    
    def to_dict(self):
        return asdict(self)


@dataclass            
class DescripcioProblema:
    # --- camps OBLIGATORIS (sense default) ---
    tipus_esdeveniment: str
    temporada: str
    n_comensals: int
    preu_pers: float
    servei: str

    # --- camps OPCIONALS (amb default) ---
    formalitat: str = "indiferent"


@dataclass
class SolucioMenu: 
    primer_plat: Plat
    segon_plat: Plat
    postres: Plat
    begudes: list            # llista de begudes
    preu_total: float
    descripcio: str = ""        # descripció textual de la solució
    
    def to_dict(self):
        return asdict(self)
    
@dataclass
class AvaluacioCas:
    derivacio: str    # origen del cas: cas real, derivat, etc
    feedback_textual: str     # puntuacio estrelles i/o text amb comentaris
    validat: bool     # exit o fracàs (ja veurem si boolea o puntuació)
    utilitat: float  # valoració numèrica de la solució
    
    def to_dict(self):
        return asdict(self)
    
@dataclass
class CasMenu:
    id_cas: int
    problema: DescripcioProblema 
    solucio: SolucioMenu
    avaluacio: AvaluacioCas