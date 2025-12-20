from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional

@dataclass 
class Plat: 
    nom: str
    ingredients: List[str]          # Llista normalitzada segons ontologia
    curs: str                       # 'primer', 'segon', 'postres'
    estil_tags: List[str] = field(default_factory=list) # Ex: ['mediterrani', 'fresc']
    rols_ingredients: List[str] = field(default_factory=list) # ['main', 'base', 'seasoning'...] (per 3.5)
    tecniques: List[str] = field(default_factory=list)        # Per justificar adaptació
    preu: float = 0.0             

    def to_dict(self):
        return asdict(self)

@dataclass
class Beguda:
    nom: str
    categoria: str # 'vi_blanc', 'aigua', 'refresc'
    maridatge_amb: str = "general" # 'primer', 'segon' o 'general'

# --- 1. ESPAI DEL PROBLEMA (P) ---
@dataclass            
class DescripcioProblema:
    # Camps OBLIGATORIS (Inputs del Retrieve)
    tipus_esdeveniment: str     # Vocabulari: casament, congres...
    n_comensals: int
    preu_pers_objectiu: float   # Canviat de 'preu_pers' per claredat
    temporada: str              # Vocabulari: estiu, hivern...
    servei: str                 # Vocabulari: assegut, cocktail...
    
    # Camps NOUS (Segons Memòria 3.2)
    estil_culinari: str = ""    # Objectiu d'adaptació (ex: 'japonès')
    restriccions: Set[str] = field(default_factory=set) # {'celiac', 'vegan'}
    formalitat: str = "indiferent"

    def to_dict(self):
        # Convertim set a list per JSON serialització
        d = asdict(self)
        d['restriccions'] = list(self.restriccions)
        return d

# --- 2. ESPAI DE LA SOLUCIÓ (S) ---
@dataclass
class SolucioMenu: 
    primer_plat: Plat
    segon_plat: Plat
    postres: Plat
    begudes: List[Beguda] = field(default_factory=list)
    preu_total_real: float = 0.0
    
    # Traçabilitat (Segons Memòria 3.3 - XCBR)
    descripcio_final: str = ""  
    logs_transformacio: List[str] = field(default_factory=list) # "Substituït X per Y..."
    
    def to_dict(self):
        return asdict(self)
    
# --- 3. ESPAI D'AVALUACIÓ (E) ---
@dataclass
class AvaluacioCas:
    derivacio: str = "original"       # ID del cas pare o 'generat'
    
    # Feedback Multidimensional (Segons Memòria 3.4)
    puntuacio_global: int = 0         # 1-5 estrelles
    feedback_textual: str = ""        # Comentaris lliures
    ingredients_rebutjats: List[str] = field(default_factory=list) # Canal A/B learning
    
    # Mètriques de Retain
    cost_adaptacio: int = 0           # Nombre de logs
    utilitat: float = 0.0             # Valor calculat
    
    # Resultat
    tipus_resultat: str = "pendent"   # 'exit', 'fracas_suau', 'fracas_critic'
    validat: bool = False
    
    def to_dict(self):
        return asdict(self)
    
# --- AGREGACIÓ DEL CAS (C) ---
@dataclass
class CasMenu:
    id_cas: int
    problema: DescripcioProblema 
    solucio: SolucioMenu
    avaluacio: AvaluacioCas
    
    def to_dict(self):
        # Helper per convertir tot l'arbre a dict
        return {
            "id_cas": self.id_cas,
            "problema": self.problema.to_dict(),
            "solucio": self.solucio.to_dict(),
            "avaluacio": self.avaluacio.to_dict()
        }