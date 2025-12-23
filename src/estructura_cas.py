from dataclasses import dataclass, asdict, field
from typing import List, Dict, Set, Optional, Any

"""
MODELS DE DADES DEL SISTEMA CBR
------------------------------
Defineix l'ontologia del sistema segons el paradigma de Raonament Basat en Casos.
Organitzat en tres espais:
1. Problema (P): Atributs de la demanda de l'usuari.
2. Solució (S): Composició del menú i traçabilitat (XCBR).
3. Avaluació (E): Feedback i mètriques d'aprenentatge per a la fase Retain.
"""

@dataclass 
class Plat: 
    nom: str
    ingredients: List[str]            # Normalitzats segons ontologia FlavorGraph
    curs: str                        # 'primer', 'segon', 'postres'
    estil_tags: List[str] = field(default_factory=list) 
    rols_ingredients: List[str] = field(default_factory=list) # 'main', 'base', 'seasoning'
    tecniques: List[str] = field(default_factory=list)        # Justificació de l'adaptació
    preu: float = 0.0             

    def to_dict(self): return asdict(self)

@dataclass
class Beguda:
    nom: str
    categoria: str                    # 'vi_blanc', 'aigua', 'refresc'
    maridatge_amb: str = "general"    # 'primer', 'segon' o 'general'

# --- 1. ESPAI DEL PROBLEMA (Inputs del Retrieve) ---
@dataclass            
class DescripcioProblema:
    tipus_esdeveniment: str           # casament, congres, aniversari...
    n_comensals: int
    preu_pers_objectiu: float
    temporada: str                    # primavera, estiu, tardor, hivern
    servei: str                       # assegut, cocktail, buffet
    alcohol: str                      # 'si' o 'no'
    estil_culinari: str = ""          # Objectiu d'adaptació (ex: 'japonès')
    restriccions: Set[str] = field(default_factory=set) # {'celiac', 'vegan'}
    formalitat: str = "indiferent"

    def to_dict(self):
        """Serialització segura per a JSON (converteix sets a lists)."""
        d = asdict(self)
        d['restriccions'] = list(self.restriccions)
        return d

# --- 2. ESPAI DE LA SOLUCIÓ (Outputs de l'Adaptation) ---
@dataclass
class SolucioMenu: 
    primer_plat: Plat
    segon_plat: Plat
    postres: Plat
    begudes: List[Beguda] = field(default_factory=list)
    preu_total_real: float = 0.0
    descripcio_final: str = ""        # Explicació generada del menú
    logs_transformacio: List[str] = field(default_factory=list) # Traçabilitat XCBR
    
    def to_dict(self): return asdict(self)
    
# --- 3. ESPAI D'AVALUACIÓ (Inputs del Revise/Retain) ---
@dataclass
class AvaluacioCas:
    derivacio: str = "original"       # ID del cas pare o 'generat'
    puntuacio_global: int = 0         # Escala 1-5
    feedback_textual: str = "" 
    ingredients_rebutjats: List[str] = field(default_factory=list) 
    cost_adaptacio: int = 0           # Quantificat per k_adapt
    utilitat: float = 0.0             # Valor calculat per a la retenció
    tipus_resultat: str = "pendent"   # 'SUCCESS', 'SOFT_FAILURE', 'CRITICAL_FAILURE'
    validat: bool = False
    
    def to_dict(self): return asdict(self)
    
# --- AGREGACIÓ COMPLETA DEL CAS ---
@dataclass
class CasMenu:
    id_cas: int
    problema: DescripcioProblema 
    solucio: SolucioMenu
    avaluacio: AvaluacioCas
    
    def to_dict(self):
        """Genera un diccionari niat de tot l'arbre del cas."""
        return {
            "id_cas": self.id_cas,
            "problema": self.problema.to_dict(),
            "solucio": self.solucio.to_dict(),
            "avaluacio": self.avaluacio.to_dict()
        }