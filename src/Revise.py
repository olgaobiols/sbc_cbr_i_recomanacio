from typing import Any, Dict, List, Optional

"""
GESTOR DE LA FASE REVISE (Avaluació)
------------------------------------
Controlador encarregat de la interacció amb l'usuari per recollir feedback.
Classifica l'èxit de la proposta en tres categories:
1. SUCCESS: Cas vàlid per ser après sense canvis.
2. SOFT_FAILURE: El cas requereix ajustos menors o té preferències negatives.
3. CRITICAL_FAILURE: El cas viola restriccions de seguretat o salut.
"""

class GestorRevise:
    def __init__(self, mem_personal: Any, mem_global: Any) -> None:
        self.mem_personal = mem_personal
        self.mem_global = mem_global
        self._ui_width = 80

    def avaluar_proposta(self, cas_proposat: Dict, user_id: str = "guest") -> Dict[str, Any]:
        """Entry point: Coordina la recollida de dades i l'etiquetatge del cas."""
        feedback = self.collect_feedback(cas_proposat, str(user_id))
        
        status = self.evaluate_result(
            puntuacio_global=feedback["puntuacio_global"],
            n2_taste=feedback["aspectes"]["gust"],
            n2_originality=feedback["aspectes"]["originalitat"],
            rejected_ingredients=feedback["ingredients_rebutjats"],
            rejected_pairs=feedback["parelles_rebutjades"],
            rejected_health=feedback["rebuigs_critics"],
            rejected_taste=feedback["rebuigs_suaus"]
        )
        
        feedback["tipus_resultat"] = status
        return feedback

    def collect_feedback(self, case: Dict, user_id: str) -> Dict[str, Any]:
        """Interfície de terminal per recollir notes i rebuigs específics."""
        self._section("AVALUACIÓ FINAL")
        self._print_star_scale()
        
        # 1. Recollida de puntuacions
        n1 = self.input_nota(self._prompt("\nNota global del menú (1-5):"))
        print("\nPuntua els detalls:")
        n2_taste = self.input_nota(self._prompt("Sabor (1-5):"))
        n2_originality = self.input_nota(self._prompt("Originalitat (1-5):"))

        # 2. Recollida de rebuigs granulars
        res = {"ing": [], "pair": [], "health": [], "taste": []}
        self._block("INGREDIENTS O COMBINACIONS A EVITAR")
        print("Exemples: 'api', 'maduixa+all' o 'NO ceba'. ('FI' per acabar)")

        while True:
            cmd = input("> ").strip()
            if not cmd or cmd.upper() == "FI": break
            
            target = cmd[3:].strip() if cmd.upper().startswith("NO ") else cmd
            target = target.lower()

            if "+" in target or "|" in target:
                pair = self._split_pair(target)
                if pair:
                    ing_a, ing_b = pair
                    norm_pair = "|".join(sorted([ing_a, ing_b]))
                    self.mem_personal.registrar_rebuig_parella(user_id, ing_a, ing_b)
                    self.mem_global.acumular_evidencia_parella(ing_a, ing_b)
                    res["pair"].append(norm_pair)
                    self._atribuir_motiu(norm_pair, res)
            else:
                self.mem_personal.registrar_rebuig_ingredient(user_id, target)
                self.mem_global.acumular_evidencia_ingredient(target)
                res["ing"].append(target)
                self._atribuir_motiu(target, res)

        return {
            "puntuacio_global": n1,
            "aspectes": {"gust": n2_taste, "originalitat": n2_originality},
            "ingredients_rebutjats": res["ing"],
            "parelles_rebutjades": res["pair"],
            "rebuigs_critics": res["health"],
            "rebuigs_suaus": res["taste"]
        }

    def evaluate_result(self, puntuacio_global: int, **kwargs) -> str:
        """Lògica de classificació segons el feedback rebut."""
        self._block("RESULTAT DE L'AVALUACIÓ")
        
        # Fracàs Crític: Salut o nota ínfima
        if puntuacio_global <= 2 or kwargs.get("rejected_health"):
            print("Estat: CRITICAL_FAILURE (Violació de restriccions o baixa qualitat)")
            return "CRITICAL_FAILURE"
        
        # Fracàs Suau: Nota mitjana o rebuigs per gust
        if puntuacio_global == 3 or kwargs.get("rejected_ingredients") or kwargs.get("rejected_pairs"):
            print("Estat: SOFT_FAILURE (Requereix ajustos o preferències no òptimes)")
            return "SOFT_FAILURE"
        
        # Èxit: Nota alta i sense incidències
        if puntuacio_global >= 4:
            print("Estat: SUCCESS (Proposta validada)")
            return "SUCCESS"

        return "SOFT_FAILURE"

    # --- Helpers d'Interfície i Lògica ---
    def _split_pair(self, raw: str) -> Optional[tuple]:
        parts = [p.strip() for p in raw.replace("+", "|").split("|") if p.strip()]
        return (parts[0], parts[1]) if len(parts) == 2 else None

    def _atribuir_motiu(self, target: str, res: dict):
        motiu = input(f"   Rebuig de '{target}' per Salut (C) o Gust (S)? [C/S]: ").lower()
        key = "health" if motiu == "c" else "taste"
        res[key].append(target)

    def input_nota(self, prompt: str) -> int:
        while True:
            try:
                v = int(input(prompt))
                if 1 <= v <= 5: return v
            except: pass
            print("  [!] Introdueix un número de l'1 al 5.")

    def _section(self, t: str): print(f"\n{'='*self._ui_width}\n{t}\n{'='*self._ui_width}")
    def _block(self, t: str): print(f"\n{'-'*self._ui_width}\n{t}\n{'-'*self._ui_width}")
    def _prompt(self, label: str): return f"> {label:<28}"
    
    def _print_star_scale(self):
        for i in range(1, 6): print(f"  {i}: {'★'*i}{'☆'*(5-i)}")