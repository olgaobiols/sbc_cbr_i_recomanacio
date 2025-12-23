from typing import Any, Dict, List, Optional


class GestorRevise:
    """
    Controlador de la fase REVISE amb doble memòria (personal + global).
    La memòria es gestiona via dos objectes passats per constructor.
    """
    def __init__(self, mem_personal: Any, mem_global: Any) -> None:
        self.mem_personal = mem_personal
        self.mem_global = mem_global
        self._ui_width = 80

    def _format_stars(self, n: int) -> str:
        return "★" * n + "☆" * (5 - n)

    def _line(self, ch: str = "=") -> str:
        return ch * self._ui_width

    def _section(self, title: str, ch: str = "=") -> None:
        line = self._line(ch)
        print("\n" + line)
        print(title)
        print(line)

    def _block(self, title: str, ch: str = "-") -> None:
        line = self._line(ch)
        print("\n" + line)
        print(title)
        print(line)

    def _prompt(self, label: str) -> str:
        return f"> {label:<28}"

    def _print_star_scale(self) -> None:
        print("Escala de valoració:")
        for i in range(1, 6):
            print(f"  {i}: {self._format_stars(i)}")

    def input_nota(self, prompt: str) -> int:
        while True:
            try:
                val = int(input(prompt))
                if 1 <= val <= 5:
                    return val
            except Exception:
                pass
            print("  Si us plau, introdueix un número de 1 a 5.")

    def _normalize_rejection(self, raw: str) -> str:
        return str(raw).strip().lower()

    def _is_pair(self, raw: str) -> bool:
        return "+" in raw or "|" in raw

    def _split_pair(self, raw: str) -> Optional[tuple[str, str]]:
        if "+" in raw:
            parts = [p.strip() for p in raw.split("+") if p.strip()]
        elif "|" in raw:
            parts = [p.strip() for p in raw.split("|") if p.strip()]
        else:
            return None
        if len(parts) != 2:
            return None
        return parts[0], parts[1]

    def _prompt_rejection_motive(self) -> str:
        motiu = input("   El rebuig és per salut/al·lèrgia (C) o per gust/preferència (S)? [C/S]: ").strip().lower()
        return "critical" if motiu == "c" else "soft"

    def collect_feedback(self, case: Dict, user_id: str) -> Dict[str, Any]:
        self._section("AVALUACIÓ FINAL")
        self._print_star_scale()
        n1 = self.input_nota(self._prompt("\nNota global del menú (1-5):"))

        print("\nSi us plau puntúa els següents detalls:")
        n2_taste = self.input_nota(self._prompt("Sabor (1-5):"))
        n2_originality = self.input_nota(self._prompt("Originalitat (1-5):"))

        rejected_ingredients: List[str] = []
        rejected_pairs: List[str] = []
        rejected_health: List[str] = []
        rejected_taste: List[str] = []

        self._block("INGREDIENTS O COMBINACIONS A EVITAR")
        print("Escriu 'NO ingredient' (ex: 'NO api') o 'NO A+B' (ex: 'NO maduixa+all').")
        print("Escriu 'FI' per acabar.")

        while True:
            cmd = input("> ").strip()
            if cmd == "" or cmd.upper() == "FI":
                break
            if cmd.upper().startswith("NO "):
                target = cmd[3:].strip()
            else:
                target = cmd

            target = self._normalize_rejection(target)
            if self._is_pair(target):
                pair = self._split_pair(target)
                if not pair:
                    print("  Format de parella invàlid. Usa 'A+B'.")
                    continue
                ing_a, ing_b = pair
                self.mem_personal.registrar_rebuig_parella(user_id, ing_a, ing_b)
                self.mem_global.acumular_evidencia_parella(ing_a, ing_b)
                normalized = "|".join(sorted([ing_a, ing_b]))
                rejected_pairs.append(normalized)
                motiu = self._prompt_rejection_motive()
                if motiu == "critical":
                    rejected_health.append(normalized)
                    print("   Entesos. Rebuig per salut/al·lèrgia registrat.")
                else:
                    rejected_taste.append(normalized)
                    print("   Entesos. Rebuig per gust/preferència registrat.")
                print(f"   He enregistrat la parella '{normalized}'.")
            else:
                self.mem_personal.registrar_rebuig_ingredient(user_id, target)
                self.mem_global.acumular_evidencia_ingredient(target)
                rejected_ingredients.append(target)
                motiu = self._prompt_rejection_motive()
                if motiu == "critical":
                    rejected_health.append(target)
                    print("   Entesos. Rebuig per salut/al·lèrgia registrat.")
                else:
                    rejected_taste.append(target)
                    print("   Entesos. Rebuig per gust/preferència registrat.")
                print(f"   He enregistrat l'ingredient '{target}'.")

        return {
            "puntuacio_global": n1,
            "aspectes": {"gust": n2_taste, "originalitat": n2_originality},
            "ingredients_rebutjats": rejected_ingredients,
            "parelles_rebutjades": rejected_pairs,
            "rebuigs_critics": rejected_health,
            "rebuigs_suaus": rejected_taste,
        }

    def _print_result_header(self) -> None:
        line = self._line("-")
        print("\n" + line)
        print("RESULTAT DE L'AVALUACIÓ")
        print(line)

    def evaluate_result(
        self,
        puntuacio_global: int,
        n2_taste: Optional[int],
        n2_originality: Optional[int],
        rejected_ingredients: List[str],
        rejected_pairs: List[str],
        rejected_health: List[str],
        rejected_taste: List[str],
    ) -> str:
        self._print_result_header()
        if puntuacio_global <= 2:
            print("Resultat: puntuació global baixa.")
            return "CRITICAL_FAILURE"
        if rejected_health:
            print("Resultat: s'ha detectat un rebuig per salut/al·lèrgia.")
            return "CRITICAL_FAILURE"
        if puntuacio_global == 3:
            print("Resultat: puntuació moderada.")
            return "SOFT_FAILURE"
        if rejected_ingredients or rejected_pairs:
            if rejected_taste:
                print("Resultat: s'ha detectat un rebuig per gust/preferència.")
            else:
                print("Resultat: hi ha rebuigs granulars pendents.")
            return "SOFT_FAILURE"

        if puntuacio_global >= 4:
            print("Resultat: valoració alta i sense incidències.")
            return "SUCCESS"

        print("Resultat: cas per defecte.")
        return "SOFT_FAILURE"

    def avaluar_proposta(self, cas_proposat: Dict, user_id: str = "guest") -> Dict[str, Any]:
        feedback = self.collect_feedback(cas_proposat, str(user_id))
        status = self.evaluate_result(
            feedback["puntuacio_global"],
            feedback["aspectes"]["gust"],
            feedback["aspectes"]["originalitat"],
            feedback["ingredients_rebutjats"],
            feedback["parelles_rebutjades"],
            feedback["rebuigs_critics"],
            feedback["rebuigs_suaus"],
        )
        feedback["tipus_resultat"] = status
        return feedback
