from typing import Any, Dict, List, Optional


class GestorRevise:
    """
    Controlador de la fase REVISE amb doble memòria (personal + global).
    La memòria es gestiona via dos objectes passats per constructor.
    """
    def __init__(self, mem_personal: Any, mem_global: Any) -> None:
        self.mem_personal = mem_personal
        self.mem_global = mem_global

    def input_nota(self, prompt: str) -> int:
        while True:
            try:
                val = int(input(prompt))
                if 1 <= val <= 5:
                    return val
            except Exception:
                pass
            print("  Si us plau, introdueix un numero de 1 a 5.")

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
        motiu = input("   Rebuig per salut/al·lergia (C) o per gust/preferencia (S)? [C/S]: ").strip().lower()
        return "critical" if motiu == "c" else "soft"

    def collect_feedback(self, case: Dict, user_id: str) -> Dict[str, Any]:
        print("\n🧐 --- FASE REVISE: AVALUACIO ---")
        print("   [Revise] Arquitectura dual: Canal A (personal) + Canal B (global).")
        n1 = self.input_nota("Puntua el menu globalment (1-5): ")

        print("Detalla una mica mes (sempre obligatori):")
        n2_taste = self.input_nota("  Nota Gust (1-5): ")
        n2_originality = self.input_nota("  Nota Originalitat (1-5): ")

        rejected_ingredients: List[str] = []
        rejected_pairs: List[str] = []
        rejected_health: List[str] = []
        rejected_taste: List[str] = []

        print("\nHi ha algun ingredient o combinacio que vulguis vetar?")
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
                    print("  Format de parella invalid. Usa 'A+B'.")
                    continue
                ing_a, ing_b = pair
                self.mem_personal.registrar_rebuig_parella(user_id, ing_a, ing_b)
                self.mem_global.acumular_evidencia_parella(ing_a, ing_b)
                normalized = "|".join(sorted([ing_a, ing_b]))
                rejected_pairs.append(normalized)
                motiu = self._prompt_rejection_motive()
                if motiu == "critical":
                    rejected_health.append(normalized)
                    print("   [Revise] Rebuig CRITICAL (salut/al·lergia) registrat.")
                else:
                    rejected_taste.append(normalized)
                    print("   [Revise] Rebuig SOFT (gust/preferencia) registrat.")
                print(f"   [Revise] Rebuig granular: parella '{normalized}' registrada a Canal A + Canal B.")
            else:
                self.mem_personal.registrar_rebuig_ingredient(user_id, target)
                self.mem_global.acumular_evidencia_ingredient(target)
                rejected_ingredients.append(target)
                motiu = self._prompt_rejection_motive()
                if motiu == "critical":
                    rejected_health.append(target)
                    print("   [Revise] Rebuig CRITICAL (salut/al·lergia) registrat.")
                else:
                    rejected_taste.append(target)
                    print("   [Revise] Rebuig SOFT (gust/preferencia) registrat.")
                print(f"   [Revise] Rebuig granular: ingredient '{target}' registrat a Canal A + Canal B.")

        return {
            "puntuacio_global": n1,
            "aspectes": {"gust": n2_taste, "originalitat": n2_originality},
            "ingredients_rebutjats": rejected_ingredients,
            "parelles_rebutjades": rejected_pairs,
            "rebuigs_critics": rejected_health,
            "rebuigs_suaus": rejected_taste,
        }

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
        if puntuacio_global <= 2:
            print("   [Revise] CRITICAL_FAILURE: puntuacio global baixa (N1 <= 2).")
            return "CRITICAL_FAILURE"
        if rejected_health:
            print("   [Revise] CRITICAL_FAILURE: rebuig per salut/al·lergia (N3) detectat.")
            return "CRITICAL_FAILURE"
        if puntuacio_global == 3:
            print("   [Revise] SOFT_FAILURE: puntuacio global moderada (N1 = 3).")
            return "SOFT_FAILURE"
        if rejected_ingredients or rejected_pairs:
            if rejected_taste:
                print("   [Revise] SOFT_FAILURE: rebuig per gust/preferencia (N3) detectat.")
            else:
                print("   [Revise] SOFT_FAILURE: rebuig granular (N3) sense salut/al·lergia.")
            return "SOFT_FAILURE"

        if puntuacio_global >= 4:
            print("   [Revise] SUCCESS: puntuacio global alta i cap violacio dura.")
            return "SUCCESS"

        print("   [Revise] SOFT_FAILURE: cas per defecte.")
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
