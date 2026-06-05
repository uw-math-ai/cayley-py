from __future__ import annotations

import sys
from pathlib import Path
import unittest


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.generators import build_generators, list_families
from app.models import GraphSpec


class GeneratorTests(unittest.TestCase):
    def test_catalog_contains_requested_families(self) -> None:
        ids = {family["id"] for family in list_families()}
        self.assertEqual(
            {
                "adjacent_transpositions",
                "all_transpositions",
                "star_transpositions",
                "adjacent_cycles",
                "consecutive_k_cycles",
                "wrapped_k_cycles",
                "pancake",
                "koltsov3",
                "cubic_pancake",
                "lrx",
            },
            ids,
        )

    def test_koltsov3_type1_expected_generators(self) -> None:
        gens = build_generators(GraphSpec("koltsov3", 5, {"permType": 1, "k": 0, "d": 2}))
        by_id = {gen.id: gen.permutation for gen in gens}
        self.assertEqual((1, 0, 3, 2, 4), by_id["I"])
        self.assertEqual((0, 2, 1, 4, 3), by_id["K"])
        self.assertEqual((2, 1, 0, 3, 4), by_id["S"])

    def test_koltsov3_type2_expected_s_generator(self) -> None:
        gens = build_generators(GraphSpec("koltsov3", 5, {"permType": 2, "k": 0}))
        by_id = {gen.id: gen.permutation for gen in gens}
        self.assertEqual((3, 2, 1, 0, 4), by_id["S"])

    def test_cubic_pancake_expected_generators(self) -> None:
        gens = build_generators(GraphSpec("cubic_pancake", 5, {"fixed": 2, "k": 3}))
        by_id = {gen.id: gen.permutation for gen in gens}
        self.assertEqual((4, 3, 2, 1, 0), by_id["R5"])
        self.assertEqual((1, 0, 2, 3, 4), by_id["R2"])
        self.assertEqual((2, 1, 0, 3, 4), by_id["R3"])

    def test_lrx_expected_generators(self) -> None:
        gens = build_generators(GraphSpec("lrx", 5))
        by_id = {gen.id: gen.permutation for gen in gens}
        self.assertEqual((1, 2, 3, 4, 0), by_id["L"])
        self.assertEqual((4, 0, 1, 2, 3), by_id["R"])
        self.assertEqual((1, 0, 2, 3, 4), by_id["X"])

    def test_inverse_policy_adds_cycle_inverses(self) -> None:
        default_gens = build_generators(GraphSpec("adjacent_cycles", 5))
        listed_gens = build_generators(GraphSpec("adjacent_cycles", 5, inverse_policy="listed"))
        self.assertEqual(6, len(default_gens))
        self.assertEqual(3, len(listed_gens))
        self.assertTrue(any(gen.inverse_of for gen in default_gens))


if __name__ == "__main__":
    unittest.main()
