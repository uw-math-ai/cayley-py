from __future__ import annotations

import sys
from pathlib import Path
import unittest


BACKEND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND))

from app.bfs import CACHE_ROOT, ensure_exact_bfs
from app.challenge import forfeit_challenge, move_challenge, start_challenge
from app.generators import build_generators
from app.graph_view import graph_summary, graph_view, shortest_paths_to_state
from app.models import GraphViewRequest
from app.models import GraphSpec
from app.permutations import apply_generator, identity, initial_state


class BfsChallengeTests(unittest.TestCase):
    def test_adjacent_transposition_bfs_certifies_reversed_path(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 3)
        bfs = ensure_exact_bfs(spec, cap=1000, use_cache=False)
        target = (2, 1, 0)
        path = bfs.path_to(target)
        self.assertEqual(3, bfs.diameter)
        self.assertEqual(3, path.length)
        self.assertEqual(identity(3), path.states[0])
        self.assertEqual(target, path.states[-1])

        state = identity(3)
        gens = {gen.id: gen for gen in build_generators(spec)}
        for gen_id in path.generator_ids:
            state = apply_generator(state, gens[gen_id].permutation)
        self.assertEqual(target, state)

    def test_spec_hash_separates_inverse_policy(self) -> None:
        default_spec = GraphSpec("adjacent_cycles", 4)
        listed_spec = GraphSpec("adjacent_cycles", 4, inverse_policy="listed")
        self.assertNotEqual(default_spec.hash(), listed_spec.hash())
        self.assertGreater(len(build_generators(default_spec)), len(build_generators(listed_spec)))

    def test_corrupt_bfs_cache_is_recomputed(self) -> None:
        spec = GraphSpec("star_transpositions", 3, {"cacheTest": "corrupt"})
        CACHE_ROOT.mkdir(parents=True, exist_ok=True)
        path = CACHE_ROOT / f"{spec.hash()}.npz"
        path.write_text("not a zip", encoding="utf-8")
        result = ensure_exact_bfs(spec, cap=100)
        self.assertEqual(6, result.n_states)
        self.assertGreater(path.stat().st_size, len("not a zip"))

    def test_challenge_completes_with_optimal_path(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4)
        session = start_challenge(spec, custom_distance=1, seed=1, exact_cap=1000)
        self.assertEqual("active", session.status)
        self.assertEqual(1, session.certified_path.length)
        moved = move_challenge(session.session_id, session.certified_path.generator_ids[0])
        self.assertEqual("completed", moved.status)
        self.assertEqual(0, moved.to_dict()["excess"])
        self.assertEqual(2, len(moved.to_dict()["userStates"]))
        self.assertEqual(moved.to_dict()["current"], moved.to_dict()["userStates"][-1])

    def test_challenge_tracks_non_optimal_completion(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4)
        session = start_challenge(spec, custom_distance=1, seed=2, exact_cap=1000)
        target_move = session.certified_path.generator_ids[0]
        other_move = next(gen.id for gen in build_generators(spec) if gen.id != target_move)
        move_challenge(session.session_id, other_move)
        move_challenge(session.session_id, other_move)
        done = move_challenge(session.session_id, target_move)
        self.assertEqual("completed", done.status)
        self.assertEqual(3, done.to_dict()["userLength"])
        self.assertEqual(4, len(done.to_dict()["userStates"]))
        self.assertEqual(1, done.to_dict()["optimalLength"])
        self.assertEqual(2, done.to_dict()["excess"])

    def test_challenge_forfeit_reveals_certified_path(self) -> None:
        spec = GraphSpec("lrx", 4)
        session = start_challenge(spec, difficulty="hard", seed=3, exact_cap=1000)
        forfeited = forfeit_challenge(session.session_id)
        self.assertEqual("forfeited", forfeited.status)
        self.assertIsNotNone(forfeited.certified_path)
        self.assertTrue(forfeited.certified_path.certified)

    def test_k_different_schreier_bfs_uses_coset_start(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4, {"stateSpace": "k_different", "differentK": 2})
        self.assertEqual((0, 1, 1, 1), initial_state(spec))
        bfs = ensure_exact_bfs(spec, cap=100, use_cache=False)
        self.assertEqual(4, bfs.n_states)
        self.assertEqual(initial_state(spec), bfs.start_state)
        self.assertEqual([0, 1, 1, 1], bfs.metadata()["startState"])
        session_dict = start_challenge(spec, custom_distance=0, exact_cap=100).to_dict()
        self.assertEqual([0, 1, 1, 1], session_dict["current"])
        self.assertEqual([0, 1, 1, 1], session_dict["start"])
        self.assertEqual([[0, 1, 1, 1]], session_dict["userStates"])

    def test_summary_and_bruhat_view_support_schreier_graphs(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 5, {"stateSpace": "k_different", "differentK": 3})
        summary = graph_summary(spec)
        self.assertEqual("k_different", summary["stateSpace"])
        self.assertEqual([0, 1, 2, 2, 2], summary["startState"])
        self.assertEqual(20, summary["estimatedGroupUpperBound"])
        view = graph_view(GraphViewRequest(spec=spec, mode="full", layout="bruhat", exact_cap=1000))
        self.assertEqual("bruhat", view["layout"])
        self.assertTrue(view["edges"])
        self.assertTrue(all(-1.0 <= node["x"] <= 1.0 and -1.0 <= node["y"] <= 1.0 for node in view["nodes"]))

    def test_new_layout_modes_return_bounded_coordinates(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4, {"stateSpace": "k_different", "differentK": 3})
        for layout in ("spectral", "lehmer", "coset", "target-distance"):
            view = graph_view(
                GraphViewRequest(
                    spec=spec,
                    mode="full",
                    layout=layout,
                    target_state=(2, 2, 0, 1),
                    exact_cap=1000,
                )
            )
            self.assertEqual(layout, view["layout"])
            self.assertTrue(view["nodes"])
            self.assertTrue(all(-1.0 <= node["x"] <= 1.0 and -1.0 <= node["y"] <= 1.0 for node in view["nodes"]))

    def test_shortest_paths_endpoint_payload_contains_all_reduced_edges(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 3)
        payload = shortest_paths_to_state(spec, (2, 1, 0), exact_cap=1000)
        self.assertTrue(payload["certified"])
        self.assertEqual(3, payload["length"])
        self.assertEqual(2, payload["pathCount"])
        self.assertEqual([0, 1, 2], payload["start"])
        self.assertEqual([2, 1, 0], payload["target"])
        self.assertIn("0,1,2", payload["nodeIds"])
        self.assertIn("2,1,0", payload["nodeIds"])
        self.assertTrue(any(edge["generatorId"] == "s1" for edge in payload["edges"]))
        self.assertTrue(any(edge["generatorId"] == "s2" for edge in payload["edges"]))

    def test_challenge_returns_session_generators_for_selected_family(self) -> None:
        spec = GraphSpec("lrx", 4)
        session = start_challenge(spec, custom_distance=1, seed=4, exact_cap=1000)
        self.assertEqual(["L", "R", "X"], [gen["id"] for gen in session.to_dict()["generators"]])

    def test_local_challenge_view_pins_certified_path_states(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4)
        session = start_challenge(spec, custom_distance=3, seed=5, exact_cap=1000)
        view = graph_view(
            GraphViewRequest(
                spec=spec,
                mode="local",
                layout="target-distance",
                focus_state=session.current,
                target_state=session.target,
                pinned_states=session.certified_path.states if session.certified_path else (),
                radius=1,
                exact_cap=1000,
            )
        )
        node_ids = {node["id"] for node in view["nodes"]}
        edge_keys = {f"{edge['source']}|{edge['target']}|{edge['generatorId']}" for edge in view["edges"]}
        self.assertIn(",".join(str(x) for x in session.target), node_ids)
        self.assertTrue(session.certified_path)
        for state in session.certified_path.states:
            self.assertIn(",".join(str(x) for x in state), node_ids)
        for source, target, gen_id in zip(
            session.certified_path.states,
            session.certified_path.states[1:],
            session.certified_path.generator_ids,
        ):
            self.assertIn(
                f"{','.join(str(x) for x in source)}|{','.join(str(x) for x in target)}|{gen_id}",
                edge_keys,
            )

    def test_exact_challenge_view_keeps_global_distance_frame_and_move_edges(self) -> None:
        spec = GraphSpec("adjacent_transpositions", 4)
        session = start_challenge(spec, custom_distance=3, seed=6, exact_cap=1000)
        self.assertTrue(session.certified_path)
        moved = move_challenge(session.session_id, session.certified_path.generator_ids[0])
        generators = build_generators(spec)
        move_targets = tuple(apply_generator(moved.current, gen.permutation) for gen in generators)
        pinned = tuple([moved.current, *move_targets, *moved.certified_path.states])
        view = graph_view(
            GraphViewRequest(
                spec=spec,
                mode="layers",
                layout="layers",
                target_state=moved.target,
                pinned_states=pinned,
                full_node_cap=3,
                exact_cap=1000,
            )
        )
        node_by_id = {node["id"]: node for node in view["nodes"]}
        edge_keys = {f"{edge['source']}|{edge['target']}|{edge['generatorId']}" for edge in view["edges"]}
        current_id = ",".join(str(x) for x in moved.current)
        self.assertEqual(1, node_by_id[current_id]["distance"])
        for gen, target in zip(generators, move_targets):
            target_id = ",".join(str(x) for x in target)
            self.assertIn(target_id, node_by_id)
            self.assertIn(f"{current_id}|{target_id}|{gen.id}", edge_keys)


if __name__ == "__main__":
    unittest.main()
