import numpy as np
import pytest
import json

from Michael.lrx_koltsov3_ppo import (
    ActorCritic,
    PPOConfig,
    anneal_ppo_learning_rates,
    apply_generator,
    beam_search_with_policy_prior,
    build_hard_state_benchmark,
    build_ppo_optimizer,
    compute_bellman_clipped_targets,
    compute_policy_imitation_targets,
    compute_gae,
    compute_potential_shaping_reward,
    extract_koltsov3_features_np,
    extract_koltsov3_features_torch,
    generate_fixed_depth_walk_states,
    generate_nonbacktracking_walk_dataset,
    get_conjectured_longest_element,
    get_koltsov3_generators,
    koltsov3_feature_dim,
    score_policy_guided_candidates,
    successor_values_to_soft_action_targets,
    torch,
    valid_koltsov3_actions,
)


def test_get_koltsov3_generators_n5_k0_matches_expected():
    gens = get_koltsov3_generators(n=5, k=0)
    expected_I = np.array([1, 0, 3, 2, 4])
    expected_K = np.array([0, 2, 1, 4, 3])
    expected_S = np.array([2, 1, 0, 3, 4])

    assert gens.shape == (3, 5)
    assert np.array_equal(gens[0], expected_I)
    assert np.array_equal(gens[1], expected_K)
    assert np.array_equal(gens[2], expected_S)


def test_apply_generator_reorders_state():
    state = np.array([0, 1, 2, 3, 4])
    generator = np.array([1, 0, 3, 2, 4])
    next_state = apply_generator(state, generator)
    assert np.array_equal(next_state, np.array([1, 0, 3, 2, 4]))


def test_compute_gae_terminal_steps_match_reward_minus_value():
    rewards = np.array([1.0, -0.5, 2.0], dtype=np.float32)
    values = np.array([0.2, 0.1, -0.3], dtype=np.float32)
    dones = np.array([True, True, True], dtype=bool)

    adv, returns = compute_gae(
        rewards=rewards,
        values=values,
        dones=dones,
        last_value=0.0,
        gamma=0.99,
        gae_lambda=0.95,
    )

    expected_adv = rewards - values
    assert np.allclose(adv, expected_adv)
    assert np.allclose(returns, rewards)


def test_ppo_config_defaults_scale_with_n():
    cfg = PPOConfig(n=21)
    assert cfg.max_episode_steps == 84
    assert cfg.max_scramble_steps == 63


def test_ppo_config_rejects_non_divisible_rollout_minibatch():
    with pytest.raises(ValueError):
        PPOConfig(rollout_steps=1000, minibatch_size=256)


def test_ppo_config_rejects_non_positive_target_kl():
    with pytest.raises(ValueError):
        PPOConfig(target_kl=0.0)


def test_potential_shaping_reward_uses_potential_drop_and_bonus():
    reward = compute_potential_shaping_reward(
        prev_potential=7.0,
        next_potential=5.5,
        step_penalty=1.0,
        success_bonus=5.0,
        done=False,
    )
    terminal_reward = compute_potential_shaping_reward(
        prev_potential=3.0,
        next_potential=0.0,
        step_penalty=1.0,
        success_bonus=5.0,
        done=True,
    )

    assert reward == pytest.approx(0.5)
    assert terminal_reward == pytest.approx(7.0)


def test_feature_encoder_identity_state_has_expected_shape_and_blocks():
    identity = np.arange(5, dtype=np.int64)

    features = extract_koltsov3_features_np(identity, n=5, k=0)

    assert features.shape == (1, koltsov3_feature_dim(5))
    assert np.allclose(features[0, :5], 0.0)
    assert np.array_equal(features[0, -5:], identity.astype(np.float32))
    assert features[0, 5] == 0.0


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_numpy_and_torch_feature_encoders_match():
    states = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
            [2, 1, 0, 3, 4],
        ],
        dtype=np.int64,
    )

    np_features = extract_koltsov3_features_np(states, n=5, k=0)
    torch_features = extract_koltsov3_features_torch(
        torch.tensor(states, dtype=torch.long),
        n=5,
        k=0,
    ).cpu().numpy()

    assert np_features.shape == torch_features.shape
    assert np.allclose(np_features, torch_features)


def test_nonbacktracking_walk_dataset_includes_identity_and_step_bounds():
    rng = np.random.default_rng(0)
    generators = get_koltsov3_generators(n=5, k=0)

    states, steps = generate_nonbacktracking_walk_dataset(
        generators,
        num_walks=4,
        walk_length=6,
        rng=rng,
        history_size=4,
    )

    assert states.shape[1] == 5
    assert steps.shape[0] == states.shape[0]
    assert np.array_equal(states[0], np.arange(5, dtype=np.int64))
    assert steps[0] == 0.0
    assert np.all((steps >= 0.0) & (steps <= 6.0))


def test_fixed_depth_walk_states_return_requested_count_and_depth():
    generators = get_koltsov3_generators(n=5, k=0)

    states, witness_lengths = generate_fixed_depth_walk_states(
        generators,
        num_states=3,
        walk_length=8,
        seed=7,
        history_size=4,
    )

    assert states.shape == (3, 5)
    assert np.array_equal(witness_lengths, np.full(3, 8.0, dtype=np.float32))
    assert len({tuple(state.tolist()) for state in states}) == 3


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_bellman_targets_clip_to_walk_depth():
    class ConstantValueModel:
        def __init__(self):
            self.training = False

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, obs):
            batch = obs.shape[0]
            logits = torch.zeros((batch, 3), dtype=torch.float32, device=obs.device)
            values = torch.full((batch,), 10.0, dtype=torch.float32, device=obs.device)
            return logits, values

    generators = get_koltsov3_generators(n=5, k=0)
    states = np.array(
        [
            [0, 1, 2, 3, 4],
            [1, 0, 3, 2, 4],
        ],
        dtype=np.int64,
    )
    upper_bounds = np.array([0.0, 1.0], dtype=np.float32)

    targets = compute_bellman_clipped_targets(
        ConstantValueModel(),
        states,
        generators,
        upper_bounds,
        device="cpu",
        batch_size=2,
    )

    assert np.array_equal(targets, np.array([0.0, 1.0], dtype=np.float32))


def test_soft_action_targets_split_ties_uniformly():
    successor_values = np.array(
        [
            [1.0, 2.0, 1.0],
            [3.0, 2.0, 0.0],
            [5.0, 5.0, 5.0],
        ],
        dtype=np.float32,
    )

    targets = successor_values_to_soft_action_targets(successor_values)

    assert np.allclose(targets[0], np.array([0.5, 0.0, 0.5], dtype=np.float32))
    assert np.allclose(targets[1], np.array([0.0, 0.0, 1.0], dtype=np.float32))
    assert np.allclose(targets[2], np.full(3, 1.0 / 3.0, dtype=np.float32))


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_policy_imitation_targets_prefer_lowest_value_successor():
    class ActionBiasedValueModel:
        def __init__(self):
            self.training = False

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, obs):
            batch = obs.shape[0]
            logits = torch.zeros((batch, 3), dtype=torch.float32, device=obs.device)
            values = obs[:, 0].to(torch.float32)
            return logits, values

    generators = get_koltsov3_generators(n=5, k=0)
    states = np.array([[0, 1, 2, 3, 4]], dtype=np.int64)

    targets = compute_policy_imitation_targets(
        ActionBiasedValueModel(),
        states,
        generators,
        device="cpu",
        batch_size=1,
    )

    assert np.allclose(targets, np.array([[0.0, 1.0, 0.0]], dtype=np.float32))


def test_valid_koltsov3_actions_respects_x_trick_gate():
    assert valid_koltsov3_actions(np.array([0, 1, 2, 3, 4]), apply_x_trick=True) == (1, 2)
    assert valid_koltsov3_actions(np.array([1, 0, 2, 3, 4]), apply_x_trick=True) == (0, 1, 2)


def test_conjectured_longest_element_matches_paper_pattern():
    assert np.array_equal(
        get_conjectured_longest_element(5),
        np.array([1, 0, 4, 3, 2], dtype=np.int64),
    )


def test_hard_state_benchmark_falls_back_when_bfs_json_missing(tmp_path):
    states, reference_lengths, source, reference_kind = build_hard_state_benchmark(
        5,
        benchmark_dir=tmp_path,
        bfs_results_dir=tmp_path,
    )

    assert source == "conjectured_longest_constructive"
    assert reference_kind == "constructive_length"
    assert len(states) == 1
    assert np.array_equal(states[0], np.array([1, 0, 4, 3, 2], dtype=np.int64))
    assert reference_lengths[tuple(states[0].tolist())] == 10


def test_hard_state_benchmark_prefers_benchmark_file(tmp_path):
    benchmark_path = tmp_path / 'koltsov3_n05_hard_states.json'
    payload = {
        'source': 'fixed_walk_test',
        'reference_kind': 'witness_length',
        'states': [[1, 0, 3, 2, 4], [2, 1, 0, 3, 4]],
        'reference_lengths': [8, 8],
    }
    benchmark_path.write_text(json.dumps(payload))

    states, reference_lengths, source, reference_kind = build_hard_state_benchmark(
        5,
        benchmark_dir=tmp_path,
        bfs_results_dir=tmp_path,
    )

    assert source == 'fixed_walk_test'
    assert reference_kind == 'witness_length'
    assert len(states) == 2
    assert reference_lengths[tuple(states[0].tolist())] == 8.0


def test_policy_guided_candidate_scores_include_logprob_bonus():
    scores = score_policy_guided_candidates(
        np.array([5.0, 5.0], dtype=np.float32),
        np.log(np.array([0.1, 0.9], dtype=np.float32)),
        policy_alpha=1.0,
    )

    assert scores[1] < scores[0]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_ppo_optimizer_splits_policy_and_shared_value_learning_rates():
    cfg = PPOConfig(policy_lr=3e-4, value_lr=1e-3)
    model = ActorCritic(n=5, hidden_dim=32, k=0)

    optimizer = build_ppo_optimizer(model, cfg)

    assert len(optimizer.param_groups) == 2
    assert sorted(group["lr"] for group in optimizer.param_groups) == [3e-4, 1e-3]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_annealed_ppo_learning_rates_scale_both_param_groups():
    cfg = PPOConfig(policy_lr=3e-4, value_lr=1e-3, anneal_learning_rate=True)
    model = ActorCritic(n=5, hidden_dim=32, k=0)

    optimizer = build_ppo_optimizer(model, cfg)
    anneal_ppo_learning_rates(optimizer, cfg, update_idx=3, total_updates=4)

    assert sorted(group["lr"] for group in optimizer.param_groups) == [1.5e-4, 5e-4]


@pytest.mark.skipif(torch is None, reason="PyTorch is not installed")
def test_policy_guided_beam_search_finds_one_step_solution():
    class DistanceModel:
        def __init__(self, n):
            self.training = False
            self.identity = torch.arange(n, dtype=torch.long)

        def eval(self):
            self.training = False

        def train(self, mode=True):
            self.training = mode

        def __call__(self, obs):
            logits = torch.zeros((obs.shape[0], 3), dtype=torch.float32, device=obs.device)
            values = (obs != self.identity.to(obs.device)).sum(dim=1).to(torch.float32)
            return logits, values

    generators = get_koltsov3_generators(n=5, k=0)
    result = beam_search_with_policy_prior(
        np.array([1, 0, 3, 2, 4], dtype=np.int64),
        DistanceModel(5),
        generators,
        beam_width=1,
        step_limit=2,
        policy_alpha=0.0,
        device="cpu",
    )

    assert result.path_found is True
    assert result.path == [0]
    assert result.path_length == 1
