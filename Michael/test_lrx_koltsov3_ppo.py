import numpy as np

from Michael.lrx_koltsov3_ppo import (
    PPOConfig,
    apply_generator,
    compute_gae,
    get_koltsov3_generators,
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
    dones = np.array([True, True, True], dtype=np.bool_)

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
