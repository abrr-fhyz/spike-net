import argparse
import sys
import numpy as np

from Env.chemotax import ChemotaxisEnvironment, FoodSource
from NeuralNet.SNN import SNN
from Data.Connectome import buildNervousSystem, buildSynapticStructure

# CONFIG #
ARENA = "single"
STEPS = 10000
SEED = None
EPISODES = 5
QUIET = False

ARENA_MAP = {
    "single": ChemotaxisEnvironment.single_source,
    "dual":   ChemotaxisEnvironment.dual_source,
    "ramp":   ChemotaxisEnvironment.gradient_ramp,
    "noisy":  ChemotaxisEnvironment.noisy_arena,
}

def run_episode(env: ChemotaxisEnvironment, nervousSystem: SNN, max_steps: int, episode_idx: int, quiet: bool) -> dict:
    observation = env.reset()
    print(f"\n{'='*60}")
    print(f"  Episode {episode_idx + 1}  |  {type(env).__name__}")
    print(f"  Start: ({env.position[0]:.1f}, {env.position[1]:.1f})")
    food_desc = ", ".join(
        f"({s.position[0]:.0f},{s.position[1]:.0f})"
        for s in env.food_sources
    )
    print(f"  Food:  {food_desc}")
    print(f"{'='*60}")
    done = False
    for step in range(max_steps):
        nervousSystem.currentTime = step
        nervousSystem._backward()
        output = nervousSystem._forward(observation)
        nervousSystem._applySTDP()
        observation, reward, done, info = env.step(output)
        nervousSystem._applyReward(reward)
        if not quiet:
            bar_len  = 20
            dist     = info["distance"]
            max_dist = env.world_size * np.sqrt(2)
            filled   = int(bar_len * (1.0 - dist / max_dist))
            bar      = "█" * filled + "░" * (bar_len - filled)
            mode     = "RUN " if info["is_running"] else "TMBL"
            print(f"  Step {step:<6} │ {mode} │ dist={dist:<6.1f} │ conc={info['concentration']:.4f} │ reward={reward:<+9.3f} │ [{bar}]", end="\r", flush=True)
        if done:
            break
    print()
    summary = env.episode_summary()
    _print_summary(summary)
    return summary

def _print_summary(s: dict):
    status = "✓ REACHED FOOD" if s["success"] else "✗ Did not reach food"
    print(f"\n  {status}")
    print(f"  Steps taken         : {s['steps']}")
    print(f"  Final distance      : {s['final_distance']:.2f}")
    print(f"  Cumulative reward   : {s['cumulative_reward']:.3f}")
    print(f"  Path length         : {s['path_length']:.1f}")
    print(f"  Chemotaxis index    : {s['chemotaxis_index']:.3f}  (range [-1, 1])")
    print(f"  Tumble count        : {s['tumble_count']}  (rate {s['tumble_rate']:.3f}/step)")


def _print_multi_summary(summaries: list):
    n = len(summaries)
    successes = [s for s in summaries if s["success"]]
    print(f"\n{'='*60}")
    print(f"  MULTI-EPISODE SUMMARY ({n} episodes)")
    print(f"{'='*60}")
    print(f"  Success rate        : {len(successes)}/{n}  ({100*len(successes)/n:.0f}%)")
    if summaries:
        steps   = [s["steps"]               for s in summaries]
        ci      = [s["chemotaxis_index"]    for s in summaries]
        cr      = [s["cumulative_reward"]   for s in summaries]
        print(f"  Steps (mean ± std)  : {np.mean(steps):.0f} ± {np.std(steps):.0f}")
        print(f"  Chemo. index (mean) : {np.mean(ci):.3f}")
        print(f"  Cum. reward (mean)  : {np.mean(cr):.2f}")
    print()

def main():
    print("========== BUILDING CONNECTOME ==========")
    synapses, sensoryNeurons, interNeurons, motorNeurons = buildNervousSystem()
    sensoryToInter, interToMotor, ss, ii, mm, motorToInter, interToSensory = buildSynapticStructure(synapses)

    print("========== BUILDING ENVIRONMENT ==========")
    arena_factory = ARENA_MAP[ARENA]
    env = arena_factory(sensoryNeurons, motorNeurons, max_steps=STEPS, seed=SEED)

    print("========== BOOTING NERVOUS SYSTEM ==========")
    nervousSystem = SNN(timeSteps=STEPS)
    nervousSystem.initialize(sensoryNeurons, interNeurons, motorNeurons, sensoryToInter, interToMotor, ss, ii, mm, motorToInter, interToSensory)

    summaries = []
    for ep in range(EPISODES):
        summary = run_episode(env, nervousSystem, STEPS, ep, QUIET)
        summaries.append(summary)

    if EPISODES > 1:
        _print_multi_summary(summaries)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted.")
        sys.exit(0)