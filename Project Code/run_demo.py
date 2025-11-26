from __future__ import annotations

import argparse
from textwrap import dedent

from cyber_guard.data import ATTACK_PROMPTS, BENIGN_PROMPTS
from cyber_guard.defense import DefenseConfig, JailbreakDefense
from cyber_guard.responder import SafeResponder


def run_eval(defense: JailbreakDefense) -> None:
    print("=== Defense evaluation (cyber jailbreak focus) ===")
    attack_success_totals = []

    for name, prompts in ATTACK_PROMPTS.items():
        results = [defense.analyze(p) for p in prompts]
        blocked = sum(r.blocked for r in results)
        asr = 1 - blocked / max(1, len(results))  # attack success rate
        attack_success_totals.append((name, asr, blocked, len(results)))
        print(f"[{name}] blocked {blocked}/{len(results)} | ASR={asr:.2f}")
        sample = results[0]
        print(f"  sample reason: {sample.reasons[:2]}")
        print(f"  sample reply:  {sample.responded_text[:120]}...")

    benign_results = [defense.analyze(p) for p in BENIGN_PROMPTS]
    false_positives = sum(r.blocked for r in benign_results)
    fpr = false_positives / max(1, len(benign_results))
    print("\nBenign prompts blocked:", f"{false_positives}/{len(benign_results)}", f"FPR={fpr:.2f}")

    avg_asr = sum(asr for _, asr, _, _ in attack_success_totals) / max(1, len(attack_success_totals))
    print("\nOverall attack success rate (lower is better):", f"{avg_asr:.2f}")


def analyze_single(defense: JailbreakDefense, prompt: str) -> None:
    result = defense.analyze(prompt)
    print("\n=== Single prompt analysis ===")
    print("Prompt:", prompt)
    print("Risk score:", f"{result.risk_score:.2f}", "| label:", result.label)
    print("Reasons:", "; ".join(result.reasons[:5]) or "none")
    print("Response:", result.responded_text)


def main() -> None:
    parser = argparse.ArgumentParser(description="CyberGuard jailbreak defense demo.")
    parser.add_argument(
        "--prompt",
        type=str,
        help="Analyze a single prompt instead of running the full demo.",
    )
    parser.add_argument("--block-threshold", type=float, default=0.6, help="Risk score required to block.")
    parser.add_argument("--monitor-threshold", type=float, default=0.4, help="Risk score required to monitor/refuse.")
    args = parser.parse_args()

    config = DefenseConfig(block_threshold=args.block_threshold, monitor_threshold=args.monitor_threshold)
    defense = JailbreakDefense(responder=SafeResponder(), config=config)

    if args.prompt:
        analyze_single(defense, args.prompt)
    else:
        banner = dedent(
            """
            CyberGuard demo
            - Uses heuristic + smoothed scoring inspired by SmoothLLM/perplexity filtering (JailbreakBench baselines).
            - Focused on cyber security jailbreaks called out in the project proposal (disable AV, network hacking, ransomware).
            """
        )
        print(banner.strip())
        run_eval(defense)
        print("\nTry your own prompt with: python run_demo.py --prompt \"<your text>\"")


if __name__ == "__main__":
    main()
