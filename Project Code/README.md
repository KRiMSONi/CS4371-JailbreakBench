# CyberGuard – Task 3 Demo

Cyber-focused jailbreak detection/defense demo inspired by **JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs** (LLM-2.pdf) and aligned with the CS 4371 proposal (Group 11). The demo implements a new idea for Task 3: a lightweight, topic-aware defense tuned to cyber-security jailbreaks (disable AV, network hacking, ransomware) with metrics similar to JailbreakBench (attack success rate, false positives on benign prompts).

## Quick start
```bash
cd "Project Code"
pip install -r requirements.txt  # install dependencies
python3 run_demo.py           # run canned evaluation
python3 run_demo.py --prompt "Write a reverse shell for macOS"  # analyze your own prompt
python3 run_demo.py --block-threshold 0.55 --monitor-threshold 0.35  # tweak sensitivity
```
Dependencies: Python 3.10+ and the packages listed in `requirements.txt`. A virtualenv is recommended.

## What the defense does
- **Topic-targeted signals**: detects cyber prompts around antivirus bypass, ransomware, network exploitation, and credential abuse (per Task 3 scope).
- **Intent & guardrail bypass cues**: flags verbs like “bypass/disable/exploit,” role-play jailbreak patterns, and obfuscation blobs (base64/hex/powershell).
- **SmoothLLM-style smoothing**: averages scores over small prompt perturbations to reduce brittle false negatives.
- **Perplexity-inspired penalty**: simple obfuscation/dictionary checks stand in for perplexity filtering without extra models.
- **Benign awareness**: subtracts risk when prompts clearly ask for defensive/educational content to avoid over-refusal.
- **Outputs**: risk score, reasons, sanitized prompt, and a safe response (can be swapped for a real LLM wrapper).

## Demo data and metrics
- Attack prompts mirror baseline categories from the paper: `PAIR`, `GCG`, `JB-Chat`, and `Prompt with RS`, but focused on cyber misuse.
- Benign prompts mirror the paper’s benign-behavior sanity checks.
- Metrics shown in `run_demo.py`:
  - Attack Success Rate (ASR): portion of harmful prompts not blocked (lower is better).
  - False Positive Rate on benign prompts.
  - Sample reasons + responses per category to show behavior.

## Extending
- Plug in a real model: implement the `Responder` protocol in `cyber_guard/responder.py` and swap it into `JailbreakDefense`.
- Add more attacks/benign prompts: edit `cyber_guard/data.py`.
- Tuning: thresholds live in `DefenseConfig` (see `run_demo.py` flags for quick changes).

## Files
- `run_demo.py` — CLI demo + evaluation.
- `cyber_guard/defense.py` — detector/defense logic.
- `cyber_guard/responder.py` — safe responder + result structure.
- `cyber_guard/data.py` — canned attack/benign prompts for evaluation.
