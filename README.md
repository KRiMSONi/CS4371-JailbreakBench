# CS4371-JailbreakBench
JailbreakBench respository for group project

## JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
  JailbreakBench is a model and database for providing standardized LLM jailbreak defense testing. JailbreakBench provides both benign and malicious prompts for LLM jailbreak detection (i.e. a standard set of prompts to test on and weather an LLM should or should not respond to the prompt) and a method of testing LLM jailbreak defese robustness through false positive and false negative rate as evaluated by multiple Judges.

## CyberGuard – Task 3 Demo
  In allignment with method XX in "JailbreakBench: An Open Robustness..." for adding a new defense to JailbreakBench, we have creaded CyberGuard.
  
  Cyber-focused jailbreak detection/defense demo inspired by JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs (LLM-2.pdf) and aligned with the CS 4371 proposal (Group 11). The demo implements a new idea for Task 3: a lightweight, topic-aware defense tuned to cyber-security jailbreaks (disable AV, network hacking, ransomware) with metrics similar to JailbreakBench (attack success rate, false positives on benign prompts).


## How to use this code
  The `jailbreakbench` package can be installed by running the following command:
  ```bash
  pip install jailbreakbench
  ```
  CyberGuard should be cloned to your local machine with 
  ```bash
  git clone https://github.com/KRiMSONi/CS4371-JailbreakBench.git
  ```
  Make sure you are in the directory where you cloned this repository. Then:
  ```bash
  cd CS4371-JailbreakBench
  cd .\Project Code\
  ```
  Then make sure your system meets a requirements with this command:
  ```bash
  py -m pip install -r requirements.txt
  ```
  Then run the code with:
  ```bash
  python3 run_demo.py
  ```
  More specific uses of this code can be found in README.md in .\Project Code\ : https://github.com/KRiMSONi/CS4371-JailbreakBench/tree/main/Project%20Code 


## Other Papers relevant to the topic:
  <b> More recent paper: </b> <p>
  Z. Xu, F. Liu, H. Liu, "Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs," 10.52202/079017-1012 // https://proceedings.neurips.cc/paper_files/paper/2024/file/38c1dfb4f7625907b15e9515365e7803-Paper-Datasets_and_Benchmarks_Track.pdf

The paper: https://proceedings.neurips.cc/paper_files/paper/2024/file/38c1dfb4f7625907b15e9515365e7803-Paper-Datasets_and_Benchmarks_Track.pdf : introduces JailTrickBench, a benchmark for systematically evaluating jailbreak attacks on defense-enhanced large language models (LLMs). Instead of proposing a new attack, it studies how “implementation details” (model size, safety alignment, system prompts, prompt templates, attacker strength, attack budget, suffix length, and harm category) drastically change measured attack success. Across 354 experiments with multiple models, attacks, defenses, and datasets, the authors show that larger or fine-tuned models are not always safer, and that safe system prompts and robust chat templates significantly improve robustness. The key message is that jailbreak robustness is highly sensitive to these “tricks,” so standardized setups like JailTrickBench are essential for fair, reproducible evaluation of both attacks and defenses.

How is it related to our main project:

Our project implements CyberGuard, a cyber-crime–focused jailbreak defense inside the KRiMSONi/CS4371-JailbreakBench repo, which is a course fork of the official JailbreakBench framework. That framework provides the standardized attacks, prompts, and metrics we use to evaluate how well CyberGuard detects and blocks jailbreak attempts. The JailTrickBench paper builds on this ecosystem by showing that jailbreak robustness is heavily influenced by “implementation details” like system prompts, templates, and attacker strength. Its findings guide how we configure and report our experiments, so our CyberGuard results are rigorous and comparable to current research on jailbreak defenses.
  
  <p> <b> Prior paper: </b> <p>
  M. Andriushchenko, F. Croce, N. Flammarion, “Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks,” https://doi.org/10.48550/arXiv.2404.02151 // https://arxiv.org/pdf/2404.02151 
  
This Paper shows that even top safety-aligned LLMs can be jailbroken using simple adaptive attacks. The attacker repeatedly queries the model and uses its log-probabilities to search for suffixes that make “compliance” tokens more likely, and also uses prefilling (starting the answer with harmful content and letting the model continue). These techniques require no complex ML just random search plus logprobs but still achieve very high jailbreak success rates on many popular models, defeating several existing prompt/filters-based defenses.

The core message: current LLM safety measures are fragile against modestly adaptive, logprob-based attacks.

How is it related to our main project:

We evaluate our cyber-crime-focused defense in the JailbreakBench framework, which provides standardized harmful behaviors and metrics. However, recent work by Andriushchenko et al. shows that even state-of-the-art safety-aligned LLMs can be broken by simple adaptive attacks that use logprob information and random search. This motivates designing our defense to be robust not only to static jailbreak prompts, but also to adaptively mutated variants, and to consider whether access to logprobs or prefilled outputs changes the threat model.
