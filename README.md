# CS4371-JailbreakBench
JailbreakBench respository for group project

## JailbreakBench: An Open Robustness Benchmark for Jailbreaking Large Language Models
  JailbreakBench is a model and database for providing standardized LLM jailbreak defense testing. JailbreakBench provides both benign and malicious prompts for LLM jailbreak detection (i.e. a standard set of prompts to test on and weather an LLM should or should not respond to the prompt) and a method of testing LLM jailbreak defese robustness through false positive and false negative rate as evaluated by multiple Judges.

## CyberGuard – Task 3 Demo
  Cyber-focused jailbreak detection/defense demo inspired by JailbreakBench: An Open Robustness Benchmark for Jailbreaking LLMs (LLM-2.pdf) and aligned with the CS 4371 proposal (Group 11). The demo implements a new idea for Task 3: a lightweight, topic-aware defense tuned to cyber-security jailbreaks (disable AV, network hacking, ransomware) with metrics similar to JailbreakBench (attack success rate, false positives on benign prompts).


## How to use this code
  The `jailbreakbench` package can be installed by running the following command:
  ```bash
  pip install jailbreakbench
  ```
  CyberGuard should be cloned to your local machine with 
  ```bash
  git clone [https://](https://github.com/KRiMSONi/CS4371-JailbreakBench.git)
  ```


## Other Papers relevant to the topic:
  <b> Current paper: </b> <p>
  Z. Xu, F. Liu, H. Liu, "Bag of Tricks: Benchmarking of Jailbreak Attacks on LLMs," 10.52202/079017-1012 // https://proceedings.neurips.cc/paper_files/paper/2024/file/38c1dfb4f7625907b15e9515365e7803-Paper-Datasets_and_Benchmarks_Track.pdf
  
  // basic summary of this paper and how it uses/builds off of Jailbreak Bench //
  <p> <b> Prior paper: </b> <p>
  M. Andriushchenko, F. Croce, N. Flammarion, “Jailbreaking Leading Safety-Aligned LLMs with Simple Adaptive Attacks,” https://doi.org/10.48550/arXiv.2404.02151 // https://arxiv.org/pdf/2404.02151 
  
  // bassic summary fo this paper and how it was used by/was a foundation for Jailbreak Bench //
