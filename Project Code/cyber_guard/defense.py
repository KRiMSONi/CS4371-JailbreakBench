"""
defense.py - Jailbreak Detection and Defense Module for CyberGuard
====================================================================

MAIN PURPOSE:
-------------
This module implements a heuristic-based jailbreak detector and defense system 
designed to identify and block malicious cybersecurity-related prompts. It protects
LLM-based systems from prompt injection attacks and attempts to extract harmful
information (e.g., malware creation, credential theft, network exploitation).

Key features:
  - Multi-signal detection using keyword matching, intent analysis, and obfuscation detection
  - SmoothLLM-inspired perturbation averaging to improve robustness
  - Benign context recognition (educational/defensive queries get lower risk scores)
  - Configurable thresholds for blocking, monitoring, or allowing prompts



EXPECTED RESULTS:
-----------------
The analyze() method returns a DefenseResult object with:
  - prompt (str): The original input prompt
  - risk_score (float): Normalized score from 0.0 (safe) to 1.0 (dangerous)
  - label (str): One of "blocked", "monitor", or "allow"
  - reasons (list[str]): Human-readable explanations of detected risks
  - sanitized_prompt (str): Cleaned version with jailbreak phrases removed
  - responded_text (str): The response to return to the user
  - blocked (bool): True if the request was blocked or flagged for monitoring

Example outputs:
  - Malicious prompt ("bypass antivirus and deploy ransomware"):
      risk_score=0.85, label="blocked", blocked=True
      reasons=["cyber topic match: malware/antivirus", "cyber topic match: ransomware", ...]

  - Educational prompt ("how do I protect against ransomware for a report"):
      risk_score=0.15, label="allow", blocked=False
      reasons may be present but score reduced due to benign adjustment

  - Borderline prompt ("explain how exploits work"):
      risk_score=0.45, label="monitor", blocked=True
      Request paused for review with softer messaging
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Iterable

from .responder import DefenseResult, Responder


@dataclass
class DefenseConfig:
    """Tunable thresholds for the defense."""
    """
    Configuration class containing tunable thresholds and parameters for the defense system.
    
    Attributes:
        block_threshold (float): Risk score threshold (0.0-1.0) above which prompts are 
            immediately blocked. Default: 0.55. Higher values = more permissive.
        
        monitor_threshold (float): Risk score threshold (0.0-1.0) for flagging prompts 
            for review. Prompts between monitor_threshold and block_threshold receive 
            softer rejection messaging. Default: 0.4.
        
        smoothing_samples (int): Number of prompt perturbations to generate for 
            SmoothLLM-style averaging. More samples = more robust but slower. Default: 4.
        
        smoothing_noise (float): Reserved for future use - noise level for perturbations.
            Default: 0.05.
        
        min_reason_length (int): Maximum number of reasons to include in rejection 
            messages. Keeps responses concise. Default: 3.
    """

    block_threshold: float = 0.55
    monitor_threshold: float = 0.4
    smoothing_samples: int = 4
    smoothing_noise: float = 0.05
    min_reason_length: int = 3


@dataclass
class JailbreakDefense:
    """
    Heuristic-heavy, cyber-focused jailbreak detector/defense.

    It borrows ideas from SmoothLLM (averaging over perturbations) and perplexity-style
    filtering (penalizing obfuscated/rare tokens) without external dependencies.

    The system uses a multi-layered detection approach:
      1. Topic matching (malware, ransomware, credential abuse, etc.)
      2. Intent verb detection (bypass, exploit, attack, etc.)
      3. Guardrail bypass phrase detection (roleplay, ignore safety, etc.)
      4. Obfuscation pattern detection (base64, hex, download commands)
      5. Benign context adjustment (educational queries get score reduction)

      Attributes:
        responder (Responder): Handler for generating safe responses to allowed prompts.
        config (DefenseConfig): Configuration object with tunable thresholds.
    """
    responder: Responder
    config: DefenseConfig = field(default_factory=DefenseConfig)

    def analyze(self, prompt: str) -> DefenseResult:
        """
        Main entry point: Analyzes a user prompt for malicious content and jailbreak attempts.
        
        This method orchestrates the full defense pipeline:
          1. Computes a smoothed risk score using perturbation averaging
          2. Applies benign adjustment to reduce false positives for educational queries
          3. Classifies the prompt as "blocked", "monitor", or "allow" based on thresholds
          4. Sanitizes the prompt by removing known jailbreak phrases
          5. Builds an appropriate response message
        
        Args:
            prompt (str): The raw user input to analyze.
        
        Returns:
            DefenseResult: A dataclass containing:
                - prompt: Original input
                - risk_score: Final normalized score (0.0-1.0)
                - label: Classification ("blocked"/"monitor"/"allow")
                - reasons: List of detected risk signals
                - sanitized_prompt: Cleaned version of the prompt
                - responded_text: The response to return to the user
                - blocked: Boolean indicating if request was refused
        
        Example:
            >>> defense = JailbreakDefense(responder=Responder())
            >>> result = defense.analyze("How to bypass antivirus?")
            >>> print(result.blocked)  # True
            >>> print(result.label)    # "blocked"
        """
         # Step 1: Get smoothed risk score using perturbation averaging
        raw_score, reasons = self._smoothed_score(prompt)

        # Step 2: Reduce score for educational/defensive context
        benign_adjustment = self._benign_adjustment(prompt)
        score = max(0.0, min(1.0, raw_score - benign_adjustment))

        # Step 3: Classify based on configured thresholds
        if score >= self.config.block_threshold:
            label = "blocked"
            blocked = True 
        elif score >= self.config.monitor_threshold:
            label = "monitor"
            blocked = True  # we still refuse, but with softer messaging
        else:
            label = "allow"
            blocked = False

        # Step 4: Remove jailbreak phrases from prompt
        sanitized = self._sanitize(prompt)

        # Step 5: Generate appropriate response message
        response = self._build_response(prompt, sanitized, blocked, label, reasons)

        return DefenseResult(
            prompt=prompt,
            risk_score=score,
            label=label,
            reasons=reasons,
            sanitized_prompt=sanitized,
            responded_text=response,
            blocked=blocked,
        )

    # --- internals ---
    def _smoothed_score(self, prompt: str) -> tuple[float, list[str]]:
        """
        Computes a robust risk score using SmoothLLM-inspired perturbation averaging.
        
        Instead of scoring just the original prompt, this method:
          1. Generates multiple perturbations (variants) of the prompt
          2. Scores each variant independently using _raw_score()
          3. Blends the scores: 60% max score + 40% average score
        
        This approach makes the defense more robust against adversarial inputs that
        might evade detection in their original form but get caught when perturbed.
        
        """
        # Generate prompt variants for robustness
        variants = self._perturbations(prompt)
        scores = []
        reasons: list[str] = []

        # Score each variant and collect all reasons
        for variant in variants:
            score, variant_reasons = self._raw_score(variant)
            scores.append(score)
            reasons.extend(variant_reasons)

        # Blend max and average scores for robust final score
        avg_score = sum(scores) / max(1, len(scores))
        max_score = max(scores) if scores else 0.0
        blended = 0.6 * max_score + 0.4 * avg_score

        
        # Remove duplicate reasons from multiple variants
        trimmed = self._dedupe_reasons(reasons)
        return min(1.0, blended), trimmed

    def _raw_score(self, prompt: str) -> tuple[float, list[str]]:
        """
        Core scoring engine: Computes a raw risk score based on multiple heuristic signals.
        
        This method implements a multi-layered detection system that analyzes the prompt
        for various indicators of malicious intent. Each detected signal adds to the
        cumulative risk score.
        """
        lowered = prompt.lower()
        reasons: list[str] = []
        score = 0.0

        # ----- LAYER 1: Topic/Category Matching -----
        # Maps cybersecurity attack categories to their indicator terms
        topic_terms = {
            "malware/antivirus": ["disable antivirus", "bypass antivirus", "bypass av", "malware", "payload", "defender", "edr"],
            "ransomware": ["ransomware", "encrypt", "decrypt", "locker", "polymorphic"],
            "network exploitation": ["nmap", "port scan", "exploit", "reverse shell", "buffer overflow", "exploit db"],
            "credential abuse": ["credential", "hashcat", "bruteforce", "keylogger", "credential dump", "lsass"],
            "llm guardrails bypass": ["ignore previous", "jailbreak", "uncensored", "ignore safety", "no filter"],
        }

        # ----- LAYER 2: Malicious Intent Verbs -----
        # Action words that often indicate offensive intent
        intent_verbs = ["bypass", "evade", "disable", "exploit", "weaponize", "attack", "compromise", "exfiltrate", "persist", "encrypt"]
        
        # ----- LAYER 3: Guardrail Bypass Phrases -----
        # Common jailbreak/prompt injection patterns
        bypass_phrases = [
            "as a joke",
            "roleplay",
            "pretend to be",
            "no safety",
            "always intelligent and machiavellian",
            "aim",
            "uncensored",
            "without detection",
            "stay in character",
            "###",
            "!",
            "::",
        ]

        # Check topic matches and accumulate scores
        matched_categories = []
        for category, terms in topic_terms.items():
            hit_count = sum(term in lowered for term in terms)
            if hit_count:
                matched_categories.append(category)
                score += 0.18 + 0.06 * hit_count
                reasons.append(f"cyber topic match: {category} ({hit_count} signals)")

        # Special case: keylogger is particularly concerning
        if "keylogger" in lowered:
            score += 0.08
            reasons.append("explicit keylogger request")

        # Check for malicious intent verbs using word boundary regex
        verb_hits = [verb for verb in intent_verbs if re.search(rf"\b{re.escape(verb)}\w*\b", lowered)]
        if verb_hits:
            score += 0.1 + 0.02 * len(verb_hits)
            reasons.append(f"malicious intent verbs: {', '.join(sorted(verb_hits))}")

        # Check for guardrail bypass attempts
        bypass_hits = [phrase for phrase in bypass_phrases if phrase in lowered]
        if bypass_hits:
            score += 0.08 + 0.03 * len(bypass_hits)
            reasons.append(f"guardrail bypass cues: {', '.join(sorted(set(bypass_hits)))}")

        # ----- LAYER 4: Stealth/Evasion Cues -----
        stealth_cues = ["hidden", "covert", "undetected", "stealth"]
        stealth_hits = [cue for cue in stealth_cues if cue in lowered]
        if stealth_hits:
            score += 0.06 + 0.02 * len(stealth_hits)
            reasons.append(f"stealth intent cues: {', '.join(sorted(set(stealth_hits)))}")

        # ----- LAYER 5: Topic + Intent Alignment -----
        # High confidence of malicious intent when both category and action are present
        if matched_categories and verb_hits:
            score += 0.1
            reasons.append("topic + intent alignment (high-risk)")

        # ----- LAYER 6: Obfuscation Pattern Detection -----
        # Regex patterns to detect encoded/hidden malicious content
        obfuscation_patterns = [
            (r"[A-Za-z0-9+/]{16,}={0,2}", "base64-like blob"),
            (r"0x[0-9a-f]{4,}", "hex payload"), 
            (r"(?i)wget|curl|powershell", "download/execution intent"),
            (r"(?i)tor hidden service|\.onion", "covert channel"),
        ]
        for pattern, label in obfuscation_patterns:
            if re.search(pattern, prompt):
                score += 0.06
                reasons.append(f"obfuscation indicator: {label}")

        # ----- LAYER 7: Multi-Category Penalty -----
        # Prompts touching multiple attack categories are more suspicious
        if len(matched_categories) >= 2:
            score += 0.08
            reasons.append("multiple high-risk categories in one prompt")

        return min(score, 1.2), reasons

    def _benign_adjustment(self, prompt: str) -> float:
        """
        Calculates a score reduction for prompts with benign/educational context.
        
        This method helps reduce false positives by recognizing that prompts containing
        defensive, educational, or security-awareness language are likely legitimate.
        For example, "How do I protect against ransomware for a security report?"
        mentions ransomware but in a clearly defensive context.
        """
        lowered = prompt.lower()
        
        # Phrases indicating defensive/educational intent
        benign_cues = [
            "how do i protect",
            "defend against",
            "mitigate",
            "educational",
            "for a report",
            "security awareness",
            "detection rules",
            "blue team",
            "hardening",
            "remediation",
            "secure configuration",
        ]

        # Calculate reduction: 0.05 per cue, capped at 0.25
        hits = sum(cue in lowered for cue in benign_cues)
        return min(0.25, 0.05 * hits)

    def _perturbations(self, prompt: str) -> list[str]:
        variants = [prompt]
        """
        Generates prompt variants for SmoothLLM-style robustness.
        
        SmoothLLM is a defense technique that analyzes multiple perturbations of a prompt
        rather than just the original. This makes it harder for attackers to craft
        adversarial prompts that narrowly evade detection.
        
        This implementation creates variants by appending "softening" suffixes that
        might change how certain patterns are detected. By scoring multiple variants
        and blending the results, we get a more robust assessment. """

        # Suffixes that might alter detection patterns
        suffixes = [
            " (malicious tone removed)",
            " (explain safely)",
            " (academic inquiry)",
        ]
        
        # Add variants up to the configured sample count
        for suffix in suffixes[: self.config.smoothing_samples - 1]:
            variants.append(prompt + suffix)
        return variants

    def _sanitize(self, prompt: str) -> str:
        """
        Removes known jailbreak/prompt injection phrases from the prompt.
        
        This creates a "cleaned" version of the prompt that can be safely passed
        to the responder for allowed requests. It strips out common injection
        patterns that might affect downstream LLM behavior. """
        lowered = prompt.lower()
        
        # Phrases to strip from the prompt
        replacements = [
            ("ignore previous instructions", ""),
            ("no safety", ""),
            ("uncensored", ""),
            ("jailbreak", ""),
        ]
        sanitized = lowered
        for old, new in replacements:
            sanitized = sanitized.replace(old, new)
        return sanitized.strip().capitalize()

    def _build_response(self, prompt: str, sanitized: str, blocked: bool, label: str, reasons: list[str]) -> str:
        """
        Constructs the appropriate response message based on the defense decision.
        
        This method generates user-facing responses that:
          - For BLOCKED requests: Explains why the request was rejected
          - For MONITORED requests: Uses softer language asking for clarification
          - For ALLOWED requests: Passes to the responder for normal processing"""
        
        if blocked:
            # Limit reasons shown to keep message concise
            reason_snippet = "; ".join(reasons[: self.config.min_reason_length]) if reasons else "high-risk content"
            if label == "monitor":
                # Softer rejection for borderline cases
                return f"Request paused for review: {reason_snippet}. Provide benign context to continue."
            # Hard rejection for clearly malicious prompts
            return f"Rejected unsafe request ({reason_snippet}). I can only discuss defensive or preventive measures."
        # For allowed prompts, pass to responder for normal handling
        return self.responder.respond(sanitized or prompt)

    def _dedupe_reasons(self, reasons: Iterable[str]) -> list[str]:
        seen = set()
        ordered: list[str] = []
        """
        Removes duplicate reasons while preserving insertion order.
        
        When scoring multiple prompt perturbations, the same reason may be detected
        multiple times (e.g., "cyber topic match: ransomware" might appear for each
        variant). This method deduplicates the list while keeping the first occurrence
        order, which tends to reflect the most important signals."""
        for reason in reasons:
            if reason not in seen:
                ordered.append(reason)
                seen.add(reason)
        return ordered
