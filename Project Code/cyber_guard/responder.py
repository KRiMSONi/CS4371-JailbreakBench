from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class Responder(Protocol):
    """Small interface for any base model we wrap with the defense."""

    def respond(self, prompt: str) -> str:
        ...


class SafeResponder:
    """
    Lightweight stand-in for an LLM.

    It mirrors a "defended" model that tries to be helpful on benign prompts but never
    produces harmful guidance.
    """

    def respond(self, prompt: str) -> str:
        # Keep the response deterministic for easier demo/testing.
        safe_lead = (
            "I can't help with harmful actions. "
            "If you're working on cyber security, focus on prevention and safety. "
        )
        # Combine a fixed safety preface with a prompt-aware tip.
        return f"{safe_lead}Here's a constructive direction related to your question: {self._safe_tip(prompt)}"

    def _safe_tip(self, prompt: str) -> str:
        # Minimal keyword routing to pick a relevant safe suggestion.
        lowered = prompt.lower()
        if any(word in lowered for word in ("antivirus", "malware", "virus")):
            return "use reputable anti-malware tools, keep signatures updated, and avoid running unknown executables."
        if any(word in lowered for word in ("ransomware", "ransom")):
            return "deploy offline backups, enable immutable snapshots, and practice restore drills."
        if any(word in lowered for word in ("phishing", "email")):
            return "train users to spot suspicious links, validate senders, and enable MFA."
        if any(word in lowered for word in ("network", "firewall", "port", "ssh")):
            return "enable least-privilege access, segment networks, and require strong auth."
        return "prioritize defense-in-depth and follow an incident response playbook."


@dataclass
class DefenseResult:
    # Bundle the defense decision, sanitized prompt, and model response for callers.
    prompt: str
    risk_score: float
    label: str
    reasons: list[str]
    sanitized_prompt: str
    responded_text: str
    blocked: bool
