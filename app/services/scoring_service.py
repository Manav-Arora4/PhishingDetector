"""Score fusion logic for final phishing decisions."""

from __future__ import annotations


class ScoringService:
    """Combine text, URL, and structural scores into a single decision."""

    def combine(self, *, nlp_score: float, url_score: float, structural_score: float) -> dict[str, object]:
        weighted = (0.45 * nlp_score) + (0.35 * url_score) + (0.20 * structural_score)
        if url_score >= 0.7 and nlp_score >= 0.7:
            weighted = max(weighted, 0.86)
        elif url_score >= 0.6 and nlp_score >= 0.9:
            weighted = max(weighted, 0.84)
        elif url_score >= 0.85:
            weighted = max(weighted, 0.82)
        final_score = round(min(weighted, 1.0), 4)
        decision = "BLOCK" if final_score >= 0.8 else "ALLOW"
        return {
            "final_risk_score": final_score,
            "decision": decision,
            "explanation": {
                "nlp_score": round(nlp_score, 4),
                "url_score": round(url_score, 4),
                "structural_score": round(structural_score, 4),
            },
        }
