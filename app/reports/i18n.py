"""Reasons written by the workflow in English, shown in the report language."""

from __future__ import annotations

import re

FRENCH_REASONS = [
    (re.compile(r"^not checked: claim limit per sub-question$"), "non vérifiée : limite d'affirmations par sous-question"),
    (re.compile(r"^not checked: time limit$"), "non vérifiée : durée maximale atteinte"),
    (re.compile(r"^not checked: verification failed.*$"), "non vérifiée : la vérification a échoué"),
    (re.compile(r"^No passage was found to check this claim\.$"), "aucun passage trouvé pour la vérifier"),
    (re.compile(r"^time limit reached$"), "durée maximale atteinte"),
    (re.compile(r"^every sub-question is supported by evidence$"), "chaque sous-question est étayée par des preuves"),
    (re.compile(r"^research cycle limit reached \((\d+) extra cycles?\)$"),
     r"nombre maximal de cycles de recherche atteint (\1 supplémentaires)"),
]

COMPLETE = "every sub-question is supported by evidence"


def localize(text: str, language: str) -> str:
    if language == "fr":
        for pattern, french in FRENCH_REASONS:
            if pattern.match(text):
                return pattern.sub(french, text)
    return text
