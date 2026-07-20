"""Taxonomia SwDA usada no novo experimento.

A ordem é a mesma das 41 entradas originais do notebook. As três classes
exploratórias (`cd`, `ac` e `err`) foram removidas porque não pertencem ao SwDA
descrito no artigo e não possuem exemplos nos CSVs de treino/validação usados.
"""

from __future__ import annotations


SWDA_CODE_TO_NAME: dict[str, str] = {
    "sd": "Statement-non-opinion",
    "b": "Acknowledge (Backchannel)",
    "sv": "Statement-opinion",
    "%": "Uninterpretable",
    "aa": "Agree/Accept",
    "ba": "Appreciation",
    "qy": "Yes-No-Question",
    "ny": "Yes Answers",
    "fc": "Conventional-closing",
    "qw": "Wh-Question",
    "nn": "No Answers",
    "bk": "Response Acknowledgement",
    "h": "Hedge",
    "qy^d": "Declarative Yes-No-Question",
    "bh": "Backchannel in Question Form",
    "^q": "Quotation",
    "bf": "Summarize/Reformulate",
    'fo_o_fw_"_by_bc': "Other",
    "na": "Affirmative Non-yes Answers",
    "ad": "Action-directive",
    "^2": "Collaborative Completion",
    "b^m": "Repeat-phrase",
    "qo": "Open-Question",
    "qh": "Rhetorical-Question",
    "^h": "Hold Before Answer/Agreement",
    "ar": "Reject",
    "ng": "Negative Non-no Answers",
    "br": "Signal-non-understanding",
    "no": "Other Answers",
    "fp": "Conventional-opening",
    "qrr": "Or-Clause",
    "arp_nd": "Dispreferred Answers",
    "t3": "3rd-party-talk",
    "oo_co_cc": "Offers, Options, Commits",
    "aap_am": "Maybe/Accept-part",
    "t1": "Downplayer",
    "bd": "Self-talk",
    "^g": "Tag-Question",
    "qw^d": "Declarative Wh-Question",
    "fa": "Apology",
    "ft": "Thanking",
}

SWDA_CODES: tuple[str, ...] = tuple(SWDA_CODE_TO_NAME)
LABEL_TO_ID: dict[str, int] = {code: index for index, code in enumerate(SWDA_CODES)}
ID_TO_LABEL: dict[int, str] = {index: code for code, index in LABEL_TO_ID.items()}


def validate_observed_labels(labels: set[str]) -> None:
    """Falha quando os dados não correspondem exatamente à taxonomia esperada."""

    expected = set(SWDA_CODES)
    if labels != expected:
        missing = sorted(expected - labels)
        unexpected = sorted(labels - expected)
        raise ValueError(
            "Os rótulos do CSV divergem das 41 classes SwDA: "
            f"ausentes={missing}; inesperados={unexpected}."
        )
