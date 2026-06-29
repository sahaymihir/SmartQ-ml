from __future__ import annotations

import re
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict, Field


MODEL_SOURCE = "specialty_hybrid_v1"
LOW_CONFIDENCE_THRESHOLD = 0.58
GENERAL_SPECIALTY = "General Practice"
GENERAL_ROUTE = "General OPD"

SPECIALTY_ROUTE_MAP: dict[str, str] = {
    "Cardiology": "Cardiology",
    "Orthopaedics": "Orthopaedics",
    "Neurology": "Neurology",
    "Dermatology": "Dermatology",
    "Gastroenterology": "Gastroenterology",
    "Paediatrics": "Paediatrics",
    "Pulmonology": "Pulmonology",
    GENERAL_SPECIALTY: GENERAL_ROUTE,
    "Infectious Disease": GENERAL_ROUTE,
    "Otolaryngology (ENT)": GENERAL_ROUTE,
    "Hematology": GENERAL_ROUTE,
    "Endocrinology": GENERAL_ROUTE,
    "Nephrology / Urology": GENERAL_ROUTE,
    "Emergency Medicine": GENERAL_ROUTE,
}

KNOWN_COLLAPSE_TOKENS = {
    "abdomen",
    "acidity",
    "allergy",
    "anemia",
    "asthma",
    "back",
    "bleeding",
    "blood",
    "breath",
    "breathing",
    "bruise",
    "burning",
    "chest",
    "child",
    "cold",
    "congestion",
    "constipation",
    "cough",
    "diabetes",
    "diarrhea",
    "dizziness",
    "ear",
    "eczema",
    "fever",
    "fracture",
    "gas",
    "giddiness",
    "hair",
    "headache",
    "heart",
    "itching",
    "joint",
    "kidney",
    "migraine",
    "nausea",
    "nose",
    "numbness",
    "pain",
    "palpitations",
    "pale",
    "phlegm",
    "pressure",
    "rash",
    "seizure",
    "shortness",
    "sinus",
    "skin",
    "speech",
    "sprain",
    "stomach",
    "stroke",
    "sugar",
    "swelling",
    "throat",
    "thyroid",
    "tingling",
    "urination",
    "vomiting",
    "weakness",
    "wheezing",
}

TOKEN_CORRECTIONS = {
    "abdominal": "abdominal",
    "anaemia": "anemia",
    "breathless": "breathless",
    "breathlessness": "breathlessness",
    "brething": "breathing",
    "cof": "cough",
    "coufh": "cough",
    "couhg": "cough",
    "coldd": "cold",
    "feveer": "fever",
    "fevar": "fever",
    "feverishh": "feverish",
    "gastic": "gastric",
    "giddyness": "giddiness",
    "hartrate": "heart",
    "headace": "headache",
    "heeart": "heart",
    "hairt": "heart",
    "infact": "infection",
    "migarine": "migraine",
    "paining": "pain",
    "palpitationz": "palpitations",
    "phlem": "phlegm",
    "sholder": "shoulder",
    "sinusses": "sinus",
    "soar": "sore",
    "stomache": "stomach",
    "thorat": "throat",
    "thyriod": "thyroid",
    "vomitting": "vomiting",
    "weezing": "wheezing",
}

PHRASE_REPLACEMENTS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"\bheart(?: is|s)? pain\b"), "heart pain"),
    (re.compile(r"\bchest(?: is|s)? pain\b"), "chest pain"),
    (re.compile(r"\bbreathless(?:ness)?\b"), "shortness of breath"),
    (
        re.compile(
            r"\b(?:can not breathe|cannot breathe|cant breathe|can't breathe|hard to breathe|difficulty breathing)\b"
        ),
        "shortness of breath",
    ),
    (re.compile(r"\bhigh bp\b"), "high blood pressure"),
    (re.compile(r"\bbody aches?\b"), "body ache"),
    (re.compile(r"\brunny nose\b"), "nose congestion"),
    (re.compile(r"\bheart burn\b"), "heartburn"),
    (re.compile(r"\bskin allergy\b"), "itchy rash"),
    (re.compile(r"\bloose motions?\b"), "diarrhea"),
    (re.compile(r"\bchest heaviness\b"), "chest pressure"),
)

CONTEXTUAL_COMPLAINT_BOOSTS: dict[str, dict[str, float]] = {
    "cardiac": {"Cardiology": 1.6, "Emergency Medicine": 0.4},
    "respiratory": {"Pulmonology": 1.3, "Infectious Disease": 0.6},
    "neurological": {"Neurology": 1.5, "Emergency Medicine": 0.4},
    "gastrointestinal": {"Gastroenterology": 1.4},
    "trauma": {"Orthopaedics": 1.6, "Emergency Medicine": 0.5},
    "renal": {"Nephrology / Urology": 1.4},
    "endocrine": {"Endocrinology": 1.5},
    "dermatological": {"Dermatology": 1.5},
    "dermatology": {"Dermatology": 1.5},
}


@dataclass(frozen=True)
class SignalRule:
    signal: str
    patterns: tuple[str, ...]
    weight: float
    red_flag: bool = False


SPECIALTY_RULES: dict[str, tuple[SignalRule, ...]] = {
    "Cardiology": (
        SignalRule(
            signal="chest pain",
            patterns=(r"\bchest pain\b", r"\bchest pressure\b", r"\bchest tight(?:ness)?\b", r"\bheart pain\b"),
            weight=3.2,
            red_flag=True,
        ),
        SignalRule(
            signal="shortness of breath",
            patterns=(r"\bshortness of breath\b", r"\bbreathlessness\b"),
            weight=2.1,
            red_flag=True,
        ),
        SignalRule(
            signal="palpitations",
            patterns=(r"\bpalpitation(?:s)?\b", r"\bheart racing\b", r"\bracing heart\b"),
            weight=1.8,
        ),
        SignalRule(
            signal="blood pressure issue",
            patterns=(r"\bhigh blood pressure\b", r"\bhypertension\b", r"\bblood pressure\b"),
            weight=1.4,
        ),
        SignalRule(
            signal="leg swelling",
            patterns=(r"\bleg swelling\b", r"\bswollen (?:legs|ankles|feet)\b"),
            weight=1.2,
        ),
    ),
    "Pulmonology": (
        SignalRule(
            signal="shortness of breath",
            patterns=(r"\bshortness of breath\b", r"\bdifficulty breathing\b", r"\bbreathing trouble\b"),
            weight=2.4,
            red_flag=True,
        ),
        SignalRule(
            signal="cough",
            patterns=(r"\bcough(?:ing)?\b", r"\bphlegm\b", r"\bsputum\b"),
            weight=1.7,
        ),
        SignalRule(
            signal="wheezing",
            patterns=(r"\bwheez(?:e|ing)\b", r"\basthma\b"),
            weight=2.1,
        ),
        SignalRule(
            signal="chest congestion",
            patterns=(r"\bchest congestion\b", r"\blung pain\b", r"\brespiratory\b"),
            weight=1.2,
        ),
    ),
    "Neurology": (
        SignalRule(
            signal="headache",
            patterns=(r"\bheadache\b", r"\bmigraine\b"),
            weight=1.6,
        ),
        SignalRule(
            signal="dizziness",
            patterns=(r"\bdizz(?:y|iness)\b", r"\bvertigo\b", r"\bgiddiness\b"),
            weight=1.5,
        ),
        SignalRule(
            signal="seizure activity",
            patterns=(r"\bseizure(?:s)?\b", r"\bfits\b", r"\bconvulsion(?:s)?\b"),
            weight=3.0,
            red_flag=True,
        ),
        SignalRule(
            signal="numbness or tingling",
            patterns=(r"\bnumb(?:ness)?\b", r"\btingling\b"),
            weight=2.2,
            red_flag=True,
        ),
        SignalRule(
            signal="focal neurological deficit",
            patterns=(r"\bslurred speech\b", r"\bface droop\b", r"\bparalysis\b", r"\bstroke\b", r"\bone sided weakness\b"),
            weight=3.2,
            red_flag=True,
        ),
    ),
    "Dermatology": (
        SignalRule(
            signal="itchy rash",
            patterns=(r"\brash\b", r"\bitch(?:ing|y)?\b", r"\bhives\b", r"\bskin allergy\b"),
            weight=2.4,
        ),
        SignalRule(
            signal="skin lesion",
            patterns=(r"\bacne\b", r"\bblister(?:s)?\b", r"\blesion(?:s)?\b", r"\bboil(?:s)?\b"),
            weight=1.7,
        ),
        SignalRule(
            signal="eczema or psoriasis",
            patterns=(r"\beczema\b", r"\bpsoriasis\b", r"\bdry skin\b"),
            weight=1.6,
        ),
        SignalRule(
            signal="hair loss",
            patterns=(r"\bhair loss\b", r"\bfalling hair\b"),
            weight=1.3,
        ),
    ),
    "Gastroenterology": (
        SignalRule(
            signal="abdominal pain",
            patterns=(r"\babdominal pain\b", r"\bstomach pain\b", r"\bbelly pain\b", r"\babdomen pain\b"),
            weight=2.3,
        ),
        SignalRule(
            signal="nausea or vomiting",
            patterns=(r"\bnausea\b", r"\bvomiting\b", r"\bthrowing up\b"),
            weight=1.7,
        ),
        SignalRule(
            signal="diarrhea",
            patterns=(r"\bdiarrhea\b", r"\bloose stool\b", r"\bloose motion\b"),
            weight=1.6,
        ),
        SignalRule(
            signal="acidity or reflux",
            patterns=(r"\bacidity\b", r"\bheartburn\b", r"\breflux\b", r"\bindigestion\b", r"\bgas\b", r"\bbloating\b"),
            weight=1.5,
        ),
        SignalRule(
            signal="liver or jaundice",
            patterns=(r"\bjaundice\b", r"\bliver\b", r"\byellow eyes\b"),
            weight=1.9,
        ),
        SignalRule(
            signal="constipation",
            patterns=(r"\bconstipation\b", r"\bconstipated\b"),
            weight=1.2,
        ),
    ),
    "Orthopaedics": (
        SignalRule(
            signal="joint pain",
            patterns=(r"\bjoint pain\b", r"\bknee pain\b", r"\bhip pain\b", r"\bshoulder pain\b"),
            weight=1.9,
        ),
        SignalRule(
            signal="back pain",
            patterns=(r"\bback pain\b", r"\bspine pain\b", r"\bneck pain\b"),
            weight=1.7,
        ),
        SignalRule(
            signal="fracture or injury",
            patterns=(r"\bfracture\b", r"\bbroken bone\b", r"\bsprain\b", r"\binjury\b", r"\bfall injury\b"),
            weight=2.8,
            red_flag=True,
        ),
        SignalRule(
            signal="swelling or muscle pain",
            patterns=(r"\bswelling\b", r"\bmuscle pain\b", r"\bstiffness\b"),
            weight=1.4,
        ),
    ),
    "Paediatrics": (
        SignalRule(
            signal="child-specific symptoms",
            patterns=(r"\bchild\b", r"\bkid\b", r"\bbaby\b", r"\binfant\b", r"\btoddler\b"),
            weight=2.2,
        ),
        SignalRule(
            signal="growth or feeding issue",
            patterns=(r"\bgrowth\b", r"\bfeeding\b", r"\bvaccination\b"),
            weight=1.5,
        ),
    ),
    "Infectious Disease": (
        SignalRule(
            signal="fever",
            patterns=(r"\bfever(?:ish)?\b", r"\bhigh fever\b", r"\btemperature\b"),
            weight=2.4,
        ),
        SignalRule(
            signal="cold or viral symptoms",
            patterns=(r"\bcold\b", r"\bflu\b", r"\bviral\b", r"\binfection\b"),
            weight=1.7,
        ),
        SignalRule(
            signal="sore throat or body ache",
            patterns=(r"\bsore throat\b", r"\bbody ache\b", r"\bchills\b", r"\bfever with cold\b"),
            weight=1.8,
        ),
        SignalRule(
            signal="persistent fever",
            patterns=(r"\bpersistent fever\b", r"\bfever for\b"),
            weight=2.2,
        ),
    ),
    "Otolaryngology (ENT)": (
        SignalRule(
            signal="sore throat",
            patterns=(r"\bsore throat\b", r"\btonsil(?:s|litis)?\b", r"\bthroat pain\b"),
            weight=2.0,
        ),
        SignalRule(
            signal="ear pain",
            patterns=(r"\bear pain\b", r"\bearache\b", r"\bblocked ear\b"),
            weight=2.0,
        ),
        SignalRule(
            signal="nose or sinus congestion",
            patterns=(r"\bnose congestion\b", r"\bsinus(?: pain| infection)?\b", r"\bstuffy nose\b"),
            weight=1.8,
        ),
        SignalRule(
            signal="runny nose",
            patterns=(r"\brunny nose\b", r"\bnasal discharge\b"),
            weight=1.2,
        ),
    ),
    "Hematology": (
        SignalRule(
            signal="anemia symptoms",
            patterns=(r"\banemia\b", r"\banaemia\b", r"\blow hemoglobin\b", r"\blow hb\b", r"\bpale\b"),
            weight=2.2,
        ),
        SignalRule(
            signal="easy bruising or bleeding",
            patterns=(r"\beasy bruising\b", r"\bbruise(?:s)?\b", r"\bbleeding gums\b", r"\bnosebleed\b"),
            weight=2.4,
            red_flag=True,
        ),
        SignalRule(
            signal="blood disorder concern",
            patterns=(r"\bblood disorder\b", r"\bplatelet(?:s)?\b"),
            weight=1.9,
        ),
    ),
    "Endocrinology": (
        SignalRule(
            signal="diabetes symptoms",
            patterns=(r"\bdiabetes\b", r"\bhigh sugar\b", r"\blow sugar\b"),
            weight=2.2,
        ),
        SignalRule(
            signal="thyroid symptoms",
            patterns=(r"\bthyroid\b", r"\bneck swelling\b", r"\bgoiter\b"),
            weight=2.0,
        ),
        SignalRule(
            signal="polyuria or thirst",
            patterns=(r"\bfrequent urination\b", r"\bexcessive thirst\b"),
            weight=1.7,
        ),
    ),
    "Nephrology / Urology": (
        SignalRule(
            signal="burning urination",
            patterns=(r"\bburning urination\b", r"\bburning urine\b", r"\bpainful urination\b"),
            weight=2.0,
        ),
        SignalRule(
            signal="blood in urine",
            patterns=(r"\bblood in urine\b", r"\bred urine\b"),
            weight=2.2,
        ),
        SignalRule(
            signal="kidney or flank pain",
            patterns=(r"\bkidney pain\b", r"\bflank pain\b", r"\bkidney stone\b"),
            weight=2.2,
        ),
    ),
    GENERAL_SPECIALTY: (
        SignalRule(
            signal="fatigue or weakness",
            patterns=(r"\bfatigue\b", r"\bweakness\b", r"\bmalaise\b", r"\btired\b"),
            weight=1.2,
        ),
        SignalRule(
            signal="general body ache",
            patterns=(r"\bbody ache\b", r"\bgeneral pain\b", r"\bnot feeling well\b"),
            weight=1.0,
        ),
        SignalRule(
            signal="routine check-up",
            patterns=(r"\bcheck(?: |-)?up\b", r"\bgeneral problem\b"),
            weight=1.0,
        ),
    ),
    "Emergency Medicine": (
        SignalRule(
            signal="loss of consciousness",
            patterns=(r"\bunconscious\b", r"\bpassed out\b", r"\bfainted\b"),
            weight=4.0,
            red_flag=True,
        ),
        SignalRule(
            signal="severe bleeding",
            patterns=(r"\bsevere bleeding\b", r"\bbleeding heavily\b"),
            weight=4.0,
            red_flag=True,
        ),
        SignalRule(
            signal="stroke warning signs",
            patterns=(r"\bslurred speech\b", r"\bface droop\b", r"\bstroke\b"),
            weight=3.8,
            red_flag=True,
        ),
        SignalRule(
            signal="can not breathe",
            patterns=(r"\bcan not breathe\b", r"\bcannot breathe\b", r"\bcant breathe\b", r"\bcan't breathe\b"),
            weight=3.6,
            red_flag=True,
        ),
        SignalRule(
            signal="crushing chest pain",
            patterns=(r"\bcrushing chest pain\b", r"\bheart attack\b"),
            weight=3.6,
            red_flag=True,
        ),
    ),
}


class SpecialtyAlternative(BaseModel):
    specialist: str
    routedSpecialty: str
    confidence: float = Field(ge=0, le=1)
    matchedSignals: list[str] = Field(default_factory=list)


class SpecialtyPredictionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    symptoms: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    temperature_c: float | None = Field(default=None, ge=0)
    pain_score: float | None = Field(default=None, ge=0, le=10)
    chief_complaint_system: str | None = None
    language: str | None = None


class SpecialtyPredictionResponse(BaseModel):
    primarySpecialist: str
    routedSpecialty: str
    confidence: float = Field(ge=0, le=1)
    lowConfidence: bool
    normalizedSymptoms: str
    extractedSignals: list[str] = Field(default_factory=list)
    alternatives: list[SpecialtyAlternative] = Field(default_factory=list)
    reasoning: str
    modelSource: str = MODEL_SOURCE


def _collapse_repeated_letters(token: str) -> str:
    return re.sub(r"(.)\1+", r"\1", token)


def normalize_symptoms_text(raw_text: str) -> str:
    lowered = raw_text.lower().replace("/", " ").replace("-", " ")
    lowered = re.sub(r"[^a-z0-9\s]", " ", lowered)

    normalized_tokens: list[str] = []
    for token in lowered.split():
        collapsed = _collapse_repeated_letters(token)
        if collapsed in KNOWN_COLLAPSE_TOKENS:
            token = collapsed
        token = TOKEN_CORRECTIONS.get(token, token)
        token = TOKEN_CORRECTIONS.get(collapsed, token)
        normalized_tokens.append(token)

    normalized = re.sub(r"\s+", " ", " ".join(normalized_tokens)).strip()
    for pattern, replacement in PHRASE_REPLACEMENTS:
        normalized = pattern.sub(replacement, normalized)
    return re.sub(r"\s+", " ", normalized).strip()


def _append_unique(target: list[str], value: str) -> None:
    if value not in target:
        target.append(value)


def _record_signal(
    specialty: str,
    signal: str,
    specialty_hits: dict[str, list[str]],
    extracted_signals: list[str],
) -> None:
    _append_unique(specialty_hits[specialty], signal)
    _append_unique(extracted_signals, signal)


def _apply_score(
    specialty: str,
    increment: float,
    signal: str | None,
    scores: dict[str, float],
    specialty_hits: dict[str, list[str]],
    extracted_signals: list[str],
) -> None:
    scores[specialty] += increment
    if signal:
        _record_signal(specialty, signal, specialty_hits, extracted_signals)


def _route_specialty(primary_specialist: str, ranked_specialties: list[tuple[str, float]]) -> str:
    routed = SPECIALTY_ROUTE_MAP.get(primary_specialist, GENERAL_ROUTE)
    if routed != GENERAL_ROUTE:
        return routed

    if primary_specialist == "Emergency Medicine":
        top_score = ranked_specialties[0][1]
        for specialty, score in ranked_specialties[1:4]:
            fallback_route = SPECIALTY_ROUTE_MAP.get(specialty, GENERAL_ROUTE)
            if fallback_route != GENERAL_ROUTE and (top_score - score) <= 1.3:
                return fallback_route

    return routed


def _build_reasoning(
    primary_specialist: str,
    routed_specialty: str,
    extracted_signals: list[str],
    low_confidence: bool,
    normalized_symptoms: str,
) -> str:
    if not extracted_signals:
        reasoning = (
            "No strong specialty-specific signal was detected in the description. "
            f"Routing conservatively to {routed_specialty}."
        )
    else:
        signal_text = ", ".join(extracted_signals[:4])
        reasoning = f"Primary clinical fit is {primary_specialist} based on signals like {signal_text}."
        if routed_specialty != primary_specialist:
            reasoning += f" SmartQ maps this to {routed_specialty} for staffed routing."
        else:
            reasoning += f" SmartQ can route directly to {routed_specialty}."

    if low_confidence:
        reasoning += " Symptom overlap is high, so manual review or doctor override is recommended."
    elif normalized_symptoms != normalized_symptoms.strip():
        reasoning += " Input text was normalized before scoring."

    return reasoning


def predict_specialty(payload: SpecialtyPredictionRequest) -> SpecialtyPredictionResponse:
    normalized_symptoms = normalize_symptoms_text(payload.symptoms)
    if not normalized_symptoms:
        normalized_symptoms = payload.symptoms.strip().lower()

    scores = {specialty: 0.0 for specialty in SPECIALTY_RULES}
    specialty_hits = {specialty: [] for specialty in SPECIALTY_RULES}
    extracted_signals: list[str] = []
    red_flag_hits = 0

    for specialty, rules in SPECIALTY_RULES.items():
        for rule in rules:
            if any(re.search(pattern, normalized_symptoms) for pattern in rule.patterns):
                _apply_score(specialty, rule.weight, rule.signal, scores, specialty_hits, extracted_signals)
                if rule.red_flag:
                    red_flag_hits += 1

    complaint = (payload.chief_complaint_system or "").strip().casefold()
    for specialty, boost in CONTEXTUAL_COMPLAINT_BOOSTS.get(complaint, {}).items():
        _apply_score(
            specialty,
            boost,
            f"{complaint} complaint",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if payload.temperature_c is not None and payload.temperature_c >= 38.0:
        _apply_score(
            "Infectious Disease",
            1.1 if payload.temperature_c < 39.0 else 1.5,
            "measured fever",
            scores,
            specialty_hits,
            extracted_signals,
        )
        if payload.temperature_c >= 39.0:
            _apply_score(
                "Emergency Medicine",
                0.7,
                "high fever alert",
                scores,
                specialty_hits,
                extracted_signals,
            )

    if payload.age is not None and payload.age <= 15:
        _apply_score(
            "Paediatrics",
            2.4,
            "pediatric age",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if payload.pain_score is not None and payload.pain_score >= 7:
        _apply_score(
            "Emergency Medicine",
            0.5,
            "severe pain score",
            scores,
            specialty_hits,
            extracted_signals,
        )

    cardio_signals = set(specialty_hits["Cardiology"])
    pulm_signals = set(specialty_hits["Pulmonology"])
    neuro_signals = set(specialty_hits["Neurology"])
    infectious_signals = set(specialty_hits["Infectious Disease"])

    if {"chest pain", "shortness of breath"}.issubset(cardio_signals):
        _apply_score(
            "Cardiology",
            1.0,
            "cardiorespiratory red flag",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if "shortness of breath" in pulm_signals and "cough" in pulm_signals:
        _apply_score(
            "Pulmonology",
            0.8,
            "respiratory cluster",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if infectious_signals.intersection({"fever", "cold or viral symptoms"}) and pulm_signals.intersection(
        {"cough", "shortness of breath"}
    ):
        _apply_score(
            "Infectious Disease",
            0.6,
            "infectious respiratory overlap",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if neuro_signals.intersection({"seizure activity", "focal neurological deficit"}):
        _apply_score(
            "Emergency Medicine",
            1.1,
            "neurological emergency overlap",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if red_flag_hits:
        _apply_score(
            "Emergency Medicine",
            min(1.2, red_flag_hits * 0.35),
            "red flag escalation",
            scores,
            specialty_hits,
            extracted_signals,
        )

    if payload.age is not None and payload.age <= 15 and infectious_signals:
        _apply_score(
            "Paediatrics",
            0.7,
            "child with infectious symptoms",
            scores,
            specialty_hits,
            extracted_signals,
        )

    max_score = max(scores.values())
    if max_score < 1.0:
        _apply_score(
            GENERAL_SPECIALTY,
            0.9,
            "non-specific symptoms",
            scores,
            specialty_hits,
            extracted_signals,
        )
    else:
        scores[GENERAL_SPECIALTY] += 0.25

    ranked_specialties = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    primary_specialist, primary_score = ranked_specialties[0]
    second_score = ranked_specialties[1][1] if len(ranked_specialties) > 1 else 0.0
    routed_specialty = _route_specialty(primary_specialist, ranked_specialties)

    signal_count = len(specialty_hits[primary_specialist])
    confidence = 0.28 + min(0.50, primary_score / 12.0) + min(0.10, signal_count * 0.04)
    if (primary_score - second_score) < 0.75:
        confidence -= 0.12
    elif (primary_score - second_score) < 1.5:
        confidence -= 0.06

    if primary_score < 1.5:
        confidence -= 0.08

    confidence = max(0.18, min(0.96, confidence))
    low_confidence = confidence < LOW_CONFIDENCE_THRESHOLD or (primary_score - second_score) < 0.75

    top_ranked = [item for item in ranked_specialties if item[1] > 0.15][:4]
    if not top_ranked:
        top_ranked = [(GENERAL_SPECIALTY, scores[GENERAL_SPECIALTY])]

    score_total = sum(score for _, score in top_ranked) or 1.0
    alternatives = [
        SpecialtyAlternative(
            specialist=specialty,
            routedSpecialty=_route_specialty(specialty, ranked_specialties),
            confidence=round(score / score_total, 4),
            matchedSignals=specialty_hits[specialty][:4],
        )
        for specialty, score in top_ranked
    ]

    reasoning = _build_reasoning(
        primary_specialist=primary_specialist,
        routed_specialty=routed_specialty,
        extracted_signals=extracted_signals,
        low_confidence=low_confidence,
        normalized_symptoms=normalized_symptoms,
    )

    return SpecialtyPredictionResponse(
        primarySpecialist=primary_specialist,
        routedSpecialty=routed_specialty,
        confidence=round(confidence, 4),
        lowConfidence=low_confidence,
        normalizedSymptoms=normalized_symptoms,
        extractedSignals=extracted_signals[:8],
        alternatives=alternatives,
        reasoning=reasoning,
        modelSource=MODEL_SOURCE,
    )
