from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

try:
    from specialty_hybrid import (
        GENERAL_ROUTE,
        SpecialtyPredictionRequest,
        SpecialtyPredictionResponse,
        normalize_symptoms_text,
        predict_specialty,
    )
except ModuleNotFoundError:
    from .specialty_hybrid import (
        GENERAL_ROUTE,
        SpecialtyPredictionRequest,
        SpecialtyPredictionResponse,
        normalize_symptoms_text,
        predict_specialty,
    )


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smartq-ml-service")

CONFIDENCE_THRESHOLD = 0.60
MODEL_VERSION = "v3"
SERVICE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SERVICE_DIR.parent

RECOMMENDATION_MAP: dict[int, str] = {
    1: "Immediate — resuscitation required",
    2: "Emergency — seen within 15 minutes",
    3: "Urgent — seen within 30 minutes",
    4: "Less urgent — seen within 60 minutes",
    5: "Non-urgent — seen within 120 minutes",
}

# v3 training-set defaults for sparse requests. Numeric values are medians.
NUMERIC_DEFAULTS: dict[str, float] = {
    "news2_score": 2.0,
    "gcs_total": 15.0,
    "pain_score": 5.0,
    "spo2": 97.0,
    "temperature_c": 37.5,
    "respiratory_rate": 17.3,
    "spo2_resp_interaction": 1671.42,
    "mean_arterial_pressure": 91.9,
    "shock_index": 0.7240272613983953,
    "num_prior_ed_visits_12m": 1.0,
    "heart_rate": 89.6,
    "diastolic_bp": 75.3,
    "multi_risk_flag": 0.0,
    "systolic_bp": 123.1,
    "pulse_pressure": 47.2,
    "hypoxia_flag": 0.0,
    "height_cm": 171.1,
    "weight_kg": 76.0,
    "bmi": 26.0,
    "age": 48.0,
    "arrival_hour": 11.0,
    "num_comorbidities": 5.0,
    "num_active_medications": 4.0,
    "arrival_month": 7.0,
    "high_fever_flag": 0.0,
    "tachycardia_flag": 0.0,
    "num_prior_admissions_12m": 0.0,
}

# Defaults stay semantically neutral where possible, while remaining valid encoder classes.
CATEGORICAL_DEFAULTS: dict[str, str] = {
    "mental_status_triage": "alert",
    "chief_complaint_system": "other",
    "pain_location": "unknown",
    "arrival_day": "Monday",
    "transport_origin": "home",
    "language": "English",
    "site_id": "SITE-TUR-01",
    "arrival_mode": "walk-in",
    "arrival_season": "summer",
    "insurance_type": "public",
    "shift": "morning",
    "age_group": "middle_aged",
    "sex": "F",
}

CATEGORY_ALIASES: dict[str, dict[str, str]] = {
    "sex": {
        "female": "F",
        "f": "F",
        "male": "M",
        "m": "M",
        "other": "Other",
        "nonbinary": "Other",
        "non-binary": "Other",
    },
    "arrival_mode": {
        "walkin": "walk-in",
        "walk": "walk-in",
        "broughtbyfamily": "brought_by_family",
        "family": "brought_by_family",
    },
}


@dataclass
class ModelArtifacts:
    model: Any
    scaler: Any
    feature_columns: list[str]
    numeric_columns: list[str]
    categorical_columns: list[str]
    feature_label_encoders: dict[str, Any]
    target_encoder: Any
    numeric_defaults: dict[str, float]
    categorical_defaults: dict[str, str]
    category_lookup: dict[str, dict[str, str]]
    model_class_labels: list[int]


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    news2_score: float | None = Field(default=None, ge=0)
    gcs_total: int | None = Field(default=None, ge=0, le=15)
    pain_score: float | None = Field(default=None, ge=0, le=10)
    spo2: float | None = Field(default=None, ge=0, le=100)
    temperature_c: float | None = Field(default=None, ge=0)
    respiratory_rate: float | None = Field(default=None, ge=0)
    spo2_resp_interaction: float | None = Field(default=None, ge=0)
    mean_arterial_pressure: float | None = Field(default=None, ge=0)
    shock_index: float | None = Field(default=None, ge=0)
    num_prior_ed_visits_12m: int | None = Field(default=None, ge=0)
    heart_rate: float | None = Field(default=None, ge=0)
    mental_status_triage: str | None = None
    diastolic_bp: float | None = Field(default=None, ge=0)
    multi_risk_flag: int | None = Field(default=None, ge=0, le=1)
    systolic_bp: float | None = Field(default=None, ge=0)
    pulse_pressure: float | None = Field(default=None)
    hypoxia_flag: int | None = Field(default=None, ge=0, le=1)
    height_cm: float | None = Field(default=None, ge=0)
    weight_kg: float | None = Field(default=None, ge=0)
    bmi: float | None = Field(default=None, ge=0)
    age: int | None = Field(default=None, ge=0, le=130)
    arrival_hour: int | None = Field(default=None, ge=0, le=23)
    chief_complaint_system: str | None = None
    num_comorbidities: int | None = Field(default=None, ge=0)
    num_active_medications: int | None = Field(default=None, ge=0)
    arrival_month: int | None = Field(default=None, ge=1, le=12)
    pain_location: str | None = None
    arrival_day: str | None = None
    transport_origin: str | None = None
    high_fever_flag: int | None = Field(default=None, ge=0, le=1)
    language: str | None = None
    site_id: str | None = None
    arrival_mode: str | None = None
    arrival_season: str | None = None
    tachycardia_flag: int | None = Field(default=None, ge=0, le=1)
    insurance_type: str | None = None
    shift: str | None = None
    num_prior_admissions_12m: int | None = Field(default=None, ge=0)
    age_group: str | None = None
    sex: str | None = None


class PredictionResponse(BaseModel):
    priority_class: int = Field(ge=1, le=5)
    confidence: float = Field(ge=0, le=1)
    low_confidence: bool
    recommendation: str
    all_class_probs: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_version: str


class RootResponse(BaseModel):
    service: str
    status: str
    model_version: str
    docs_url: str
    health_url: str
    predict_url: str


class TestRecommendationRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)
    priority_class: int | None = Field(default=None, ge=1, le=5)
    chief_complaint_system: str | None = None
    age: int | None = Field(default=None, ge=0, le=130)
    temperature_c: float | None = Field(default=None, ge=0)
    spo2: float | None = Field(default=None, ge=0, le=100)
    heart_rate: float | None = Field(default=None, ge=0)
    systolic_bp: float | None = Field(default=None, ge=0)
    pain_score: float | None = Field(default=None, ge=0, le=10)
    gcs_total: int | None = Field(default=None, ge=0, le=15)
    symptoms: str | None = None


class TestRecommendation(BaseModel):
    test: str
    rationale: str
    urgency: str = Field(description="immediate | urgent | routine")


class TestRecommendationResponse(BaseModel):
    recommendations: list[TestRecommendation]
    source: str = "rule_based_v1"
    low_confidence: bool = False


class SafetyRuleMatch(BaseModel):
    ruleId: str
    severity: str
    forcedPriorityClass: int | None = Field(default=None, ge=1, le=5)
    preferredRoute: str | None = None
    rationale: str


class QueueRouteOption(BaseModel):
    route: str = Field(min_length=1)
    currentQueueLength: int = Field(default=0, ge=0)
    availableDoctors: int = Field(default=1, ge=0)
    avgWaitMinutes: float | None = Field(default=None, ge=0)
    acceptsFallback: bool = False


class QueueAssignmentResponse(BaseModel):
    selectedRoute: str
    routeType: str
    rationale: str
    currentQueueLength: int = Field(default=0, ge=0)
    availableDoctors: int = Field(default=0, ge=0)
    avgWaitMinutes: float | None = Field(default=None, ge=0)


class PriorityPipelineResponse(BaseModel):
    modelPriorityClass: int | None = Field(default=None, ge=1, le=5)
    modelConfidence: float = Field(ge=0, le=1)
    lowConfidence: bool
    modelRecommendation: str
    allClassProbs: dict[str, float] = Field(default_factory=dict)
    guardrailedPriorityClass: int = Field(ge=1, le=5)
    guardrailedRecommendation: str
    source: str


class PatientFlowRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", str_strip_whitespace=True)

    symptoms: str = Field(min_length=1)
    age: int | None = Field(default=None, ge=0, le=130)
    sex: str | None = None
    mental_status_triage: str | None = None
    chief_complaint_system: str | None = None
    language: str | None = None
    temperature_c: float | None = Field(default=None, ge=0)
    pain_score: float | None = Field(default=None, ge=0, le=10)
    spo2: float | None = Field(default=None, ge=0, le=100)
    respiratory_rate: float | None = Field(default=None, ge=0)
    heart_rate: float | None = Field(default=None, ge=0)
    systolic_bp: float | None = Field(default=None, ge=0)
    diastolic_bp: float | None = Field(default=None, ge=0)
    gcs_total: int | None = Field(default=None, ge=0, le=15)
    news2_score: float | None = Field(default=None, ge=0)
    availableRoutes: list[QueueRouteOption] = Field(default_factory=list)


class PatientFlowResponse(BaseModel):
    normalizedSymptoms: str
    derivedChiefComplaintSystem: str
    safety: list[SafetyRuleMatch] = Field(default_factory=list)
    priority: PriorityPipelineResponse
    specialty: SpecialtyPredictionResponse
    queueAssignment: QueueAssignmentResponse
    tests: TestRecommendationResponse


_COMPLAINT_TESTS: dict[str, list[dict]] = {
    "respiratory": [
        {"test": "Chest X-ray", "rationale": "Assess lung fields and cardiac silhouette", "urgency": "urgent"},
        {"test": "SpO2 / ABG", "rationale": "Quantify hypoxia severity", "urgency": "immediate"},
        {"test": "CBC", "rationale": "Detect infection or anaemia", "urgency": "urgent"},
    ],
    "cardiac": [
        {"test": "12-lead ECG", "rationale": "Rule out ST-elevation MI and arrhythmia", "urgency": "immediate"},
        {"test": "Troponin I/T", "rationale": "Biomarker for myocardial injury", "urgency": "immediate"},
        {"test": "Chest X-ray", "rationale": "Assess cardiac size and pulmonary oedema", "urgency": "urgent"},
    ],
    "neurological": [
        {"test": "Non-contrast CT Head", "rationale": "Exclude haemorrhage or mass", "urgency": "immediate"},
        {"test": "Blood glucose", "rationale": "Rule out hypoglycaemia mimicking stroke", "urgency": "immediate"},
        {"test": "CBC + CRP", "rationale": "Screen for infectious cause", "urgency": "urgent"},
    ],
    "gastrointestinal": [
        {"test": "Liver function tests", "rationale": "Assess hepatobiliary involvement", "urgency": "urgent"},
        {"test": "Serum amylase / lipase", "rationale": "Screen for pancreatitis", "urgency": "urgent"},
        {"test": "Abdominal ultrasound", "rationale": "Visualise solid organs and free fluid", "urgency": "routine"},
    ],
    "trauma": [
        {"test": "Full trauma series X-rays", "rationale": "Identify fractures and internal injury", "urgency": "immediate"},
        {"test": "FAST ultrasound", "rationale": "Detect haemoperitoneum", "urgency": "immediate"},
        {"test": "CBC + coagulation screen", "rationale": "Baseline haematology post-trauma", "urgency": "urgent"},
    ],
    "renal": [
        {"test": "Urine dipstick / microscopy", "rationale": "Detect infection, blood, or casts", "urgency": "urgent"},
        {"test": "Serum creatinine / eGFR", "rationale": "Assess renal function", "urgency": "urgent"},
        {"test": "Renal ultrasound", "rationale": "Exclude obstruction", "urgency": "routine"},
    ],
    "endocrine": [
        {"test": "Blood glucose (capillary + venous)", "rationale": "Diagnose hyper/hypoglycaemia", "urgency": "immediate"},
        {"test": "HbA1c", "rationale": "Long-term glycaemic control assessment", "urgency": "routine"},
        {"test": "Thyroid function tests", "rationale": "Rule out thyroid emergency", "urgency": "urgent"},
    ],
}

_HIGH_FEVER_TESTS = [
    {"test": "Blood cultures (x2)", "rationale": "Identify bacteraemia / sepsis source", "urgency": "immediate"},
    {"test": "CBC + CRP + procalcitonin", "rationale": "Sepsis work-up", "urgency": "immediate"},
    {"test": "Urinalysis + urine culture", "rationale": "Common infection source", "urgency": "urgent"},
]

_SEVERE_PAIN_TESTS = [
    {"test": "Serum lactate", "rationale": "Assess tissue perfusion in severe pain states", "urgency": "urgent"},
]

_PEDIATRIC_TESTS = [
    {"test": "Peripheral blood smear", "rationale": "Paediatric infectious and haematological screen", "urgency": "urgent"},
]

_ELDERLY_VITALS_TESTS = [
    {"test": "Electrolytes + renal panel", "rationale": "Elderly patients at risk of electrolyte disturbance", "urgency": "urgent"},
]

SYMPTOM_COMPLAINT_HINTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("cardiac", ("chest pain", "chest pressure", "palpitation", "heart pain", "heart attack")),
    ("respiratory", ("shortness of breath", "breath", "cough", "wheez", "phlegm")),
    ("neurological", ("headache", "dizz", "seizure", "stroke", "numb", "tingling", "face droop")),
    ("gastrointestinal", ("abdominal", "stomach", "vomit", "nausea", "diarrhea", "constipat")),
    ("trauma", ("fracture", "injury", "fall", "sprain", "bleeding", "trauma")),
    ("renal", ("urination", "urine", "kidney", "flank")),
    ("endocrine", ("diabetes", "sugar", "thyroid", "thirst")),
    ("dermatological", ("rash", "itch", "eczema", "skin", "hives")),
)

SPECIALTY_TO_COMPLAINT_MAP: dict[str, str] = {
    "Cardiology": "cardiac",
    "Pulmonology": "respiratory",
    "Infectious Disease": "respiratory",
    "Neurology": "neurological",
    "Gastroenterology": "gastrointestinal",
    "Orthopaedics": "trauma",
    "Dermatology": "dermatological",
    "Nephrology / Urology": "renal",
    "Endocrinology": "endocrine",
    "Paediatrics": "respiratory",
}

FALLBACK_ROUTE_NAMES = {"general opd", "general practice", "general"}


def _deduplicate(recs: list[dict]) -> list[TestRecommendation]:
    seen: set[str] = set()
    out: list[TestRecommendation] = []
    for recommendation in recs:
        key = recommendation["test"].lower()
        if key not in seen:
            seen.add(key)
            out.append(TestRecommendation(**recommendation))
    return out


def generate_test_recommendations(payload: TestRecommendationRequest) -> TestRecommendationResponse:
    recs: list[dict] = []

    complaint = (payload.chief_complaint_system or "").lower().strip()
    if complaint in _COMPLAINT_TESTS:
        recs.extend(_COMPLAINT_TESTS[complaint])

    if payload.temperature_c is not None and payload.temperature_c >= 38.5:
        recs.extend(_HIGH_FEVER_TESTS)

    if payload.pain_score is not None and payload.pain_score >= 7:
        recs.extend(_SEVERE_PAIN_TESTS)

    if payload.age is not None and payload.age <= 15:
        recs.extend(_PEDIATRIC_TESTS)

    if payload.age is not None and payload.age > 65:
        recs.extend(_ELDERLY_VITALS_TESTS)

    if payload.priority_class is not None and payload.priority_class <= 2:
        recs = [
            {
                "test": "ABG (arterial blood gas)",
                "rationale": "Critical patient - assess oxygenation and acid-base",
                "urgency": "immediate",
            },
            {
                "test": "Stat CBC + CMP + coagulation",
                "rationale": "Baseline panel for critical triage",
                "urgency": "immediate",
            },
            *recs,
        ]

    if not recs:
        recs = [
            {"test": "CBC", "rationale": "Standard baseline haematology", "urgency": "routine"},
            {"test": "Metabolic panel", "rationale": "Baseline chemistry screen", "urgency": "routine"},
        ]

    return TestRecommendationResponse(
        recommendations=_deduplicate(recs),
        source="rule_based_v1",
        low_confidence=len(recs) < 2,
    )


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    lowered = text.casefold()
    return any(term in lowered for term in terms)


def evaluate_safety_rules(payload: PatientFlowRequest, normalized_symptoms: str) -> list[SafetyRuleMatch]:
    matches: list[SafetyRuleMatch] = []
    mental_status = (payload.mental_status_triage or "").strip().casefold()

    if payload.spo2 is not None and payload.spo2 <= 90:
        matches.append(
            SafetyRuleMatch(
                ruleId="critical_hypoxia",
                severity="critical",
                forcedPriorityClass=1,
                preferredRoute="Pulmonology",
                rationale="SpO2 at or below 90 suggests immediate respiratory compromise.",
            )
        )
    elif payload.spo2 is not None and payload.spo2 <= 93:
        matches.append(
            SafetyRuleMatch(
                ruleId="moderate_hypoxia",
                severity="urgent",
                forcedPriorityClass=2,
                preferredRoute="Pulmonology",
                rationale="SpO2 at or below 93 warrants urgent respiratory review.",
            )
        )

    if payload.gcs_total is not None and payload.gcs_total <= 8:
        matches.append(
            SafetyRuleMatch(
                ruleId="severely_reduced_consciousness",
                severity="critical",
                forcedPriorityClass=1,
                preferredRoute="Neurology",
                rationale="GCS 8 or below indicates a high-risk altered consciousness state.",
            )
        )
    elif (payload.gcs_total is not None and payload.gcs_total <= 12) or mental_status in {"unresponsive", "drowsy"}:
        matches.append(
            SafetyRuleMatch(
                ruleId="altered_mental_status",
                severity="urgent",
                forcedPriorityClass=2,
                preferredRoute="Neurology",
                rationale="Reduced GCS or abnormal mental status requires urgent escalation.",
            )
        )

    if _contains_any(
        normalized_symptoms,
        ("slurred speech", "face droop", "one sided weakness", "stroke", "paralysis"),
    ):
        matches.append(
            SafetyRuleMatch(
                ruleId="possible_stroke",
                severity="critical",
                forcedPriorityClass=1,
                preferredRoute="Neurology",
                rationale="Symptoms match a possible stroke pattern.",
            )
        )

    if _contains_any(
        normalized_symptoms,
        ("unconscious", "passed out", "fainted", "seizure", "convulsion"),
    ):
        matches.append(
            SafetyRuleMatch(
                ruleId="loss_of_consciousness_or_seizure",
                severity="critical",
                forcedPriorityClass=1,
                preferredRoute="Neurology",
                rationale="Loss of consciousness or seizure activity is a high-acuity emergency.",
            )
        )

    if _contains_any(
        normalized_symptoms,
        ("can not breathe", "cannot breathe", "cant breathe", "can't breathe", "shortness of breath"),
    ) and (
        (payload.respiratory_rate is not None and payload.respiratory_rate >= 24)
        or (payload.spo2 is not None and payload.spo2 <= 93)
    ):
        matches.append(
            SafetyRuleMatch(
                ruleId="respiratory_distress",
                severity="critical" if (payload.spo2 is not None and payload.spo2 <= 90) else "urgent",
                forcedPriorityClass=1 if (payload.spo2 is not None and payload.spo2 <= 90) else 2,
                preferredRoute="Pulmonology",
                rationale="Breathing difficulty plus abnormal respiratory vitals suggests acute respiratory distress.",
            )
        )

    if _contains_any(
        normalized_symptoms,
        ("crushing chest pain", "heart attack", "chest pain", "chest pressure"),
    ) and (
        _contains_any(normalized_symptoms, ("shortness of breath", "palpitations", "heart racing"))
        or (payload.pain_score is not None and payload.pain_score >= 7)
    ):
        matches.append(
            SafetyRuleMatch(
                ruleId="cardiac_red_flag",
                severity="urgent",
                forcedPriorityClass=2,
                preferredRoute="Cardiology",
                rationale="Chest pain with cardiopulmonary overlap needs urgent cardiac evaluation.",
            )
        )

    if payload.temperature_c is not None and payload.temperature_c >= 39.5 and (
        (payload.heart_rate is not None and payload.heart_rate >= 110)
        or (payload.systolic_bp is not None and payload.systolic_bp < 95)
    ):
        matches.append(
            SafetyRuleMatch(
                ruleId="possible_sepsis",
                severity="urgent",
                forcedPriorityClass=2,
                preferredRoute="General OPD",
                rationale="High fever plus unstable vitals suggests possible sepsis physiology.",
            )
        )

    return matches


def resolve_guardrailed_priority(
    model_prediction: PredictionResponse,
    safety_matches: list[SafetyRuleMatch],
) -> PriorityPipelineResponse:
    guardrailed_class = model_prediction.priority_class
    source = "ml_v3"

    forced_classes = [match.forcedPriorityClass for match in safety_matches if match.forcedPriorityClass is not None]
    if forced_classes:
        strongest_override = min(forced_classes)
        if strongest_override < guardrailed_class:
            guardrailed_class = strongest_override
            source = "ml_v3_guardrailed"

    return PriorityPipelineResponse(
        modelPriorityClass=model_prediction.priority_class,
        modelConfidence=model_prediction.confidence,
        lowConfidence=model_prediction.low_confidence,
        modelRecommendation=model_prediction.recommendation,
        allClassProbs=model_prediction.all_class_probs,
        guardrailedPriorityClass=guardrailed_class,
        guardrailedRecommendation=RECOMMENDATION_MAP[guardrailed_class],
        source=source,
    )


def infer_chief_complaint_system(payload: PatientFlowRequest, normalized_symptoms: str) -> str:
    if payload.chief_complaint_system:
        return payload.chief_complaint_system.strip().casefold()

    for complaint, terms in SYMPTOM_COMPLAINT_HINTS:
        if _contains_any(normalized_symptoms, terms):
            return complaint

    return "other"


def resolve_test_complaint_system(
    derived_complaint: str,
    payload: PatientFlowRequest,
    specialty_prediction: SpecialtyPredictionResponse,
) -> str:
    if payload.chief_complaint_system:
        return payload.chief_complaint_system.strip().casefold()
    if derived_complaint != "other":
        return derived_complaint
    return SPECIALTY_TO_COMPLAINT_MAP.get(specialty_prediction.primarySpecialist, "other")


def select_route_hint(
    specialty_prediction: SpecialtyPredictionResponse,
    safety_matches: list[SafetyRuleMatch],
) -> tuple[str, str]:
    preferred_route = specialty_prediction.routedSpecialty or GENERAL_ROUTE
    safety_routes = [match.preferredRoute for match in safety_matches if match.preferredRoute]

    if not safety_routes:
        return preferred_route, "primary"

    if any(match.severity == "critical" for match in safety_matches):
        return GENERAL_ROUTE, "safety_override"

    if preferred_route in safety_routes:
        return preferred_route, "safety_override"

    most_common_route = Counter(safety_routes).most_common(1)[0][0]
    return most_common_route, "safety_override"


def assign_queue_route(
    payload: PatientFlowRequest,
    priority: PriorityPipelineResponse,
    specialty_prediction: SpecialtyPredictionResponse,
    safety_matches: list[SafetyRuleMatch],
) -> QueueAssignmentResponse:
    route_hint, route_type_base = select_route_hint(specialty_prediction, safety_matches)

    if not payload.availableRoutes:
        rationale = (
            "No live queue-state context was supplied, so the clinically preferred route was returned directly."
        )
        return QueueAssignmentResponse(
            selectedRoute=route_hint,
            routeType=route_type_base,
            rationale=rationale,
            currentQueueLength=0,
            availableDoctors=0,
            avgWaitMinutes=None,
        )

    def route_key(value: str) -> str:
        return value.strip().casefold()

    preferred_key = route_key(route_hint)
    exact_matches = [option for option in payload.availableRoutes if route_key(option.route) == preferred_key]
    fallback_routes = [
        option
        for option in payload.availableRoutes
        if option.acceptsFallback or route_key(option.route) in FALLBACK_ROUTE_NAMES
    ]

    candidates = exact_matches + [
        option for option in fallback_routes if route_key(option.route) != preferred_key
    ]
    if not candidates:
        candidates = payload.availableRoutes

    staffed_candidates = [option for option in candidates if option.availableDoctors > 0]
    if staffed_candidates:
        candidates = staffed_candidates

    priority_weight = 1.3 if priority.guardrailedPriorityClass <= 2 else 1.0

    def candidate_score(option: QueueRouteOption) -> float:
        wait = option.avgWaitMinutes if option.avgWaitMinutes is not None else option.currentQueueLength * 8.0
        route_penalty = 0.0 if route_key(option.route) == preferred_key else 6.0
        doctor_bonus = min(option.availableDoctors, 3) * 2.5
        return (wait * priority_weight) + (option.currentQueueLength * 1.5) + route_penalty - doctor_bonus

    selected = min(candidates, key=candidate_score)
    estimated_wait = selected.avgWaitMinutes if selected.avgWaitMinutes is not None else float(selected.currentQueueLength * 8)
    route_type = route_type_base if route_key(selected.route) == preferred_key else "fallback"
    rationale = (
        f"Selected {selected.route} using route hint {route_hint}; "
        f"queue={selected.currentQueueLength}, doctors={selected.availableDoctors}, "
        f"estimated_wait={estimated_wait:.0f}m."
    )

    return QueueAssignmentResponse(
        selectedRoute=selected.route,
        routeType=route_type,
        rationale=rationale,
        currentQueueLength=selected.currentQueueLength,
        availableDoctors=selected.availableDoctors,
        avgWaitMinutes=selected.avgWaitMinutes,
    )


def build_priority_request(
    payload: PatientFlowRequest,
    chief_complaint_system: str,
) -> PredictionRequest:
    return PredictionRequest(
        news2_score=payload.news2_score,
        gcs_total=payload.gcs_total,
        pain_score=payload.pain_score,
        spo2=payload.spo2,
        temperature_c=payload.temperature_c,
        respiratory_rate=payload.respiratory_rate,
        heart_rate=payload.heart_rate,
        mental_status_triage=payload.mental_status_triage,
        diastolic_bp=payload.diastolic_bp,
        systolic_bp=payload.systolic_bp,
        age=payload.age,
        chief_complaint_system=chief_complaint_system,
        language=payload.language,
        sex=payload.sex,
    )


def run_patient_flow(payload: PatientFlowRequest, artifacts: ModelArtifacts) -> PatientFlowResponse:
    normalized_symptoms = normalize_symptoms_text(payload.symptoms)
    if not normalized_symptoms:
        normalized_symptoms = payload.symptoms.strip().lower()

    safety_matches = evaluate_safety_rules(payload, normalized_symptoms)

    derived_complaint = infer_chief_complaint_system(payload, normalized_symptoms)

    priority_request = build_priority_request(payload, derived_complaint)
    priority_frame = build_feature_frame(priority_request, artifacts)
    model_priority = run_inference(priority_frame, artifacts)
    guardrailed_priority = resolve_guardrailed_priority(model_priority, safety_matches)

    specialty_request = SpecialtyPredictionRequest(
        symptoms=payload.symptoms,
        age=payload.age,
        sex=payload.sex,
        temperature_c=payload.temperature_c,
        pain_score=payload.pain_score,
        chief_complaint_system=derived_complaint,
        language=payload.language,
    )
    specialty_prediction = predict_specialty(specialty_request)

    queue_assignment = assign_queue_route(payload, guardrailed_priority, specialty_prediction, safety_matches)

    complaint = resolve_test_complaint_system(derived_complaint, payload, specialty_prediction)
    tests_request = TestRecommendationRequest(
        priority_class=guardrailed_priority.guardrailedPriorityClass,
        chief_complaint_system=complaint,
        age=payload.age,
        temperature_c=payload.temperature_c,
        spo2=payload.spo2,
        heart_rate=payload.heart_rate,
        systolic_bp=payload.systolic_bp,
        pain_score=payload.pain_score,
        gcs_total=payload.gcs_total,
        symptoms=payload.symptoms,
    )
    test_recommendations = generate_test_recommendations(tests_request)

    return PatientFlowResponse(
        normalizedSymptoms=normalized_symptoms,
        derivedChiefComplaintSystem=derived_complaint,
        safety=safety_matches,
        priority=guardrailed_priority,
        specialty=specialty_prediction,
        queueAssignment=queue_assignment,
        tests=test_recommendations,
    )


def canonicalize_category_token(value: str) -> str:
    return "".join(ch for ch in value.strip().casefold() if ch.isalnum())


def resolve_artifact_path(filename: str) -> Path:
    candidates = [
        # New structure: models/triage_v3/model/ (versioned artifact location)
        SERVICE_DIR / "models" / "triage_v3" / "model" / filename,
        # Fallback to models/ (for backward compatibility)
        SERVICE_DIR / "models" / filename,
        # Legacy: project root
        PROJECT_ROOT / filename,
        # Legacy: ml_service/models/
        PROJECT_ROOT / "ml_service" / "models" / filename,
    ]
    for path in candidates:
        if path.exists():
            logger.info(f"Found artifact at: {path}")
            return path
    raise FileNotFoundError(f"Could not locate required artifact: {filename}")


def coerce_float(value: Any, default: float) -> float:
    if value is None:
        return float(default)
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid numeric value: {value}") from exc


def derive_age_group(age: float) -> str:
    if age <= 15:
        return "pediatric"
    if age <= 39:
        return "young_adult"
    if age <= 64:
        return "middle_aged"
    return "elderly"


def derive_shift(arrival_hour: float) -> str:
    hour = int(round(arrival_hour)) % 24
    if 0 <= hour <= 5:
        return "night"
    if 6 <= hour <= 13:
        return "morning"
    if 14 <= hour <= 19:
        return "afternoon"
    return "evening"


def derive_arrival_season(arrival_month: float) -> str:
    month = int(round(arrival_month))
    if month in {12, 1, 2}:
        return "winter"
    if month in {3, 4, 5}:
        return "spring"
    if month in {6, 7, 8}:
        return "summer"
    return "autumn"


def build_category_lookup(feature_label_encoders: dict[str, Any]) -> dict[str, dict[str, str]]:
    lookup: dict[str, dict[str, str]] = {}
    for field_name, encoder in feature_label_encoders.items():
        field_lookup = {
            canonicalize_category_token(str(category)): str(category)
            for category in encoder.classes_
        }
        for alias, canonical in CATEGORY_ALIASES.get(field_name, {}).items():
            field_lookup[canonicalize_category_token(alias)] = canonical
        lookup[field_name] = field_lookup
    return lookup


def load_artifacts() -> ModelArtifacts:
    model_path = resolve_artifact_path("triage_model_v3.pkl")
    scaler_path = resolve_artifact_path("scaler_v3.pkl")
    feature_path = resolve_artifact_path("feature_cols_v3.pkl")

    bundle = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    feature_columns = list(joblib.load(feature_path))

    required_bundle_keys = {
        "model",
        "selected_features",
        "numeric_columns",
        "categorical_columns",
        "feature_label_encoders",
        "target_encoder",
    }
    missing_keys = required_bundle_keys - set(bundle)
    if missing_keys:
        raise RuntimeError(f"Model bundle is missing keys: {sorted(missing_keys)}")

    selected_features = list(bundle["selected_features"])
    if feature_columns != selected_features:
        raise RuntimeError("feature_cols_v3.pkl does not match the model bundle feature order")

    numeric_columns = list(bundle["numeric_columns"])
    categorical_columns = list(bundle["categorical_columns"])
    if scaler.n_features_in_ != len(numeric_columns):
        raise RuntimeError(
            "Scaler feature count does not match numeric feature count "
            f"({scaler.n_features_in_} != {len(numeric_columns)})"
        )

    target_encoder = bundle["target_encoder"]
    model = bundle["model"]
    encoded_classes = np.asarray(model.classes_, dtype=int)
    model_class_labels = [int(label) for label in target_encoder.inverse_transform(encoded_classes)]

    artifacts = ModelArtifacts(
        model=model,
        scaler=scaler,
        feature_columns=feature_columns,
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        feature_label_encoders=bundle["feature_label_encoders"],
        target_encoder=target_encoder,
        numeric_defaults=dict(NUMERIC_DEFAULTS),
        categorical_defaults=dict(CATEGORICAL_DEFAULTS),
        category_lookup=build_category_lookup(bundle["feature_label_encoders"]),
        model_class_labels=model_class_labels,
    )

    logger.info(
        "Loaded SmartQ triage artifacts from %s with %d features",
        model_path,
        len(feature_columns),
    )
    return artifacts


def normalize_categorical_value(field_name: str, raw_value: Any, artifacts: ModelArtifacts) -> str:
    encoder = artifacts.feature_label_encoders[field_name]
    fallback = artifacts.categorical_defaults.get(field_name, str(encoder.classes_[0]))

    if raw_value is None:
        candidate = fallback
    else:
        candidate = str(raw_value).strip()
        if not candidate:
            candidate = fallback

    if candidate in encoder.classes_:
        return candidate

    canonical = canonicalize_category_token(candidate)
    lookup = artifacts.category_lookup[field_name]
    matched = lookup.get(canonical)
    if matched is not None and matched in encoder.classes_:
        return matched

    return fallback if fallback in encoder.classes_ else str(encoder.classes_[0])


def apply_engineered_features(row: dict[str, Any], artifacts: ModelArtifacts) -> None:
    age = coerce_float(row.get("age"), artifacts.numeric_defaults["age"])
    arrival_hour = coerce_float(row.get("arrival_hour"), artifacts.numeric_defaults["arrival_hour"])
    arrival_month = coerce_float(row.get("arrival_month"), artifacts.numeric_defaults["arrival_month"])
    heart_rate = coerce_float(row.get("heart_rate"), artifacts.numeric_defaults["heart_rate"])
    systolic_bp = coerce_float(row.get("systolic_bp"), artifacts.numeric_defaults["systolic_bp"])
    diastolic_bp = coerce_float(row.get("diastolic_bp"), artifacts.numeric_defaults["diastolic_bp"])
    spo2 = coerce_float(row.get("spo2"), artifacts.numeric_defaults["spo2"])
    respiratory_rate = coerce_float(
        row.get("respiratory_rate"),
        artifacts.numeric_defaults["respiratory_rate"],
    )
    temperature_c = coerce_float(
        row.get("temperature_c"),
        artifacts.numeric_defaults["temperature_c"],
    )
    weight_kg = coerce_float(row.get("weight_kg"), artifacts.numeric_defaults["weight_kg"])
    height_cm = coerce_float(row.get("height_cm"), artifacts.numeric_defaults["height_cm"])

    row["age_group"] = derive_age_group(age)
    row["shift"] = derive_shift(arrival_hour)
    row["arrival_season"] = derive_arrival_season(arrival_month)
    row["mean_arterial_pressure"] = (systolic_bp + (2 * diastolic_bp)) / 3
    row["pulse_pressure"] = systolic_bp - diastolic_bp
    row["shock_index"] = (
        heart_rate / systolic_bp if systolic_bp > 0 else artifacts.numeric_defaults["shock_index"]
    )
    row["spo2_resp_interaction"] = spo2 * respiratory_rate
    row["hypoxia_flag"] = 1.0 if spo2 < 94 else 0.0
    row["high_fever_flag"] = 1.0 if temperature_c > 38.5 else 0.0
    row["tachycardia_flag"] = 1.0 if heart_rate > 100 else 0.0
    low_bp_flag = 1.0 if systolic_bp < 90 else 0.0
    elderly_flag = 1.0 if age > 65 else 0.0

    if height_cm > 0:
        row["bmi"] = weight_kg / ((height_cm / 100) ** 2)

    row["multi_risk_flag"] = (
        row["high_fever_flag"]
        + low_bp_flag
        + row["tachycardia_flag"]
        + row["hypoxia_flag"]
        + elderly_flag
    )


def build_feature_frame(payload: PredictionRequest, artifacts: ModelArtifacts) -> pd.DataFrame:
    row: dict[str, Any] = {}
    payload_data = payload.model_dump(exclude_none=True)

    for feature in artifacts.feature_columns:
        if feature in payload_data:
            row[feature] = payload_data[feature]
        elif feature in artifacts.numeric_columns:
            row[feature] = artifacts.numeric_defaults.get(feature, 0.0)
        else:
            row[feature] = artifacts.categorical_defaults.get(feature)

    apply_engineered_features(row, artifacts)

    ordered_row: dict[str, Any] = {}
    for feature in artifacts.feature_columns:
        if feature in artifacts.numeric_columns:
            default = artifacts.numeric_defaults.get(feature, 0.0)
            ordered_row[feature] = coerce_float(row.get(feature), default)
        else:
            ordered_row[feature] = normalize_categorical_value(feature, row.get(feature), artifacts)

    frame = pd.DataFrame([ordered_row], columns=artifacts.feature_columns)

    for column in artifacts.categorical_columns:
        encoder = artifacts.feature_label_encoders[column]
        frame[column] = encoder.transform(frame[column].astype(str))

    frame[artifacts.numeric_columns] = frame[artifacts.numeric_columns].astype(float)
    frame[artifacts.numeric_columns] = artifacts.scaler.transform(frame[artifacts.numeric_columns])
    return frame


def run_inference(frame: pd.DataFrame, artifacts: ModelArtifacts) -> PredictionResponse:
    encoded_prediction = int(artifacts.model.predict(frame)[0])
    class_probabilities = artifacts.model.predict_proba(frame)[0]
    priority_class = int(artifacts.target_encoder.inverse_transform([encoded_prediction])[0])
    confidence = float(np.max(class_probabilities))
    all_class_probs = {
        str(class_label): round(float(probability), 4)
        for class_label, probability in zip(artifacts.model_class_labels, class_probabilities)
    }

    return PredictionResponse(
        priority_class=priority_class,
        confidence=round(confidence, 4),
        low_confidence=confidence < CONFIDENCE_THRESHOLD,
        recommendation=RECOMMENDATION_MAP[priority_class],
        all_class_probs=all_class_probs,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.artifacts = load_artifacts()
    yield


app = FastAPI(
    title="SmartQ Triage ML Service",
    version=MODEL_VERSION,
    lifespan=lifespan,
)


def get_artifacts(request: Request) -> ModelArtifacts:
    artifacts = getattr(request.app.state, "artifacts", None)
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model artifacts are not loaded")
    return artifacts


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", model_version=MODEL_VERSION)


@app.get("/", response_model=RootResponse)
async def root() -> RootResponse:
    return RootResponse(
        service="SmartQ Triage ML Service",
        status="ok",
        model_version=MODEL_VERSION,
        docs_url="/docs",
        health_url="/health",
        predict_url="/predict",
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(payload: PredictionRequest, request: Request) -> PredictionResponse:
    artifacts = get_artifacts(request)
    try:
        frame = build_feature_frame(payload, artifacts)
        return run_inference(frame, artifacts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed") from exc


@app.post("/specialty", response_model=SpecialtyPredictionResponse)
async def specialty(payload: SpecialtyPredictionRequest) -> SpecialtyPredictionResponse:
    try:
        return predict_specialty(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Specialty prediction failed")
        raise HTTPException(status_code=500, detail="Specialty prediction failed") from exc


@app.post("/patient-flow", response_model=PatientFlowResponse)
async def patient_flow(payload: PatientFlowRequest, request: Request) -> PatientFlowResponse:
    artifacts = get_artifacts(request)
    try:
        return run_patient_flow(payload, artifacts)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Patient flow orchestration failed")
        raise HTTPException(status_code=500, detail="Patient flow orchestration failed") from exc


@app.post("/test-recommendations", response_model=TestRecommendationResponse)
async def test_recommendations(payload: TestRecommendationRequest) -> TestRecommendationResponse:
    """
    Rule-based test recommendation engine.

    Accepts the same clinical fields used by /predict (all optional) and returns
    a prioritised list of diagnostic tests with urgency labels. The rule engine
    is the baseline; a supervised model can be hot-swapped in later by replacing
    the generate_test_recommendations() call.
    """
    try:
        return generate_test_recommendations(payload)
    except Exception as exc:
        logger.exception("Test recommendation failed")
        raise HTTPException(status_code=500, detail="Recommendation engine error") from exc


@app.get("/playground", response_class=HTMLResponse)
async def playground():
    """
    Serves the clinical testing playground UI.
    """
    html_path = SERVICE_DIR / "static" / "index.html"
    if not html_path.exists():
        raise HTTPException(status_code=404, detail="Playground UI file not found")
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))

# Mount the static directory for serving playground CSS/JS
app.mount("/static", StaticFiles(directory=SERVICE_DIR / "static"), name="static")
