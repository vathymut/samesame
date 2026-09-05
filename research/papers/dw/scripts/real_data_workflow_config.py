"""Configuration for the OpenML-backed mirrored real-data workflow figure."""

from __future__ import annotations

from dataclasses import dataclass

WORKFLOW_ORDER: tuple[str, ...] = ("risk", "confidence", "error")
WORKFLOW_LABELS = {
    "risk": "Predicted risk",
    "confidence": "Model confidence",
    "error": "Prediction error",
}
WORKFLOW_DIRECTIONS = {
    "risk": "higher",
    "confidence": "lower",
    "error": "higher",
}


@dataclass(frozen=True, slots=True)
class TaskSpec:
    identifier: str
    label: str
    short_label: str
    domain: str
    shift_variable: str
    data_source: str
    source_definition: str
    target_definition: str
    label_definition: str
    narrative_role: str
    openml_dataset_name: str
    openml_data_id: int
    status: str = "ready"
    note: str | None = None


TASK_SPECS = {
    "heloc": TaskSpec(
        identifier="heloc",
        label="FICO HELOC",
        short_label="HELOC",
        domain="credit risk",
        shift_variable="ExternalRiskEstimate",
        data_source="OpenML heloc (46932)",
        source_definition="ExternalRiskEstimate > 63 with a stratified source holdout",
        target_definition="ExternalRiskEstimate <= 63",
        label_definition="RiskPerformance == 'Bad'",
        narrative_role="anchor",
        openml_dataset_name="heloc",
        openml_data_id=46932,
    ),
    "diabetes_readmission": TaskSpec(
        identifier="diabetes_readmission",
        label="Hospital readmission",
        short_label="Readmission",
        domain="healthcare",
        shift_variable="admission_source_id",
        data_source="OpenML Diabetes130US (46922)",
        source_definition="admission_source_id != 7 with a stratified source holdout",
        target_definition="admission_source_id == 7",
        label_definition="EarlyReadmission == 'Yes'",
        narrative_role="comparison",
        openml_dataset_name="Diabetes130US",
        openml_data_id=46922,
    ),
    "acsincome": TaskSpec(
        identifier="acsincome",
        label="ACS income",
        short_label="Income",
        domain="socioeconomic screening",
        shift_variable="DIVISION (derived from ST)",
        data_source="OpenML ACSIncome (43141)",
        source_definition="DIVISION != '01' with a stratified source holdout",
        target_definition="DIVISION == '01' (New England)",
        label_definition="PINCP <= 56000",
        narrative_role="comparison",
        openml_dataset_name="ACSIncome",
        openml_data_id=43141,
    ),
    "acspubcov": TaskSpec(
        identifier="acspubcov",
        label="ACS public coverage",
        short_label="Coverage",
        domain="public benefits screening",
        shift_variable="DIS",
        data_source="OpenML ACSPublicCoverage (43140)",
        source_definition="DIS != 1 with a stratified source holdout",
        target_definition="DIS == 1 (with a disability)",
        label_definition="PUBCOV == 1",
        narrative_role="comparison",
        openml_dataset_name="ACSPublicCoverage",
        openml_data_id=43140,
    ),
    "physionet": TaskSpec(
        identifier="physionet",
        label="PhysioNet sepsis",
        short_label="Sepsis",
        domain="critical care",
        shift_variable="ICULOS",
        data_source="OpenML physionet_sepsis (46888)",
        source_definition="ICULOS <= 47 with a stratified source holdout",
        target_definition="ICULOS > 47",
        label_definition="SepsisLabel == 1",
        narrative_role="comparison",
        openml_dataset_name="physionet_sepsis",
        openml_data_id=46888,
        status="blocked",
        note="The current OpenML physionet_sepsis artifact fails checksum validation via scikit-learn/OpenML, so this mirror is not presently executable.",
    ),
    "college_scorecard": TaskSpec(
        identifier="college_scorecard",
        label="College scorecard",
        short_label="College",
        domain="education policy",
        shift_variable="CCBASIC",
        data_source="OpenML college_scorecard (46805)",
        source_definition="CCBASIC not in the TableShift OOD institution-type set, with a stratified source holdout",
        target_definition="CCBASIC in the TableShift OOD institution-type set",
        label_definition="Completion_rate_for_first_time_full_time_target <= 0.5",
        narrative_role="comparison",
        openml_dataset_name="college_scorecard",
        openml_data_id=46805,
        status="blocked",
        note="The OpenML mirror does not expose the CCBASIC Carnegie basic classification column required to recreate the TableShift split.",
    ),
    "mimic_extract_los_3": TaskSpec(
        identifier="mimic_extract_los_3",
        label="MIMIC LOS >= 3",
        short_label="MIMIC LOS",
        domain="critical care",
        shift_variable="insurance",
        data_source="OpenML mimic_extract_los_3 (46887)",
        source_definition="insurance != Medicare with a stratified source holdout",
        target_definition="insurance == Medicare",
        label_definition="los_3 == 1",
        narrative_role="comparison",
        openml_dataset_name="mimic_extract_los_3",
        openml_data_id=46887,
        status="blocked",
        note="The current OpenML mirror exposes medication-route features instead of the los_3 target and insurance split needed to mirror TableShift.",
    ),
    "nsw": TaskSpec(
        identifier="nsw",
        label="NSW employment program",
        short_label="NSW",
        domain="employment policy",
        shift_variable="treat",
        data_source="Rdatasets MatchIt::lalonde (614 obs)",
        source_definition="treat == 0 (PSID control group, n=429)",
        target_definition="treat == 1 (NSW treatment group, n=185)",
        label_definition="re78 > 5000 (post-program earnings above ~$5K)",
        narrative_role="motivation",
        openml_dataset_name="",
        openml_data_id=0,
        status="ready",
        note="LaLonde (1986) NSW employment experiment; downloaded from Rdatasets MatchIt::lalonde",
    ),
}

INITIAL_TASK_ORDER: tuple[str, ...] = (
    "heloc",
    "diabetes_readmission",
    "acsincome",
    "acspubcov",
)

DEFAULT_SPOTLIGHT_TASK = "heloc"
