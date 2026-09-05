"""Per-task recipes mapping OpenML datasets to source/target splits."""

from __future__ import annotations

from scripts._loaders import (
    TaskRecipe,
    derive_acs_division,
    drop_combined_columns,
    drop_located_columns,
    drop_named_columns,
    label_membership,
    label_numeric_indicator,
    label_numeric_threshold,
    source_greater_than,
    source_less_equal,
    source_not_in,
    split_column_values,
)

TASK_RECIPES: dict[str, TaskRecipe] = {
    "heloc": TaskRecipe(
        split_values=split_column_values(
            "ExternalRiskEstimate",
            "External Risk Estimate",
            numeric=True,
        ),
        label_values=label_membership("Bad"),
        source_mask=source_greater_than(63.0),
        drop_columns=drop_located_columns(
            "ExternalRiskEstimate",
            "External Risk Estimate",
        ),
        empty_split_message="HELOC split must produce non-empty source and target pools",
    ),
    "diabetes_readmission": TaskRecipe(
        split_values=split_column_values("admission_source_id"),
        label_values=label_membership("Yes"),
        source_mask=source_not_in(7, "Emergency Room"),
        drop_columns=drop_combined_columns(
            drop_named_columns("encounter_id", "patient_nbr"),
            drop_located_columns("admission_source_id"),
        ),
        empty_split_message="readmission split must produce non-empty admission_source_id pools",
    ),
    "acsincome": TaskRecipe(
        split_values=derive_acs_division,
        label_values=label_numeric_threshold(
            threshold=56_000,
            positive_when_leq=True,
        ),
        source_mask=source_not_in("01"),
        drop_columns=drop_located_columns("ST", "State", "State_postcode"),
        empty_split_message="ACS income split must produce non-empty DIVISION pools",
    ),
    "acspubcov": TaskRecipe(
        split_values=split_column_values("DIS"),
        label_values=label_numeric_indicator(1),
        source_mask=source_not_in(1, "1.0", "01", "With a disability"),
        drop_columns=drop_located_columns("DIS"),
        empty_split_message="ACS public coverage split must produce non-empty DIS pools",
    ),
    "physionet": TaskRecipe(
        split_values=split_column_values(
            "ICULOS",
            "ICU_length_of_stay__hours_since_ICU_admission",
            numeric=True,
        ),
        label_values=label_numeric_indicator(1),
        source_mask=source_less_equal(47.0),
        drop_columns=drop_combined_columns(
            drop_located_columns(
                "ICULOS",
                "ICU_length_of_stay__hours_since_ICU_admission",
            ),
            drop_named_columns("Hour"),
        ),
        empty_split_message="PhysioNet split must produce non-empty ICULOS pools",
    ),
}
