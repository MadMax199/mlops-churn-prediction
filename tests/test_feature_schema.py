from src.features.schema import FEATURE_COLUMNS, ID_COLUMN, TARGET_COLUMN


def test_feature_schema_has_no_direct_personal_identifiers():
    assert {"email", "firstname", "lastname", "address"}.isdisjoint(FEATURE_COLUMNS)


def test_identifier_and_target_are_not_features():
    assert ID_COLUMN not in FEATURE_COLUMNS
    assert TARGET_COLUMN not in FEATURE_COLUMNS
