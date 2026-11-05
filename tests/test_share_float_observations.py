from __future__ import annotations

import hashlib
import json
import socket

import pandas as pd
import pytest

from ashare_quant.data.datasets import get_dataset_spec
from ashare_quant.data.exceptions import DataValidationError
from ashare_quant.data.share_float_observations import capture_observations
from ashare_quant.data.storage import ParquetDataStore
from ashare_quant.data.validation import DataValidator


@pytest.fixture(autouse=True)
def no_network(monkeypatch) -> None:
    def reject(*args, **kwargs):
        raise AssertionError("network is forbidden in observation tests")

    monkeypatch.setattr(socket.socket, "connect", reject)
    monkeypatch.setattr(socket, "create_connection", reject)


def observations() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ts_code": ["874865", "874865"],
            "ann_date": ["20261104", "20261104"],
            "float_date": ["20271108", "20271108"],
            "holder_name": ["synthetic holder", "synthetic holder"],
            "share_type": ["strategic placement", "strategic placement"],
            "float_share": [843170.0, 489125.0],
            "float_ratio": [1.1094, 0.6436],
        }
    )


def test_same_business_key_retains_all_observations_without_winner(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    original = observations()
    store.write(spec, original)
    store.write(spec, original.iloc[::-1])
    stored = store.read_dataset(spec)
    assert len(stored) == 2
    assert set(stored.float_share) == {843170.0, 489125.0}
    assert stored.observation_hash.nunique() == 2
    assert set(stored.identity_status) == {"AMBIGUOUS"}
    assert set(stored.identity_variant_count) == {2}
    for row in stored.to_dict("records"):
        assert json.loads(row["raw_row_json"])["float_share"] == row["float_share"]
    result = DataValidator(store).validate_dataset(spec)
    assert result.ok
    assert any("AMBIGUOUS" in warning for warning in result.warnings)


def test_new_observation_preserves_previous_value_and_marks_both(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    frame = observations()
    store.write(spec, frame.iloc[:1])
    first_hash = store.read_dataset(spec).observation_hash.iloc[0]
    store.write(spec, frame.iloc[1:])
    stored = store.read_dataset(spec)
    assert first_hash in set(stored.observation_hash)
    assert set(stored.identity_status) == {"AMBIGUOUS"}
    assert set(stored.identity_variant_count) == {2}


def test_legacy_read_does_not_write_and_merge_preserves_legacy_row(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    path = store.dataset_dir(spec) / "year=2026/month=11/data.parquet"
    path.parent.mkdir(parents=True)
    observations().iloc[:1].to_parquet(path, index=False)
    before = path.read_bytes()
    legacy = store.read_dataset(spec)
    assert legacy.legacy_provenance.all()
    assert set(legacy.identity_status) == {"SINGLE_OBSERVATION"}
    assert DataValidator(store).validate_dataset(spec).ok
    assert path.read_bytes() == before
    store.write(spec, observations().iloc[1:])
    assert len(store.read_dataset(spec)) == 2


@pytest.mark.parametrize(
    ("column", "value"),
    [
        ("observation_hash", "0" * 64),
        ("raw_row_json", "{}"),
        ("float_share", 1.0),
        ("identity_status", "SINGLE_OBSERVATION"),
        ("identity_variant_count", 1),
        ("observation_contract", "unknown"),
        ("source_requests_json", "not-json"),
    ],
)
def test_validator_rejects_corrupt_observations_and_annotations(tmp_path, column, value) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    store.write(spec, observations())
    path = next(store.dataset_dir(spec).glob("**/*.parquet"))
    frame = pd.read_parquet(path)
    frame.loc[0, column] = value
    frame.to_parquet(path, index=False)
    result = DataValidator(store).validate_dataset(spec)
    assert not result.ok
    assert result.status == "invalid"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(DataValidationError):
        store.write(spec, observations())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_unknown_is_not_zero_and_extra_provider_fields_are_preserved(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    frame = observations()
    frame["float_share"] = [None, 0.0]
    frame["provider_note"] = ["unknown", "explicit zero"]
    store.write(spec, frame)
    stored = store.read_dataset(spec)
    assert len(stored) == 2
    assert stored.float_share.isna().sum() == 1
    assert stored.float_share.eq(0).sum() == 1
    assert set(stored.provider_note) == {"unknown", "explicit zero"}
    assert set(stored.identity_status) == {"AMBIGUOUS"}


def test_same_observation_merges_request_references_without_changing_identity(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    frame = observations().iloc[:1]
    requests = [
        {"endpoint": "share_float", "params": {"ann_date": "20261104", "limit": size, "offset": 0}}
        for size in (6000, 2000)
    ]
    for request in requests + requests:
        store.write(spec, capture_observations(frame, request))
    stored = store.read_dataset(spec)
    assert len(stored) == 1
    assert not stored.legacy_provenance.any()
    assert len(json.loads(stored.source_requests_json.iloc[0])) == 2
    assert set(stored.identity_status) == {"SINGLE_OBSERVATION"}


def test_partial_metadata_cannot_masquerade_as_legacy(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    store.write(spec, observations())
    path = next(store.dataset_dir(spec).glob("**/*.parquet"))
    frame = pd.read_parquet(path).drop(columns="observation_hash")
    frame.to_parquet(path, index=False)
    result = DataValidator(store).validate_dataset(spec)
    assert not result.ok
    assert "incomplete observation metadata" in result.errors[0]


@pytest.mark.parametrize("mutation", ["date", "offset", "limit", "endpoint", "missing"])
def test_request_provenance_is_validated(tmp_path, mutation) -> None:
    request = {
        "endpoint": "share_float",
        "params": {"ann_date": "20261104", "limit": 6000, "offset": 0},
    }
    frame = capture_observations(observations(), request)
    if mutation == "date":
        request["params"]["ann_date"] = "20261103"
    elif mutation == "offset":
        request["params"]["offset"] = -1
    elif mutation == "limit":
        request["params"]["limit"] = 0
    elif mutation == "endpoint":
        request["endpoint"] = "daily"
    payload = [] if mutation == "missing" else [request]
    frame["source_requests_json"] = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    with pytest.raises(DataValidationError, match="provenance"):
        ParquetDataStore(tmp_path).write(get_dataset_spec("share_float"), frame)


def test_changed_raw_value_and_rehashed_row_cannot_hide_group_ambiguity(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    store.write(spec, observations())
    path = next(store.dataset_dir(spec).glob("**/*.parquet"))
    frame = pd.read_parquet(path)
    frame["identity_status"] = "SINGLE_OBSERVATION"
    frame["identity_variant_count"] = 1
    raw = json.loads(frame.raw_row_json.iloc[0])
    raw["float_share"] = 12345.0
    encoded = json.dumps(raw, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    frame.loc[0, "raw_row_json"] = encoded
    frame.loc[0, "float_share"] = 12345.0
    frame.loc[0, "observation_hash"] = hashlib.sha256(
        f"share_float_observation_v1\n{encoded}".encode()
    ).hexdigest()
    frame.to_parquet(path, index=False)
    result = DataValidator(store).validate_dataset(spec)
    assert not result.ok
    assert "ambiguity annotation mismatch" in result.errors[0]


def test_raw_schema_evolution_preserves_missing_vs_explicit_null(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    first = observations().iloc[:1]
    second = first.assign(provider_note=None)
    store.write(spec, first)
    store.write(spec, second)
    stored = store.read_dataset(spec)
    assert len(stored) == 2
    assert {"provider_note" in json.loads(raw) for raw in stored.raw_row_json} == {True, False}
    assert set(stored.identity_status) == {"AMBIGUOUS"}
    assert DataValidator(store).validate_dataset(spec).ok


def test_legacy_other_partition_unchanged_and_new_partition_is_valid(tmp_path) -> None:
    store = ParquetDataStore(tmp_path)
    spec = get_dataset_spec("share_float")
    legacy = store.dataset_dir(spec) / "year=2026/month=10/data.parquet"
    legacy.parent.mkdir(parents=True)
    observations().assign(ann_date="20261025").to_parquet(legacy, index=False)
    before = legacy.read_bytes()
    store.write(spec, observations())
    stored = store.read_dataset(spec)
    assert len(stored) == 4
    assert set(stored.identity_variant_count) == {2}
    assert DataValidator(store).validate_dataset(spec).ok
    assert legacy.read_bytes() == before


@pytest.mark.parametrize("value", [float("inf"), float("-inf")])
def test_infinite_provider_amount_is_not_accepted(tmp_path, value) -> None:
    with pytest.raises(DataValidationError, match="non-finite"):
        ParquetDataStore(tmp_path).write(
            get_dataset_spec("share_float"), observations().assign(float_share=value)
        )


def test_cli_init_continues_after_ambiguous_day_with_visible_quality_warning(
    tmp_path, monkeypatch, capsys
) -> None:
    from ashare_quant import cli
    from ashare_quant.config import load_settings
    from ashare_quant.data.ingestion import DataIngestionService

    class Client:
        def query(self, endpoint, **params):
            assert endpoint == "share_float"
            return observations().assign(ann_date=params["ann_date"])

    settings = load_settings("config/default.yaml")
    settings.paths.data_quality_logs = tmp_path / "logs"
    settings.data.run_baostock_post_ingestion_check = False
    monkeypatch.setattr(cli, "load_settings", lambda _: settings)
    monkeypatch.setattr(DataIngestionService, "_get_client", lambda _: Client())
    code = cli.main(
        [
            "--config",
            "config/default.yaml",
            "data",
            "--storage-root",
            str(tmp_path / "raw"),
            "init",
            "--dataset",
            "share_float",
            "--start-date",
            "20261104",
            "--end-date",
            "20261105",
        ]
    )
    assert code == 0
    captured = capsys.readouterr()
    assert "rows_written=4" in captured.out
    assert "ambiguous raw observations retained" in captured.err + captured.out
    store = ParquetDataStore(tmp_path / "raw")
    spec = get_dataset_spec("share_float")
    assert len(store.read_dataset(spec)) == 4
    assert store.status(spec).max_date == "20261105"
    assert all(result.ok for result in DataValidator(store).validate_all(("share_float",)))
    events = list((tmp_path / "logs").glob("**/*.jsonl"))
    assert events
    assert any("AMBIGUOUS" in path.read_text() for path in events)
