import pandas as pd

from phishshield.config import load_params
from phishshield.data.ingest import stratified_subsample, to_schema
from phishshield.data.split import stratified_split
from phishshield.data.validate import validate

CFG = load_params()["validation"]


def _df(n=1200):
    return pd.DataFrame({"url": [f"http://site{i}.com/page" for i in range(n)], "label": [i % 2 for i in range(n)]})


def test_phiusiil_label_is_flipped_to_phishing_equals_1():
    raw = pd.DataFrame({"FILENAME": ["a", "b"], "URL": ["https://good.com", "http://bad.xyz"], "label": [1, 0]})
    out = to_schema(raw)
    assert out["label"].tolist() == [0, 1]          # PhiUSIIL: 1 = legit  ->  ours: 1 = phishing
    assert list(out.columns) == ["url", "label", "source"]


def test_stratified_subsample_keeps_class_ratio():
    df = pd.DataFrame({"url": range(10000), "label": [1] * 3000 + [0] * 7000})
    sub = stratified_subsample(df, 1000, seed=0)
    assert len(sub) == 1000 and abs(sub["label"].mean() - 0.3) < 0.01


def test_validation_passes_clean_data():
    clean, rep = validate(_df(), CFG)
    assert rep["passed"] and len(clean) == 1200


def test_validation_schema_failure():
    _, rep = validate(pd.DataFrame({"url": ["http://a.com"]}), CFG)
    assert not rep["passed"] and rep["checks"]["schema_required_columns"]["missing"] == ["label"]


def test_validation_drops_nulls_bad_labels_duplicates_and_adds_scheme():
    df = pd.concat([_df(), pd.DataFrame({"url": [None, "http://x.com", "http://site1.com/page", "example.org/a"],
                                         "label": [1, 7, 1, 0]})], ignore_index=True)
    clean, rep = validate(df, CFG)
    c = rep["checks"]
    assert c["nulls"]["null_rows"] == 1
    assert c["range_label"]["out_of_range"] == 1
    assert c["duplicates"]["duplicates"] == 1
    assert c["format_url"]["scheme_added"] == 1 and "http://example.org/a" in set(clean["url"])


def test_validation_fails_on_class_imbalance():
    df = pd.DataFrame({"url": [f"http://s{i}.com" for i in range(2000)], "label": [1] * 20 + [0] * 1980})
    _, rep = validate(df, CFG)
    assert not rep["passed"] and not rep["checks"]["range_class_balance"]["passed"]


def test_validation_fails_on_url_length_range():
    df = _df()
    df.loc[:100, "url"] = "http://" + "a" * 3000 + ".com"
    _, rep = validate(df, CFG)
    assert not rep["checks"]["range_url_length"]["passed"]


def test_stratified_split_sizes_and_no_overlap():
    df = _df(2000)
    parts = stratified_split(df, 0.15, 0.15, seed=0)
    assert sum(len(p) for p in parts.values()) == 2000
    assert not set(parts["train"].url) & set(parts["test"].url)
    for p in parts.values():
        assert abs(p["label"].mean() - 0.5) < 0.02
