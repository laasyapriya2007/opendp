"""Tests for the OpenDP uniform random permutation expression."""

import pytest

import opendp.extras.polars  # noqa: F401 - registers the dp expression namespace

pl = pytest.importorskip("polars")


def test_random_permutation_all():
    data = pl.DataFrame({"x": list(range(40))})
    result = data.select(pl.col("x").dp.random_permutation())["x"].to_list()
    assert sorted(result) == list(range(40))


@pytest.mark.parametrize("k", [0, 1, 5, 40, 100, 2**32, 2**63, 2**128])
def test_random_permutation_limit(k):
    data = pl.DataFrame({"x": list(range(40))})
    result = data.select(pl.col("x").dp.random_permutation(k))["x"].to_list()
    assert len(result) == min(k, 40)
    assert len(set(result)) == len(result)
    assert all(0 <= value < 40 for value in result)


def test_random_permutation_struct_alignment():
    data = pl.DataFrame({"x": list(range(30)), "y": [n * 17 for n in range(30)]})
    sample = data.select(pl.struct("x", "y").dp.random_permutation(12)).to_series()
    rows = sample.to_list()
    assert len(rows) == 12
    assert all(row["y"] == 17 * row["x"] for row in rows)
    assert len({row["x"] for row in rows}) == 12


@pytest.mark.parametrize("k", [-1, True, False, 1.2, "1"])
def test_random_permutation_rejects_invalid_limit(k):
    with pytest.raises(ValueError):
        pl.col("x").dp.random_permutation(k)


@pytest.mark.parametrize("values", [[], [None], [None, 1, 1, 2]])
@pytest.mark.parametrize("k", [None, 0, 1, 10])
def test_random_permutation_nullable_and_empty(values, k):
    data = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    result = data.select(pl.col("x").dp.random_permutation(k))["x"]
    assert result.dtype == pl.Int64
    assert len(result) == (len(values) if k is None else min(k, len(values)))
    if k is None or k >= len(values):
        assert result.sort().to_list() == data["x"].sort().to_list()


@pytest.mark.parametrize("k", [None, 0, 1, 10])
def test_random_permutation_grouped_struct(k):
    data = pl.DataFrame({"g": [0, 0, 0, 1, 1], "x": [1, 2, 3, 4, 5]})
    data = data.with_columns(y=pl.col("x") * 17)
    result = data.group_by("g").agg(
        pl.struct("x", "y").dp.random_permutation(k).alias("sample")
    )
    for group, rows in result.iter_rows():
        expected = data.filter(pl.col("g") == group)["x"].to_list()
        assert len(rows) == (len(expected) if k is None else min(k, len(expected)))
        assert len({row["x"] for row in rows}) == len(rows)
        assert all(row["x"] in expected and row["y"] == 17 * row["x"] for row in rows)
