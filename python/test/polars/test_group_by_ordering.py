"""Stability of sample-and-aggregate under ordered and unordered adjacency."""

import pytest

import opendp.prelude as dp

pl = pytest.importorskip("polars")
pl_testing = pytest.importorskip("polars.testing")


def _input():
    data = pl.LazyFrame(
        {"key": [1, 2, 1, 2], "i": [1, 2, 3, 4], "x": [1.0, 2.0, 3.0, 4.0]}
    )
    domain = dp.lazyframe_domain(
        [
            dp.series_domain("key", dp.atom_domain(T=dp.i64)),
            dp.series_domain("i", dp.atom_domain(T=dp.i64)),
            dp.series_domain("x", dp.atom_domain(T=dp.f64)),
        ]
    )
    return domain, data


def _stable(domain, metric, plan):
    return dp.t.make_stable_lazyframe(
        domain, metric, plan, MO="FrameDistance<SymmetricDistance>"
    )


@pytest.mark.parametrize("method", ["first", "sum", "mean", "min", "max"])
def test_unordered_group_by_requires_invariant_aggregate(method):
    domain, data = _input()
    with pytest.raises(dp.OpenDPException, match="Order-sensitive aggregation"):
        _stable(
            domain,
            dp.symmetric_distance(),
            data.group_by("key").agg(getattr(pl.col("x"), method)()),
        )


@pytest.mark.parametrize("ordered", [False, True])
def test_group_by_rejects_positional_literals(ordered):
    domain, data = _input()
    weights = pl.lit(pl.Series([1, 0, 0, 0]))
    plan = data.group_by("key").agg((pl.col("i") * weights).sum())
    metric = dp.insert_delete_distance() if ordered else dp.symmetric_distance()
    with pytest.raises(dp.OpenDPException, match="Only scalar literals"):
        _stable(domain, metric, plan)


@pytest.mark.parametrize("ordered", [False, True])
def test_group_by_rejects_physical_group_indices(ordered):
    domain, data = _input()
    metric = dp.insert_delete_distance() if ordered else dp.symmetric_distance()
    with pytest.warns(DeprecationWarning, match=r"agg_groups\(\) is deprecated"):
        plan = data.group_by("key").agg(pl.col("x").sort().agg_groups())
    with pytest.raises(dp.OpenDPException, match="physical row indices"):
        _stable(domain, metric, plan)


@pytest.mark.parametrize("ordered", [False, True])
@pytest.mark.parametrize(
    "quantile",
    ["private", "series", None, -0.1, 1.1, float("nan"), float("inf"), "0.5", True],
)
def test_group_by_rejects_nonpublic_or_invalid_quantile(ordered, quantile):
    domain, data = _input()
    if quantile == "private":
        quantile = pl.col("x").first()
    elif quantile == "series":
        quantile = pl.lit(pl.Series([0.5]))
    else:
        quantile = pl.lit(quantile)
    metric = dp.insert_delete_distance() if ordered else dp.symmetric_distance()
    plan = data.group_by("key").agg(pl.col("x").sort().quantile(quantile))
    with pytest.raises(dp.OpenDPException, match="public numeric scalar literal"):
        _stable(domain, metric, plan)


@pytest.mark.parametrize("maintain_order", [False, True])
def test_unordered_group_by_accepts_random_permutation(maintain_order):
    domain, data = _input()
    values = pl.col("x").dp.random_permutation()  # type: ignore[attr-defined]
    plan = data.group_by("key", maintain_order=maintain_order).agg(
        (pl.col("i").sum() + 1).alias("integer"), values.mean().alias("mean")
    )
    transformation = _stable(domain, dp.symmetric_distance(), plan)
    expected = pl.DataFrame({"key": [1, 2], "integer": [5, 7], "mean": [2.0, 3.0]})
    for frame in [data, data.reverse()]:
        pl_testing.assert_frame_equal(
            transformation(frame).collect().sort("key"), expected
        )


@pytest.mark.parametrize(
    "metric_name", ["symmetric", "ordered", "frame_ordered", "hamming"]
)
@pytest.mark.parametrize("maintain_order", [False, True])
@pytest.mark.parametrize("postprocess", [None, "select", "filter", "with_columns"])
def test_group_by_output_metric_and_postprocessing(
    metric_name, maintain_order, postprocess
):
    domain, data = _input()
    metric = {
        "symmetric": dp.symmetric_distance(),
        "ordered": dp.insert_delete_distance(),
        "frame_ordered": dp.frame_distance(dp.insert_delete_distance()),
        "hamming": dp.hamming_distance(),
    }[metric_name]
    plan = data.group_by("key", maintain_order=maintain_order).agg(pl.col("i").sum())
    if postprocess == "select":
        plan = plan.select("key", "i")
    elif postprocess == "filter":
        plan = plan.filter(pl.col("i") > 0)
    elif postprocess == "with_columns":
        plan = plan.with_columns((pl.col("i") + 1).alias("i"))
    transformation = _stable(domain, metric, plan)
    assert str(transformation.output_metric) == "FrameDistance(SymmetricDistance())"
    distance = [dp.polars.Bound(per_group=1)] if metric_name == "frame_ordered" else 1
    assert transformation.map(distance)[0].per_group == (
        4 if metric_name == "hamming" else 2
    )
    assert transformation(data).collect().height == 2


@pytest.mark.parametrize("method", ["first", "last", "mean", "sum"])
def test_float_sort_does_not_establish_canonical_order(method):
    domain = dp.lazyframe_domain(
        [
            dp.series_domain("key", dp.atom_domain(T=dp.i64)),
            dp.series_domain("x", dp.atom_domain(T=dp.f64)),
        ]
    )
    for values in [[0.0, -0.0], [-0.0, 0.0]]:
        frame = pl.LazyFrame({"key": [1, 1], "x": values})
        plan = frame.group_by("key").agg(getattr(pl.col("x").sort(), method)())
        with pytest.raises(dp.OpenDPException, match="Order-sensitive aggregation"):
            _stable(domain, dp.symmetric_distance(), plan)


def test_integer_sort_does_not_certify_sorted_metadata():
    domain, _ = _input()
    # Polars may use this metadata to skip sort; it isn't an ordering proof.
    data = pl.DataFrame(
        {
            "key": [1, 1, 1],
            "i": pl.Series([3, 1, 2], dtype=pl.Int64).set_sorted(),
            "x": [1.0, 2.0, 3.0],
        }
    ).lazy()
    plan = data.group_by("key").agg(pl.col("i").sort().first())
    with pytest.raises(dp.OpenDPException, match="Order-sensitive aggregation"):
        _stable(domain, dp.symmetric_distance(), plan)


def test_unordered_group_by_rejects_floating_key_representatives():
    domain, _ = _input()
    for values in [[0.0, -0.0], [-0.0, 0.0]]:
        data = pl.LazyFrame({"key": [1, 1], "i": [1, 2], "x": values})
        with pytest.raises(dp.OpenDPException, match="Floating-point grouping keys"):
            _stable(domain, dp.symmetric_distance(), data.group_by("x").agg(pl.len()))


@pytest.mark.parametrize("method", ["min", "max"])
def test_float_reducers_reject_sorted_metadata_fast_paths(method):
    domain, _ = _input()
    for values in [[0.0, -0.0], [-0.0, 0.0]]:
        data = pl.DataFrame(
            {"key": [1, 1], "i": [1, 2], "x": pl.Series(values).set_sorted()}
        ).lazy()
        plan = data.group_by("key").agg(getattr(pl.col("x"), method)())
        with pytest.raises(dp.OpenDPException, match="Order-sensitive aggregation"):
            _stable(domain, dp.symmetric_distance(), plan)
        safe_plan = data.group_by("key").agg(
            getattr(pl.col("x").dp.random_permutation(), method)()
        )
        assert (
            _stable(domain, dp.symmetric_distance(), safe_plan)(data).collect().height
            == 1
        )


def test_group_by_cannot_preserve_ordered_output_metric():
    domain, data = _input()
    with pytest.raises(dp.OpenDPException, match="group-by cannot preserve"):
        dp.t.make_stable_lazyframe(
            domain,
            dp.frame_distance(dp.insert_delete_distance()),
            data.group_by("key").agg(pl.col("i").sum()),
        )


def test_hamming_rejects_unsupported_output_metric():
    domain, data = _input()
    with pytest.raises(dp.OpenDPException, match="Unsupported output metric"):
        dp.t.make_stable_lazyframe(
            domain, dp.hamming_distance(), data, MO="FrameDistance<SymmetricIdDistance>"
        )


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("measure_name", ["max", "zcdp", "approx_max", "approx_zcdp"])
def test_private_select_after_ordered_group_by(bounded, measure_name):
    domain, data = _input()
    measure = (
        dp.max_divergence()
        if measure_name.endswith("max")
        else dp.zero_concentrated_divergence()
    )
    if measure_name.startswith("approx"):
        measure = dp.approximate(measure)
    metric = dp.hamming_distance() if bounded else dp.insert_delete_distance()
    # The first aggregation may rely on ordered input. Only its output loses order.
    plan = data.group_by("key").agg(pl.col("x").first()).select(dp.len(signed=True))
    measurement = dp.m.make_private_lazyframe(
        domain, metric, measure, plan, global_scale=1.0
    )
    privacy_loss = measurement.map(1)
    if measure_name.startswith("approx"):
        privacy_loss, delta = privacy_loss
        assert delta == 0.0
    sensitivity = 4 if bounded else 2
    expected = sensitivity if measure_name.endswith("max") else sensitivity**2 / 2
    assert privacy_loss == pytest.approx(expected)
    release = measurement(data).collect()
    assert release.shape == (1, 1)
    assert release.schema["len"] == pl.Int64


@pytest.mark.parametrize("bounded", [False, True])
def test_private_group_by_after_ordered_group_by(bounded):
    domain, data = _input()
    metric = dp.hamming_distance() if bounded else dp.insert_delete_distance()
    keys = pl.LazyFrame({"key": [1, 2]})
    plan = (
        data.group_by("key")
        .agg(pl.col("x").first())
        .group_by("key")
        .agg(dp.len(signed=True))
        .join(keys, on="key", how="right")
    )
    measurement = dp.m.make_private_lazyframe(
        domain, metric, dp.max_divergence(), plan, global_scale=1.0
    )
    assert measurement.map(1) == pytest.approx(4 if bounded else 2)
    release = measurement(data).collect().sort("key")
    assert release["key"].to_list() == [1, 2]


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("zcdp", [False, True])
def test_ordered_private_select_without_group_by_unchanged(bounded, zcdp):
    domain, data = _input()
    domain = dp.with_margin(domain, dp.polars.Margin(max_length=4))
    metric = dp.hamming_distance() if bounded else dp.insert_delete_distance()
    measure = dp.zero_concentrated_divergence() if zcdp else dp.max_divergence()
    plan = data.select(pl.col("x").dp.sum((0.0, 4.0)))  # type: ignore[attr-defined]
    measurement = dp.m.make_private_lazyframe(
        domain, metric, measure, plan, global_scale=1.0
    )
    assert measurement.map(1) == pytest.approx(0.5 if zcdp else 1.0)
    assert measurement(data).collect().shape == (1, 1)


def test_float_sum_after_group_by_uses_unordered_analysis():
    domain, data = _input()
    domain = dp.with_margin(domain, dp.polars.Margin(max_length=4))
    grouped = data.group_by("key").agg(pl.col("x").first())
    total = pl.col("x").dp.sum((0.0, 4.0))
    stable = _stable(domain, dp.insert_delete_distance(), grouped)
    measurement = dp.m.make_private_lazyframe(
        domain,
        dp.insert_delete_distance(),
        dp.max_divergence(),
        grouped.select(total),
        global_scale=1.0,
    )
    # Compare with explicit composition through the unordered intermediate space.
    intermediate = stable(data).collect().clear().lazy()
    after = dp.m.make_private_lazyframe(
        stable.output_domain,
        stable.output_metric,
        dp.max_divergence(),
        intermediate.select(total),
        global_scale=1.0,
    )
    assert measurement.map(1) == after.map(stable.map(1))
    assert measurement(data).collect().shape == (1, 1)


def test_ordered_private_retry_preserves_original_error():
    domain, data = _input()
    plan = data.group_by("key").agg(pl.col("x").first()).select(pl.col("x").sum())
    # Both analyses reject this non-private selection. Keep the original error,
    # rather than replacing it with a speculative fallback's diagnostic.
    with pytest.raises(dp.OpenDPException, match="group-by cannot preserve"):
        dp.m.make_private_lazyframe(
            domain,
            dp.insert_delete_distance(),
            dp.max_divergence(),
            plan,
            global_scale=1.0,
        )
