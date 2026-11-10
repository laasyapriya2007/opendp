use super::*;

fn plugin(limit: Expr, flags: FunctionOptions) -> Expr {
    Expr::Function {
        input: vec![col("x"), limit],
        function: FunctionExpr::FfiPlugin {
            lib: "opendp".into(),
            symbol: "random_permutation_plugin".into(),
            kwargs: Default::default(),
            flags,
        },
    }
}

#[test]
fn test_match_random_permutation_parameters() {
    for limit in [lit(NULL), lit(0u64), lit(1i32), lit(u64::MAX)] {
        let flags = if matches!(&limit, Expr::Literal(lit) if lit.is_null()) {
            FunctionOptions::length_preserving()
        } else {
            FunctionOptions::groupwise()
        };
        assert_eq!(
            match_random_permutation(&plugin(limit, flags)),
            Some(&col("x"))
        );
    }
    for limit in [
        col("k"),
        lit(-1),
        lit(true),
        lit(0.5),
        lit("1"),
        lit(Series::new("k".into(), [1u64])),
        lit(Series::new_null("k".into(), 1)),
    ] {
        assert!(
            match_random_permutation(&plugin(limit, FunctionOptions::groupwise())).is_none()
        );
    }
    // Elementwise flags would shuffle chunks independently, not the group.
    assert!(
        match_random_permutation(&plugin(lit(NULL), FunctionOptions::elementwise())).is_none()
    );
    assert!(
        match_random_permutation(&plugin(lit(1), FunctionOptions::length_preserving()))
            .is_none()
    );
    let mut expr = plugin(lit(NULL), FunctionOptions::length_preserving());
    if let Expr::Function {
        function: FunctionExpr::FfiPlugin { symbol, .. },
        ..
    } = &mut expr
    {
        *symbol = "other_plugin".into();
    }
    assert!(match_random_permutation(&expr).is_none());
}

#[test]
fn test_random_permutation_sizes_and_dtype() -> PolarsResult<()> {
    let data = Series::new("x".into(), [Some(1i64), None, Some(3)]);
    for limit in [None, Some(0u64), Some(1), Some(3), Some(u64::MAX)] {
        let limit_series = Series::new("k".into(), [limit]);
        let output = random_permutation(&[data.clone(), limit_series])?;
        assert_eq!(
            output.len(),
            limit.map_or(3, |k| usize::try_from(k).unwrap_or(usize::MAX).min(3))
        );
        assert_eq!(output.dtype(), data.dtype());
        assert_eq!(output.name(), data.name());
        if limit.is_none() {
            assert_eq!(output.null_count(), 1);
        }
    }
    let empty = Series::new_empty("x".into(), &DataType::Int64);
    assert!(random_permutation(&[empty, Series::new("k".into(), [None::<u64>])])?.is_empty());
    for limit in [
        Series::new("k".into(), [-1i64]),
        Series::new("k".into(), [true]),
        Series::new("k".into(), [0.5]),
        Series::new("k".into(), [1u64, 2]),
        Series::new_empty("k".into(), &DataType::UInt64),
    ] {
        assert!(random_permutation(&[data.clone(), limit]).is_err());
    }
    Ok(())
}
