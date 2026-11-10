//! Uniform random ordered samples of Polars expression rows.
//!
//! Struct fields share a permutation. Sampling materializes the input and an
//! O(n) index buffer, even when only a partial permutation is requested.

use polars::prelude::*;
use polars_plan::{
    plans::{DynLiteralValue, LiteralValue},
    prelude::FunctionOptions,
};
use pyo3_polars::derive::polars_expr;
use rand::Rng;

use crate::traits::samplers::GeneratorOpenDP;

/// Return the input field: sampling changes neither its name nor dtype.
fn random_permutation_type_udf(fields: &[Field]) -> PolarsResult<Field> {
    let [data, limit] = fields else {
        polars_bail!(InvalidOperation: "random_permutation expects data and a limit");
    };
    polars_ensure!(limit.dtype().is_integer() || limit.dtype() == &DataType::Null,
        InvalidOperation: "random_permutation limit must be an integer or null");
    Ok(data.clone())
}

fn sample_size(limit: AnyValue<'_>, n: usize) -> PolarsResult<usize> {
    let value = match limit {
        AnyValue::Null => return Ok(n),
        AnyValue::UInt8(v) => usize::from(v),
        AnyValue::UInt16(v) => usize::from(v),
        AnyValue::UInt32(v) => usize::try_from(v).unwrap_or(usize::MAX),
        AnyValue::UInt64(v) => usize::try_from(v).unwrap_or(usize::MAX),
        AnyValue::Int8(v) if v >= 0 => v as usize,
        AnyValue::Int16(v) if v >= 0 => v as usize,
        AnyValue::Int32(v) if v >= 0 => usize::try_from(v).unwrap_or(usize::MAX),
        AnyValue::Int64(v) if v >= 0 => usize::try_from(v).unwrap_or(usize::MAX),
        AnyValue::Int128(v) if v >= 0 => usize::try_from(v).unwrap_or(usize::MAX),
        _ => {
            polars_bail!(InvalidOperation: "random_permutation limit must be a nonnegative integer or null")
        }
    };
    Ok(value.min(n))
}

/// Partial Fisher-Yates generates a uniform ordered sample without replacement.
#[polars_expr(output_type_func=random_permutation_type_udf)]
fn random_permutation_plugin(inputs: &[Series]) -> PolarsResult<Series> {
    random_permutation(inputs)
}

fn random_permutation(inputs: &[Series]) -> PolarsResult<Series> {
    let [data, limit] = inputs else {
        polars_bail!(ComputeError: "random_permutation expects data and a limit");
    };
    polars_ensure!(limit.len() == 1, InvalidOperation: "random_permutation limit must be scalar");
    let n = data.len();
    let k = sample_size(limit.get(0)?, n)?;
    if k == 0 {
        return Ok(data.slice(0, 0));
    }
    let end = IdxSize::try_from(n).map_err(
        |_| polars_err!(ComputeError: "random_permutation input exceeds Polars index range"),
    )?;
    let mut indices: Vec<IdxSize> = (0..end).collect();
    let mut rng = GeneratorOpenDP::new();
    // The final position has no choice and requires no randomness.
    for i in 0..k.min(n.saturating_sub(1)) {
        let j = rng.gen_range(i, n);
        indices.swap(i, j);
    }
    rng.error
        .map_err(|e| polars_err!(ComputeError: "random_permutation entropy error: {}", e))?;

    indices.truncate(k);
    data.take(&IdxCa::from_vec("index".into(), indices))
}

/// Recognize only the OpenDP sampler with public, valid parameters and the
/// groupwise execution flags used by the Python wrapper. In particular,
/// elementwise execution would not uniformly permute the entire group.
pub(crate) fn match_random_permutation(expr: &Expr) -> Option<&Expr> {
    let Expr::Function {
        input,
        function:
            FunctionExpr::FfiPlugin {
                lib,
                symbol,
                kwargs,
                flags,
            },
    } = expr
    else {
        return None;
    };
    if !lib.contains("opendp")
        || symbol.as_str() != "random_permutation_plugin"
        || !kwargs.is_empty()
    {
        return None;
    }
    let [data, Expr::Literal(limit)] = input.as_slice() else {
        return None;
    };
    // Series/ranges are positional, not public scalar parameters. Avoid
    // materializing dynamic list literals, which Polars does not support here.
    if !matches!(
        limit,
        LiteralValue::Scalar(_) | LiteralValue::Dyn(DynLiteralValue::Int(_))
    ) {
        return None;
    }
    sample_size(limit.to_any_value()?, 0).ok()?;
    let expected = if limit.is_null() {
        FunctionOptions::length_preserving()
    } else {
        FunctionOptions::groupwise()
    };
    if flags != &expected {
        return None;
    }
    Some(data)
}

#[cfg(test)]
mod test;
