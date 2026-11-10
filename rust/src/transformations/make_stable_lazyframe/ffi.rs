use std::ffi::c_char;

use polars::{lazy::frame::LazyFrame, prelude::DslPlan};

use crate::{
    core::{FfiResult, IntoAnyTransformationFfiResultExt, Metric, MetricSpace},
    domains::{DslPlanDomain, LazyFrameDomain},
    error::Fallible,
    ffi::{
        any::{AnyDomain, AnyMetric, AnyObject, AnyTransformation, Downcast},
        util::{Type, TypeContents, to_str},
    },
    metrics::{
        ChangeOneDistance, ChangeOneIdDistance, FrameDistance, HammingDistance,
        InsertDeleteDistance, SymmetricDistance, SymmetricIdDistance,
    },
    transformations::{
        StableDslPlan,
        traits::{BoundedMetric, UnboundedMetric},
    },
};

use super::make_stable_lazyframe;

#[unsafe(no_mangle)]
pub extern "C" fn opendp_transformations__make_stable_lazyframe(
    input_domain: *const AnyDomain,
    input_metric: *const AnyMetric,
    lazyframe: *const AnyObject,
    MO: *const c_char,
) -> FfiResult<*mut AnyTransformation> {
    let input_domain = try_!(try_as_ref!(input_domain).downcast_ref::<LazyFrameDomain>()).clone();
    let input_metric = try_as_ref!(input_metric);
    let MI = input_metric.type_.clone();
    let MO = match try_!(to_str(MO)) {
        "MI" => MI.clone(),
        v => try_!(Type::try_from(v)),
    };

    let lazyframe = try_!(try_as_ref!(lazyframe).downcast_ref::<LazyFrame>()).clone();

    fn is_in<T>() -> Option<()> {
        Some(())
    }

    if let TypeContents::GENERIC { .. } = MI.contents {
        // FrameDistance<MI> -> FrameDistance<MI>
        if MI == MO {
            fn monomorphize<MI: 'static + Metric>(
                input_domain: LazyFrameDomain,
                input_metric: &AnyMetric,
                lazyframe: LazyFrame,
            ) -> Fallible<AnyTransformation>
            where
                DslPlan: StableDslPlan<MI, MI>,
                (LazyFrameDomain, MI): MetricSpace,
                (DslPlanDomain, MI): MetricSpace,
            {
                let input_metric = input_metric.downcast_ref::<MI>()?.clone();
                make_stable_lazyframe(input_domain, input_metric, lazyframe).into_any()
            }
            dispatch!(
                monomorphize,
                [(MI, [FrameDistance<SymmetricDistance>, FrameDistance<SymmetricIdDistance>, FrameDistance<InsertDeleteDistance>])],
                (
                    input_domain,
                    input_metric,
                    lazyframe
                )
            )
        // FrameDistance<InsertDeleteDistance> -> FrameDistance<SymmetricDistance>
        } else if MI == Type::of::<FrameDistance<InsertDeleteDistance>>() && MO == Type::of::<FrameDistance<SymmetricDistance>>() {
            let input_metric = try_!(input_metric.downcast_ref::<FrameDistance<InsertDeleteDistance>>()).clone();
            make_stable_lazyframe::<_, FrameDistance<SymmetricDistance>>(input_domain, input_metric, lazyframe).into_any()
        // FrameDistance<SymmetricIdDistance> -> FrameDistance<SymmetricDistance>
        } else if MI == Type::of::<FrameDistance<SymmetricIdDistance>>() && MO == Type::of::<FrameDistance<SymmetricDistance>>() {
            let input_metric = try_!(input_metric.downcast_ref::<FrameDistance<SymmetricIdDistance>>()).clone();
            make_stable_lazyframe::<_, FrameDistance<SymmetricDistance>>(input_domain, input_metric, lazyframe).into_any()
        } else {
            err!(FFI, "MI ({:?}) must match MO ({:?}), or FrameDistance<InsertDeleteDistance/SymmetricIdDistance> -> FrameDistance<SymmetricDistance>", MI.descriptor, MO.descriptor).into()
        }
    } else if dispatch!(is_in, [(MI, [SymmetricDistance, SymmetricIdDistance, InsertDeleteDistance])], ()).is_some() {
        // MI -> FrameDistance<MI>
        if MI == try_!(MO.get_atom()) {
            fn monomorphize<MI: UnboundedMetric>(
                input_domain: LazyFrameDomain,
                input_metric: &AnyMetric,
                lazyframe: LazyFrame,
            ) -> Fallible<AnyTransformation>
            where
                DslPlan: StableDslPlan<MI, FrameDistance<MI>> + StableDslPlan<FrameDistance<MI>, FrameDistance<MI>>,
            {
                let input_metric = input_metric.downcast_ref::<MI>()?.clone();
                make_stable_lazyframe(input_domain, input_metric, lazyframe).into_any()
            }
            dispatch!(
                monomorphize,
                [(MI, [SymmetricDistance, SymmetricIdDistance, InsertDeleteDistance])],
                (
                    input_domain,
                    input_metric,
                    lazyframe
                )
            )
        // InsertDeleteDistance -> FrameDistance<SymmetricDistance>
        } else if MI == Type::of::<InsertDeleteDistance>() && MO == Type::of::<FrameDistance<SymmetricDistance>>() {
            let input_metric = try_!(input_metric.downcast_ref::<InsertDeleteDistance>()).clone();
            make_stable_lazyframe::<_, FrameDistance<SymmetricDistance>>(input_domain, input_metric, lazyframe).into_any()
        // SymmetricIdDistance -> FrameDistance<SymmetricDistance>
        } else if MI == Type::of::<SymmetricIdDistance>() && MO == Type::of::<FrameDistance<SymmetricDistance>>() {
            let input_metric = try_!(input_metric.downcast_ref::<SymmetricIdDistance>()).clone();
            make_stable_lazyframe::<_, FrameDistance<SymmetricDistance>>(input_domain, input_metric, lazyframe).into_any()
        } else {
            err!(FFI, "MI ({:?}) must match MO ({:?}), or InsertDeleteDistance/SymmetricIdDistance -> FrameDistance<SymmetricDistance>", MI.descriptor, MO.descriptor).into()
        }
    } else if MI == Type::of::<HammingDistance>() && MO == Type::of::<FrameDistance<SymmetricDistance>>() {
        let input_metric = try_!(input_metric.downcast_ref::<HammingDistance>()).clone();
        make_stable_lazyframe::<_, FrameDistance<SymmetricDistance>>(input_domain, input_metric, lazyframe).into_any()
    } else {
        fn monomorphize<MI: BoundedMetric>(
            input_domain: LazyFrameDomain,
            input_metric: &AnyMetric,
            lazyframe: LazyFrame,
            output_metric: Type,
        ) -> Fallible<AnyTransformation>
        where
            DslPlan: StableDslPlan<MI, FrameDistance<MI::UnboundedMetric>>,
        {
            // Preserve the historical default, but do not silently ignore an
            // explicitly requested, unsupported output metric.
            if output_metric != Type::of::<MI>() && output_metric != Type::of::<FrameDistance<MI::UnboundedMetric>>() {
                return fallible!(FFI, "Unsupported output metric: {:?}", output_metric.descriptor);
            }
            let input_metric = input_metric.downcast_ref::<MI>()?.clone();
            make_stable_lazyframe(input_domain, input_metric, lazyframe).into_any()
        }
        dispatch!(
            monomorphize,
            [(MI, [ChangeOneDistance, HammingDistance, ChangeOneIdDistance])],
            (
                input_domain,
                input_metric,
                lazyframe,
                MO
            )
        )
    }.into()
}
