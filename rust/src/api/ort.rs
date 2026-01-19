use anyhow::{Ok, Result};
use flutter_rust_bridge::frb;
use ort::{
    environment::GlobalThreadPoolOptions,
    session::{
        builder::{GraphOptimizationLevel, SessionBuilder},
        Session,
    },
};

#[derive(Debug, Clone, Default)]
pub struct OrtEnvironmentOptions {
    pub name: Option<String>,
    pub dylib_path: Option<String>,
    pub inter_threads: Option<i64>,
    pub intra_threads: Option<i64>,
    pub spin_control: Option<bool>,
    pub intra_affinity: Option<String>,
    pub telemetry: Option<bool>,
}

#[derive(Debug, Clone, Default)]
pub struct OrtSessionOptions {
    pub intra_threads: Option<i64>,
    pub inter_threads: Option<i64>,
    pub parallel_execution: Option<bool>,
    /// 0..=3 mapping to ORT graph optimization levels.
    pub optimization_level: Option<i64>,
}

#[derive(Debug, Clone, Default)]
pub struct OrtInitOptions {
    pub environment: Option<OrtEnvironmentOptions>,
    pub session: Option<OrtSessionOptions>,
}

#[frb(sync)]
pub fn init_ort(name: String, path: Option<String>) -> Result<bool> {
    let res = match path {
        Some(p) => ort::init_from(p)?.with_name(name).commit(),
        None => ort::init().with_name(name).commit(),
    };
    return Ok(res);
}

#[frb(sync)]
pub fn init_ort_with_options(options: OrtEnvironmentOptions) -> Result<bool> {
    init_ort_from_options(&options)
}

pub fn build_session_from_file_with_init(
    model_path: String,
    ort_options: Option<OrtInitOptions>,
) -> Result<Session> {
    if let Some(options) = ort_options {
        if let Some(env) = options.environment {
            init_ort_from_options(&env)?;
        }
        return build_session_from_file(model_path, options.session);
    }
    build_session_from_file(model_path, None)
}

pub fn build_session_from_file(
    model_path: String,
    session_options: Option<OrtSessionOptions>,
) -> Result<Session> {
    let builder = Session::builder()?;
    let builder = apply_session_options(builder, session_options)?;
    Ok(builder.commit_from_file(model_path)?)
}
 
fn init_ort_from_options(options: &OrtEnvironmentOptions) -> Result<bool> {
    let mut builder = match &options.dylib_path {
        Some(path) => ort::init_from(path)?,
        None => ort::init(),
    };
    let name = options
        .name
        .clone()
        .unwrap_or_else(|| "flutter_embedder".to_string());
    builder = builder.with_name(name);

    if let Some(telemetry) = options.telemetry {
        builder = builder.with_telemetry(telemetry);
    }

    if has_thread_options(options) {
        let mut thread_options = GlobalThreadPoolOptions::default();
        if let Some(count) = to_positive_usize(options.inter_threads) {
            thread_options = thread_options.with_inter_threads(count)?;
        }
        if let Some(count) = to_positive_usize(options.intra_threads) {
            thread_options = thread_options.with_intra_threads(count)?;
        }
        if let Some(spin) = options.spin_control {
            thread_options = thread_options.with_spin_control(spin)?;
        }
        if let Some(affinity) = options.intra_affinity.as_ref() {
            if !affinity.is_empty() {
                thread_options = thread_options.with_intra_affinity(affinity)?;
            }
        }
        builder = builder.with_global_thread_pool(thread_options);
    }

    Ok(builder.commit())
}

fn apply_session_options(
    mut builder: SessionBuilder,
    options: Option<OrtSessionOptions>,
) -> Result<SessionBuilder> {
    let mut optimization_level = GraphOptimizationLevel::Level3;
    let mut intra_threads = Some(1usize);
    let mut inter_threads = None;
    let mut parallel_execution = None;

    if let Some(opts) = options {
        if let Some(level) = opts.optimization_level {
            optimization_level = match level {
                0 => GraphOptimizationLevel::Disable,
                1 => GraphOptimizationLevel::Level1,
                2 => GraphOptimizationLevel::Level2,
                _ => GraphOptimizationLevel::Level3,
            };
        }
        if let Some(count) = to_positive_usize(opts.intra_threads) {
            intra_threads = Some(count);
        }
        inter_threads = to_positive_usize(opts.inter_threads);
        parallel_execution = opts.parallel_execution;
    }

    builder = builder.with_optimization_level(optimization_level)?;
    if let Some(count) = intra_threads {
        builder = builder.with_intra_threads(count)?;
    }
    if let Some(count) = inter_threads {
        builder = builder.with_inter_threads(count)?;
        if parallel_execution.is_none() {
            parallel_execution = Some(true);
        }
    }
    if let Some(enable) = parallel_execution {
        builder = builder.with_parallel_execution(enable)?;
    }

    Ok(builder)
}

fn has_thread_options(options: &OrtEnvironmentOptions) -> bool {
    options.inter_threads.is_some()
        || options.intra_threads.is_some()
        || options.spin_control.is_some()
        || options.intra_affinity.is_some()
}

fn to_positive_usize(value: Option<i64>) -> Option<usize> {
    match value {
        Some(v) if v > 0 => usize::try_from(v).ok(),
        _ => None,
    }
}
