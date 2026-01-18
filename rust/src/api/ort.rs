use anyhow::{Ok, Result};
use flutter_rust_bridge::frb;
use ort;

#[frb(sync)]
pub fn init_ort(name: String, path: Option<String>) -> Result<bool> {
    let res = match path {
        Some(p) => ort::init_from(p)?.with_name(name).commit(),
        None => ort::init().with_name(name).commit(),
    };
    return Ok(res);
}
