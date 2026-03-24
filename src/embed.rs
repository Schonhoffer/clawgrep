//! Embedding model using tract for ONNX inference.
//!
//! Uses tract-onnx to load and run a BERT-class ONNX model, with the
//! `tokenizers` crate for text tokenization. Model weights are downloaded
//! from HuggingFace Hub on first use (~65 MB) into the `models/`
//! subdirectory of the cache directory.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

use anyhow::{Context, Result};
use hf_hub::api::sync::ApiBuilder;
use rayon::prelude::*;
use tokenizers::Tokenizer;
use tract_onnx::prelude::*;

use crate::cache;

/// Default model — BAAI/bge-small-en-v1.5.
pub const DEFAULT_MODEL: &str = "BAAI/bge-small-en-v1.5";

/// Dimensionality of the default model's output vectors.
pub const DEFAULT_DIM: usize = 384;

/// Fixed sequence length for model inference.  All inputs are padded or
/// truncated to this length so that tract can fully optimize the ONNX
/// graph for a single known shape.  128 is enough for 20-line chunks
/// and keeps per-inference time low (attention is O(n²) in seq length).
const MAX_SEQ_LEN: usize = 128;

type TractModel = SimplePlan<TypedFact, Box<dyn TypedOp>, Graph<TypedFact, Box<dyn TypedOp>>>;

/// Wraps a loaded ONNX model and its tokenizer.
pub struct Embedder {
    model: TractModel,
    tokenizer: Tokenizer,
    model_name: String,
}

impl Embedder {
    /// Load the default model.  First call downloads weights (~65 MB)
    /// into `models/` inside the cache directory.  Pass the same
    /// `custom_cache` you use for the SQLite embeddings cache so both
    /// live in one place.
    pub fn new(custom_cache: Option<&Path>) -> Result<Self> {
        let cache_dir = model_cache_dir(custom_cache);
        let api = ApiBuilder::new()
            .with_cache_dir(cache_dir)
            .build()
            .context("failed to initialize model downloader")?;
        let repo = api.model(DEFAULT_MODEL.to_string());

        let model_path = repo
            .get("onnx/model.onnx")
            .context("failed to download ONNX model")?;
        let tokenizer_path = repo
            .get("tokenizer.json")
            .context("failed to download tokenizer")?;

        let mut tokenizer = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| anyhow::anyhow!("loading tokenizer: {e}"))?;
        tokenizer
            .with_truncation(Some(tokenizers::TruncationParams {
                max_length: MAX_SEQ_LEN,
                ..Default::default()
            }))
            .map_err(|e| anyhow::anyhow!("setting truncation: {e}"))?;
        tokenizer.with_padding(Some(tokenizers::PaddingParams {
            strategy: tokenizers::PaddingStrategy::Fixed(MAX_SEQ_LEN),
            ..Default::default()
        }));

        let model = tract_onnx::onnx()
            .model_for_path(model_path)
            .context("loading ONNX model")?
            .with_input_fact(
                0,
                InferenceFact::dt_shape(DatumType::I64, &[1, MAX_SEQ_LEN]),
            )?
            .with_input_fact(
                1,
                InferenceFact::dt_shape(DatumType::I64, &[1, MAX_SEQ_LEN]),
            )?
            .with_input_fact(
                2,
                InferenceFact::dt_shape(DatumType::I64, &[1, MAX_SEQ_LEN]),
            )?
            .into_optimized()
            .context("optimizing model")?
            .into_runnable()
            .context("preparing model for inference")?;

        Ok(Self {
            model,
            tokenizer,
            model_name: DEFAULT_MODEL.to_string(),
        })
    }

    /// The identifier of the loaded model (e.g. `"BAAI/bge-small-en-v1.5"`).
    pub fn model_name(&self) -> &str {
        &self.model_name
    }

    /// Embed a batch of texts. Returns one `Vec<f32>` per input, each of
    /// length [`DEFAULT_DIM`].  Uses rayon to embed chunks in parallel.
    /// Both `SimplePlan::run()` and `Tokenizer::encode()` take `&self`
    /// and allocate their own working buffers, so concurrent calls from
    /// separate threads are safe with no contention.
    pub fn embed_batch(&self, texts: &[&str], verbose: bool) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }
        let total = texts.len();
        let done = AtomicUsize::new(0);
        texts
            .par_iter()
            .map(|t| {
                let result = self.embed_one(t);
                let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                if verbose {
                    eprintln!("clawgrep: embedded {}/{} segments", n, total);
                }
                result
            })
            .collect()
    }

    /// Embed a single string.
    pub fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let encoding = self
            .tokenizer
            .encode(text, true)
            .map_err(|e| anyhow::anyhow!("tokenization failed: {e}"))?;

        let ids: Vec<i64> = encoding.get_ids().iter().map(|&v| v as i64).collect();
        let mask: Vec<i64> = encoding
            .get_attention_mask()
            .iter()
            .map(|&v| v as i64)
            .collect();
        let mask_f: Vec<f32> = mask.iter().map(|&v| v as f32).collect();
        let type_ids: Vec<i64> = encoding.get_type_ids().iter().map(|&v| v as i64).collect();
        let seq_len = ids.len();

        let ids_t = tract_ndarray::Array2::from_shape_vec((1, seq_len), ids)?;
        let mask_t = tract_ndarray::Array2::from_shape_vec((1, seq_len), mask)?;
        let type_t = tract_ndarray::Array2::from_shape_vec((1, seq_len), type_ids)?;

        let outputs = self
            .model
            .run(tvec![
                ids_t.into_tvalue(),
                mask_t.into_tvalue(),
                type_t.into_tvalue(),
            ])
            .context("model inference failed")?;

        // Last hidden state: [1, seq_len, hidden_dim]
        let hidden = outputs[0].to_array_view::<f32>()?;

        // Mean pooling using attention mask
        let mask_sum: f32 = mask_f.iter().sum();
        let mut pooled = vec![0.0f32; DEFAULT_DIM];
        if mask_sum > 0.0 {
            for (i, &m) in mask_f.iter().enumerate() {
                if m > 0.0 {
                    for j in 0..DEFAULT_DIM {
                        pooled[j] += hidden[[0, i, j]];
                    }
                }
            }
            for v in &mut pooled {
                *v /= mask_sum;
            }
        }

        // L2 normalize
        let norm: f32 = pooled.iter().map(|v| v * v).sum::<f32>().sqrt();
        if norm > 0.0 {
            for v in &mut pooled {
                *v /= norm;
            }
        }

        Ok(pooled)
    }
}

// ---------------------------------------------------------------------------
// Model cache
// ---------------------------------------------------------------------------

/// Return a persistent directory for model weights.  Lives inside the
/// same cache directory used for the SQLite embeddings DB, under a
/// `models/` subdirectory.
fn model_cache_dir(custom_cache: Option<&Path>) -> PathBuf {
    let dir = cache::cache_dir(custom_cache).join("models");
    let _ = std::fs::create_dir_all(&dir);
    dir
}

// ---------------------------------------------------------------------------
// Vector math helpers – operate on slices so they work with both owned and
// borrowed data.
// ---------------------------------------------------------------------------

/// Cosine similarity between two vectors.
/// Returns 0.0 when either vector has zero magnitude.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }
    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5);
    }

    #[test]
    fn cosine_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5);
    }
}
