#![deny(unsafe_code)]

//! JEPA (Joint-Embedding Predictive Architecture) for tile representation learning in PLATO rooms.
//!
//! Provides tile embedding, multi-step prediction in latent space, VICReg loss,
//! and similarity-based tile retrieval and clustering.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

/// Output of encode_and_predict: context embedding, prediction, target, and loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JEPAOutput {
    pub context_embedding: Vec<f64>,
    pub prediction: Vec<f64>,
    pub target_embedding: Vec<f64>,
    pub loss: f64,
}

/// Learned projection matrix for embedding raw tile data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileEmbedding {
    /// Projection matrix: `dims` rows × input_dim columns.
    /// Embeds an input of length `input_dim` into a vector of length `dims`.
    pub projection: Vec<Vec<f64>>,
    pub dims: usize,
    pub input_dim: usize,
}

/// Multi-step predictor operating in embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictor {
    /// Transition matrix for one prediction step: `dim × dim`.
    pub weights: Vec<Vec<f64>>,
    pub dim: usize,
}

/// VICReg loss computer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VICRegLoss {
    pub variance_threshold: f64,
    pub mu: f64,        // invariance weight
    pub lambda: f64,    // variance weight
    pub nu: f64,        // covariance weight
}

/// JEPA encoder that orchestrates embedding, prediction, and loss.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JEPAEncoder {
    pub embedder: TileEmbedding,
    pub predictor: Predictor,
    pub vicreg: VICRegLoss,
}

/// Tile similarity utilities.
pub struct TileSimilarity;

// ---------------------------------------------------------------------------
// TileEmbedding
// ---------------------------------------------------------------------------

impl TileEmbedding {
    /// Create a new embedder with a deterministic pseudo-random projection matrix.
    /// Uses a simple linear congruential generator seeded from dims + input_dim
    /// so results are reproducible across runs.
    pub fn new(input_dim: usize, dims: usize) -> Self {
        let mut seed = (dims.wrapping_mul(2654435761) ^ input_dim.wrapping_mul(2246822519)) as u64;
        let mut next = || -> f64 {
            // LCG: x_{n+1} = x_n * 6364136223846793005 + 1442695040888963407
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // Map to [-1, 1) range
            ((seed >> 33) as i64 as f64) / (1i64 << 31) as f64
        };

        let mut projection = Vec::with_capacity(dims);
        for _ in 0..dims {
            let mut row = Vec::with_capacity(input_dim);
            for _ in 0..input_dim {
                row.push(next());
            }
            projection.push(row);
        }

        Self {
            projection,
            dims,
            input_dim,
        }
    }

    /// Embed raw tile data into a `dims`-dimensional vector via learned projection.
    pub fn embed(&self, tile_data: &[f64], dims: usize) -> Vec<f64> {
        assert_eq!(
            dims, self.dims,
            "requested dims ({}) must match embedder dims ({})",
            dims, self.dims
        );
        let mut result = vec![0.0; self.dims];
        for (i, row) in self.projection.iter().enumerate() {
            let mut sum = 0.0;
            for (j, w) in row.iter().enumerate() {
                let v = tile_data.get(j).copied().unwrap_or(0.0);
                sum += w * v;
            }
            result[i] = sum;
        }
        result
    }
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

impl Predictor {
    /// Create a new predictor with identity-like transition matrix (slight perturbation).
    pub fn new(dim: usize) -> Self {
        let mut seed = dim as u64;
        let mut next = || -> u64 {
            seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            seed
        };

        let mut weights = Vec::with_capacity(dim);
        for i in 0..dim {
            let mut row = vec![0.0; dim];
            row[i] = 1.0;
            // Small perturbation to off-diagonal
            for j in 0..dim {
                if j != i {
                    row[j] = ((next() >> 33) as i64 as f64) / (1i64 << 35) as f64;
                }
            }
            weights.push(row);
        }
        Self { weights, dim }
    }

    /// Predict `steps_ahead` steps in embedding space.
    /// Each step applies the transition matrix: `embedding = weights * embedding`.
    pub fn predict(&self, embedding_a: &[f64], steps_ahead: usize) -> Vec<f64> {
        assert_eq!(embedding_a.len(), self.dim, "embedding dimension mismatch");
        let mut current = embedding_a.to_vec();
        for _ in 0..steps_ahead {
            let mut next = vec![0.0; self.dim];
            for (i, row) in self.weights.iter().enumerate() {
                for (j, w) in row.iter().enumerate() {
                    next[i] += w * current[j];
                }
            }
            current = next;
        }
        current
    }
}

// ---------------------------------------------------------------------------
// VICRegLoss
// ---------------------------------------------------------------------------

impl Default for VICRegLoss {
    fn default() -> Self {
        Self {
            variance_threshold: 1.0,
            mu: 10.0,     // invariance (MSE) weight
            lambda: 10.0, // variance weight
            nu: 1.0,      // covariance weight
        }
    }
}

impl VICRegLoss {
    pub fn new(variance_threshold: f64, mu: f64, lambda: f64, nu: f64) -> Self {
        Self {
            variance_threshold,
            mu,
            lambda,
            nu,
        }
    }

    /// Compute VICReg loss between predicted and target embeddings.
    ///
    /// - **Variance**: penalizes dimensions with std < threshold (prevents collapse)
    /// - **Invariance**: MSE between predicted and target (alignment)
    /// - **Covariance**: penalizes off-diagonal entries of the covariance matrix (decorrelation)
    pub fn compute_loss(&self, predicted: &[f64], target: &[f64]) -> f64 {
        assert_eq!(predicted.len(), target.len(), "dimension mismatch");
        let n = predicted.len();
        if n == 0 {
            return 0.0;
        }

        // Invariance: MSE
        let inv: f64 = predicted
            .iter()
            .zip(target.iter())
            .map(|(p, t)| (p - t).powi(2))
            .sum::<f64>()
            / n as f64;

        // Variance: penalty for std < threshold per dimension
        // For a single pair, variance term uses both vectors
        let combined: Vec<f64> = predicted.iter().chain(target.iter()).copied().collect();
        let n_combined = combined.len() as f64;
        let mean: f64 = combined.iter().sum::<f64>() / n_combined;
        let var: f64 = combined.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n_combined;
        let std = var.sqrt();
        let var_loss = if std < self.variance_threshold {
            (self.variance_threshold - std).powi(2)
        } else {
            0.0
        };

        // Covariance: off-diagonal penalty
        // Compute covariance between predicted and target dimensions
        let pred_mean: f64 = predicted.iter().sum::<f64>() / n as f64;
        let targ_mean: f64 = target.iter().sum::<f64>() / n as f64;
        let mut cov_loss = 0.0;
        for i in 0..n {
            for j in 0..n {
                if i != j {
                    let pi = predicted[i] - pred_mean;
                    let pj = predicted[j] - pred_mean;
                    let ti = target[i] - targ_mean;
                    let tj = target[j] - targ_mean;
                    cov_loss += (pi * pj + ti * tj).powi(2);
                }
            }
        }
        cov_loss /= (n * n - n).max(1) as f64;

        self.mu * inv + self.lambda * var_loss + self.nu * cov_loss
    }
}

// ---------------------------------------------------------------------------
// JEPAEncoder
// ---------------------------------------------------------------------------

impl JEPAEncoder {
    pub fn new(input_dim: usize, embedding_dim: usize) -> Self {
        Self {
            embedder: TileEmbedding::new(input_dim, embedding_dim),
            predictor: Predictor::new(embedding_dim),
            vicreg: VICRegLoss::default(),
        }
    }

    /// Encode a sequence of tiles and produce a prediction.
    ///
    /// - `tiles`: raw tile data vectors
    /// - `context_length`: how many tiles to use as context
    /// - `prediction_steps`: how far ahead to predict
    pub fn encode_and_predict(
        &self,
        tiles: &[Vec<f64>],
        context_length: usize,
        prediction_steps: usize,
    ) -> JEPAOutput {
        assert!(!tiles.is_empty(), "need at least one tile");

        // Embed all tiles
        let embeddings: Vec<Vec<f64>> = tiles
            .iter()
            .map(|t| self.embedder.embed(t, self.embedder.dims))
            .collect();

        // Context: average of first `context_length` embeddings
        let cl = context_length.min(embeddings.len());
        let context_embedding = if cl > 0 {
            let mut avg = vec![0.0; self.embedder.dims];
            for emb in &embeddings[..cl] {
                for (i, v) in emb.iter().enumerate() {
                    avg[i] += v;
                }
            }
            for v in avg.iter_mut() {
                *v /= cl as f64;
            }
            avg
        } else {
            vec![0.0; self.embedder.dims]
        };

        // Predict from context
        let prediction = self.predictor.predict(&context_embedding, prediction_steps);

        // Target: embedding of the tile just after context
        let target_idx = cl; // index right after context
        let target_embedding = if target_idx < embeddings.len() {
            embeddings[target_idx].clone()
        } else {
            // If no target available, use last embedding
            embeddings.last().cloned().unwrap_or_default()
        };

        // Compute loss
        let loss = self.vicreg.compute_loss(&prediction, &target_embedding);

        JEPAOutput {
            context_embedding,
            prediction,
            target_embedding,
            loss,
        }
    }
}

// ---------------------------------------------------------------------------
// TileSimilarity
// ---------------------------------------------------------------------------

impl TileSimilarity {
    /// Compute cosine similarity between two vectors.
    pub fn cosine_similarity(a: &[f64], b: &[f64]) -> f64 {
        assert_eq!(a.len(), b.len(), "dimension mismatch");
        if a.is_empty() {
            return 0.0;
        }
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        if na < 1e-12 || nb < 1e-12 {
            return 0.0;
        }
        (dot / (na * nb)).clamp(-1.0, 1.0)
    }

    /// Find the k nearest neighbors of `query` in `tile_set` by cosine similarity.
    /// Returns indices into `tile_set` sorted by descending similarity.
    pub fn knn_tiles(query: &[f64], tile_set: &[Vec<f64>], k: usize) -> Vec<usize> {
        let mut scored: Vec<(usize, f64)> = tile_set
            .iter()
            .enumerate()
            .map(|(i, t)| (i, Self::cosine_similarity(query, t)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().map(|(i, _)| i).collect()
    }

    /// Cluster tile embeddings by cosine similarity threshold.
    /// Returns groups of indices where each pair has similarity >= threshold.
    pub fn cluster_by_similarity(
        embeddings: &[Vec<f64>],
        threshold: f64,
    ) -> Vec<Vec<usize>> {
        let n = embeddings.len();
        let mut visited = vec![false; n];
        let mut clusters: Vec<Vec<usize>> = Vec::new();

        for i in 0..n {
            if visited[i] {
                continue;
            }
            let mut cluster = vec![i];
            visited[i] = true;
            for j in (i + 1)..n {
                if !visited[j] && Self::cosine_similarity(&embeddings[i], &embeddings[j]) >= threshold {
                    cluster.push(j);
                    visited[j] = true;
                }
            }
            clusters.push(cluster);
        }
        clusters
    }
}

// ===========================================================================
// Tests (40+)
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // ----- TileEmbedding tests (10) -----

    #[test]
    fn test_embedder_creation_dimensions() {
        let emb = TileEmbedding::new(4, 8);
        assert_eq!(emb.projection.len(), 8);
        assert_eq!(emb.projection[0].len(), 4);
        assert_eq!(emb.dims, 8);
        assert_eq!(emb.input_dim, 4);
    }

    #[test]
    fn test_embed_output_length() {
        let emb = TileEmbedding::new(3, 5);
        let result = emb.embed(&[1.0, 2.0, 3.0], 5);
        assert_eq!(result.len(), 5);
    }

    #[test]
    fn test_embed_zeros_gives_zeros() {
        let emb = TileEmbedding::new(4, 4);
        let result = emb.embed(&[0.0, 0.0, 0.0, 0.0], 4);
        assert!(result.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn test_embed_deterministic() {
        let emb1 = TileEmbedding::new(5, 3);
        let emb2 = TileEmbedding::new(5, 3);
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let r1 = emb1.embed(&data, 3);
        let r2 = emb2.embed(&data, 3);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_embed_shorter_input_pads() {
        let emb = TileEmbedding::new(4, 2);
        // Input has 2 elements, projection expects 4 — missing are treated as 0.0
        let result = emb.embed(&[1.0, 2.0], 2);
        assert_eq!(result.len(), 2);
        // Should not panic
    }

    #[test]
    fn test_embed_different_inputs_different_outputs() {
        let emb = TileEmbedding::new(4, 4);
        let r1 = emb.embed(&[1.0, 0.0, 0.0, 0.0], 4);
        let r2 = emb.embed(&[0.0, 1.0, 0.0, 0.0], 4);
        assert_ne!(r1, r2);
    }

    #[test]
    fn test_embed_same_input_same_output() {
        let emb = TileEmbedding::new(3, 6);
        let data = vec![0.5, -0.3, 0.8];
        let r1 = emb.embed(&data, 6);
        let r2 = emb.embed(&data, 6);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_embed_nonzero_input_nonzero_output() {
        let emb = TileEmbedding::new(4, 4);
        let result = emb.embed(&[1.0, 1.0, 1.0, 1.0], 4);
        // With non-zero projection and non-zero input, output should be non-zero
        assert!(result.iter().any(|v| v.abs() > 0.01));
    }

    #[test]
    #[should_panic(expected = "requested dims")]
    fn test_embed_wrong_dims_panics() {
        let emb = TileEmbedding::new(4, 8);
        emb.embed(&[1.0, 2.0, 3.0, 4.0], 5);
    }

    #[test]
    fn test_embed_large_input_dim() {
        let emb = TileEmbedding::new(100, 8);
        let data: Vec<f64> = (0..100).map(|i| i as f64 * 0.01).collect();
        let result = emb.embed(&data, 8);
        assert_eq!(result.len(), 8);
    }

    // ----- Predictor tests (8) -----

    #[test]
    fn test_predictor_creation() {
        let pred = Predictor::new(4);
        assert_eq!(pred.weights.len(), 4);
        assert_eq!(pred.weights[0].len(), 4);
        assert_eq!(pred.dim, 4);
    }

    #[test]
    fn test_predict_single_step() {
        let pred = Predictor::new(3);
        let emb = vec![1.0, 0.0, 0.0];
        let result = pred.predict(&emb, 1);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_predict_zero_steps_returns_input() {
        let pred = Predictor::new(3);
        let emb = vec![1.0, 2.0, 3.0];
        let result = pred.predict(&emb, 0);
        assert_eq!(result, emb);
    }

    #[test]
    fn test_predict_multiple_steps() {
        let pred = Predictor::new(4);
        let emb = vec![1.0, 0.5, -0.3, 0.8];
        let r1 = pred.predict(&emb, 1);
        let r3 = pred.predict(&emb, 3);
        assert_eq!(r1.len(), 4);
        assert_eq!(r3.len(), 4);
        // Different steps should generally give different results
        assert_ne!(r1, r3);
    }

    #[test]
    fn test_predict_deterministic() {
        let p1 = Predictor::new(4);
        let p2 = Predictor::new(4);
        let emb = vec![1.0, 2.0, 3.0, 4.0];
        assert_eq!(p1.predict(&emb, 2), p2.predict(&emb, 2));
    }

    #[test]
    fn test_predict_zeros_stay_small() {
        let pred = Predictor::new(4);
        let result = pred.predict(&[0.0; 4], 5);
        assert!(result.iter().all(|v| v.abs() < 1e-12));
    }

    #[test]
    fn test_predict_preserves_dimension() {
        for dim in [1, 4, 8, 16] {
            let pred = Predictor::new(dim);
            let emb = vec![1.0; dim];
            assert_eq!(pred.predict(&emb, 3).len(), dim);
        }
    }

    #[test]
    #[should_panic(expected = "dimension mismatch")]
    fn test_predict_wrong_dim_panics() {
        let pred = Predictor::new(4);
        pred.predict(&[1.0, 2.0], 1);
    }

    // ----- VICRegLoss tests (8) -----

    #[test]
    fn test_vicreg_identical_vectors_low_inv_loss() {
        let vr = VICRegLoss::default();
        let v = vec![1.0, 2.0, 3.0, 4.0];
        let loss = vr.compute_loss(&v, &v);
        // Invariance term should be 0, but variance/covariance terms may add
        // Total loss should be relatively small for identical vectors
        assert!(loss.is_finite());
    }

    #[test]
    fn test_vicreg_different_vectors_higher_loss() {
        let vr = VICRegLoss::default();
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![10.0, 20.0, 30.0, 40.0];
        let loss_same = vr.compute_loss(&a, &a);
        let loss_diff = vr.compute_loss(&a, &b);
        assert!(loss_diff > loss_same);
    }

    #[test]
    fn test_vicreg_zeros() {
        let vr = VICRegLoss::default();
        let loss = vr.compute_loss(&[0.0; 4], &[0.0; 4]);
        // Variance should penalize zero std
        assert!(loss > 0.0);
    }

    #[test]
    fn test_vicreg_custom_weights() {
        let vr = VICRegLoss::new(1.0, 25.0, 25.0, 5.0);
        let a = vec![1.0, 2.0];
        let b = vec![3.0, 4.0];
        let loss = vr.compute_loss(&a, &b);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_vicreg_empty_vectors() {
        let vr = VICRegLoss::default();
        let loss = vr.compute_loss(&[], &[]);
        assert_eq!(loss, 0.0);
    }

    #[test]
    fn test_vicreg_single_dim() {
        let vr = VICRegLoss::new(1.0, 1.0, 1.0, 1.0);
        let loss = vr.compute_loss(&[5.0], &[3.0]);
        assert!(loss.is_finite() && loss > 0.0);
    }

    #[test]
    fn test_vicreg_symmetry() {
        let vr = VICRegLoss::default();
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let lab = vr.compute_loss(&a, &b);
        let lba = vr.compute_loss(&b, &a);
        assert!((lab - lba).abs() < 1e-10, "VICReg loss should be symmetric");
    }

    #[test]
    fn test_vicreg_variance_prevents_collapse() {
        let vr = VICRegLoss::new(5.0, 0.0, 1.0, 0.0); // high threshold, only variance
        let loss = vr.compute_loss(&[0.1, 0.1], &[0.1, 0.1]);
        // Low variance should trigger penalty
        assert!(loss > 0.0);
    }

    // ----- JEPAEncoder tests (8) -----

    #[test]
    fn test_encoder_creation() {
        let enc = JEPAEncoder::new(4, 8);
        assert_eq!(enc.embedder.dims, 8);
        assert_eq!(enc.embedder.input_dim, 4);
        assert_eq!(enc.predictor.dim, 8);
    }

    #[test]
    fn test_encode_and_predict_basic() {
        let enc = JEPAEncoder::new(3, 4);
        let tiles = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let out = enc.encode_and_predict(&tiles, 2, 1);
        assert_eq!(out.context_embedding.len(), 4);
        assert_eq!(out.prediction.len(), 4);
        assert_eq!(out.target_embedding.len(), 4);
        assert!(out.loss.is_finite());
    }

    #[test]
    fn test_encode_and_predict_context_uses_first_tiles() {
        let enc = JEPAEncoder::new(2, 4);
        let tiles = vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]];
        let out = enc.encode_and_predict(&tiles, 2, 1);
        // Target should be the tile at index 2
        let target_emb = enc.embedder.embed(&tiles[2], 4);
        assert_eq!(out.target_embedding, target_emb);
    }

    #[test]
    fn test_encode_and_predict_zero_steps() {
        let enc = JEPAEncoder::new(2, 4);
        let tiles = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
        let out = enc.encode_and_predict(&tiles, 1, 0);
        // With 0 prediction steps, prediction == context_embedding
        assert_eq!(out.prediction, out.context_embedding);
    }

    #[test]
    fn test_encode_and_predict_loss_positive() {
        let enc = JEPAEncoder::new(3, 4);
        let tiles = vec![
            vec![1.0, 2.0, 3.0],
            vec![4.0, 5.0, 6.0],
            vec![7.0, 8.0, 9.0],
        ];
        let out = enc.encode_and_predict(&tiles, 2, 1);
        // Loss should generally be positive (prediction ≠ target)
        assert!(out.loss >= 0.0);
    }

    #[test]
    fn test_encode_and_predict_single_tile() {
        let enc = JEPAEncoder::new(3, 4);
        let tiles = vec![vec![1.0, 2.0, 3.0]];
        let out = enc.encode_and_predict(&tiles, 1, 1);
        assert_eq!(out.context_embedding.len(), 4);
        assert_eq!(out.prediction.len(), 4);
    }

    #[test]
    fn test_encode_and_predict_deterministic() {
        let enc1 = JEPAEncoder::new(3, 4);
        let enc2 = JEPAEncoder::new(3, 4);
        let tiles = vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]];
        let out1 = enc1.encode_and_predict(&tiles, 1, 2);
        let out2 = enc2.encode_and_predict(&tiles, 1, 2);
        assert_eq!(out1.context_embedding, out2.context_embedding);
        assert_eq!(out1.prediction, out2.prediction);
    }

    #[test]
    #[should_panic(expected = "need at least one tile")]
    fn test_encode_and_predict_empty_panics() {
        let enc = JEPAEncoder::new(3, 4);
        enc.encode_and_predict(&[], 1, 1);
    }

    // ----- TileSimilarity tests (8) -----

    #[test]
    fn test_cosine_similarity_identical() {
        let v = vec![1.0, 2.0, 3.0];
        let sim = TileSimilarity::cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        let sim = TileSimilarity::cosine_similarity(&a, &b);
        assert!((sim - (-1.0)).abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let sim = TileSimilarity::cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-10);
    }

    #[test]
    fn test_cosine_similarity_zeros() {
        let sim = TileSimilarity::cosine_similarity(&[0.0; 3], &[1.0, 2.0, 3.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_similarity_symmetric() {
        let a = vec![1.0, 2.0, 3.0];
        let b = vec![4.0, 5.0, 6.0];
        let sab = TileSimilarity::cosine_similarity(&a, &b);
        let sba = TileSimilarity::cosine_similarity(&b, &a);
        assert!((sab - sba).abs() < 1e-10);
    }

    #[test]
    fn test_knn_returns_correct_count() {
        let query = vec![1.0, 0.0, 0.0];
        let tiles = vec![
            vec![0.9, 0.1, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.8, 0.2, 0.0],
        ];
        let knn = TileSimilarity::knn_tiles(&query, &tiles, 2);
        assert_eq!(knn.len(), 2);
    }

    #[test]
    fn test_knn_most_similar_first() {
        let query = vec![1.0, 0.0];
        let tiles = vec![
            vec![0.5, 0.5],  // 45°
            vec![1.0, 0.0],  // identical
            vec![0.0, 1.0],  // orthogonal
        ];
        let knn = TileSimilarity::knn_tiles(&query, &tiles, 3);
        assert_eq!(knn[0], 1); // identical should be first
    }

    #[test]
    fn test_cluster_by_similarity() {
        // Two groups: [a,b] similar, [c] different
        let a = vec![1.0, 0.0];
        let b = vec![0.95, 0.05];
        let c = vec![0.0, 1.0];
        let clusters = TileSimilarity::cluster_by_similarity(&[a, b, c], 0.9);
        assert!(clusters.len() >= 1);
        // a and b should be in the same cluster
        let ab_cluster = clusters.iter().find(|c| c.contains(&0) && c.contains(&1));
        assert!(ab_cluster.is_some(), "a and b should cluster together");
    }

    // ----- Integration / misc tests (4+) -----

    #[test]
    fn test_full_pipeline() {
        let enc = JEPAEncoder::new(4, 8);
        let tiles: Vec<Vec<f64>> = (0..5)
            .map(|i| (0..4).map(|j| (i * 4 + j) as f64).collect())
            .collect();

        let out = enc.encode_and_predict(&tiles, 3, 2);
        assert!(out.loss >= 0.0);
        assert_eq!(out.context_embedding.len(), 8);

        // Check knn on embedded tiles
        let embeddings: Vec<Vec<f64>> = tiles.iter().map(|t| enc.embedder.embed(t, 8)).collect();
        let knn = TileSimilarity::knn_tiles(&embeddings[0], &embeddings, 3);
        assert_eq!(knn.len(), 3);
        assert!(knn.contains(&0)); // self should be included
    }

    #[test]
    fn test_no_unsafe_code() {
        // This test exists to ensure #![deny(unsafe_code)] is active.
        // If any unsafe code is introduced, compilation will fail.
    }

    #[test]
    fn test_vicreg_loss_range() {
        let vr = VICRegLoss::default();
        let a = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let b = vec![2.0, 3.0, 4.0, 5.0, 6.0];
        let loss = vr.compute_loss(&a, &b);
        assert!(loss.is_finite());
        assert!(loss >= 0.0);
    }

    #[test]
    fn test_predictor_identity_like() {
        let pred = Predictor::new(4);
        // Diagonal should be 1.0
        for i in 0..4 {
            assert!((pred.weights[i][i] - 1.0).abs() < 1e-10);
        }
    }

    #[test]
    fn test_knn_empty_set() {
        let query = vec![1.0, 2.0];
        let knn = TileSimilarity::knn_tiles(&query, &[], 3);
        assert!(knn.is_empty());
    }

    #[test]
    fn test_cluster_single_element() {
        let clusters = TileSimilarity::cluster_by_similarity(&[vec![1.0, 2.0]], 0.5);
        assert_eq!(clusters.len(), 1);
        assert_eq!(clusters[0], vec![0]);
    }

    #[test]
    fn test_embed_projection_not_all_zero() {
        let emb = TileEmbedding::new(4, 4);
        // Projection matrix should have non-zero entries
        assert!(emb.projection.iter().flat_map(|r| r.iter()).any(|v| v.abs() > 0.01));
    }

    #[test]
    fn test_vicreg_default_weights() {
        let vr = VICRegLoss::default();
        assert_eq!(vr.variance_threshold, 1.0);
        assert_eq!(vr.mu, 10.0);
        assert_eq!(vr.lambda, 10.0);
        assert_eq!(vr.nu, 1.0);
    }

    #[test]
    fn test_jepa_output_serialization() {
        let out = JEPAOutput {
            context_embedding: vec![1.0, 2.0],
            prediction: vec![1.5, 2.5],
            target_embedding: vec![2.0, 3.0],
            loss: 0.5,
        };
        let json = serde_json::to_string(&out).unwrap();
        let parsed: JEPAOutput = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.loss, 0.5);
        assert_eq!(parsed.context_embedding, vec![1.0, 2.0]);
    }
}
