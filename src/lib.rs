use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TileEmbedding {
    pub vector: Vec<f64>,
    pub dimension: usize,
    pub tile_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionContext {
    pub past_tiles: Vec<TileEmbedding>,
    pub target_tile: Option<TileEmbedding>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionResult {
    pub predicted: TileEmbedding,
    pub actual: Option<TileEmbedding>,
    pub loss: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JepaConfig {
    pub embedding_dim: usize,
    pub context_window: usize,
    pub prediction_horizon: usize,
    pub learning_rate: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatentSpace {
    pub embeddings: HashMap<Uuid, TileEmbedding>,
    pub dimension: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Predictor {
    pub weights: Vec<Vec<f64>>,
    pub config: JepaConfig,
    pub trained: bool,
}

// ---------------------------------------------------------------------------
// JepaConfig
// ---------------------------------------------------------------------------

impl Default for JepaConfig {
    fn default() -> Self {
        Self {
            embedding_dim: 16,
            context_window: 5,
            prediction_horizon: 1,
            learning_rate: 0.01,
        }
    }
}

// ---------------------------------------------------------------------------
// TileEmbedding
// ---------------------------------------------------------------------------

impl TileEmbedding {
    /// Simple projection: pad or truncate `values` to `dim` dimensions.
    pub fn from_tile_values(values: &[f64], dim: usize) -> Self {
        let mut vector = values.to_vec();
        vector.resize(dim, 0.0);
        Self {
            vector,
            dimension: dim,
            tile_id: Uuid::new_v4(),
        }
    }

    fn norm(&self) -> f64 {
        self.vector.iter().map(|v| v * v).sum::<f64>().sqrt()
    }

    pub fn cosine_similarity(a: &Self, b: &Self) -> f64 {
        let dot: f64 = a.vector.iter().zip(&b.vector).map(|(x, y)| x * y).sum();
        let na = a.norm();
        let nb = b.norm();
        if na == 0.0 || nb == 0.0 {
            return 0.0;
        }
        dot / (na * nb)
    }

    pub fn euclidean_distance(a: &Self, b: &Self) -> f64 {
        a.vector
            .iter()
            .zip(&b.vector)
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

// ---------------------------------------------------------------------------
// LatentSpace
// ---------------------------------------------------------------------------

impl LatentSpace {
    pub fn new(dimension: usize) -> Self {
        Self {
            embeddings: HashMap::new(),
            dimension,
        }
    }

    pub fn insert(&mut self, embedding: TileEmbedding) {
        self.embeddings.insert(embedding.tile_id, embedding);
    }

    /// Return the k nearest neighbors (embedding ref, distance) sorted ascending.
    pub fn nearest_neighbors(&self, query: &TileEmbedding, k: usize) -> Vec<(&TileEmbedding, f64)> {
        let mut scored: Vec<_> = self
            .embeddings
            .values()
            .map(|e| (e, TileEmbedding::euclidean_distance(query, e)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }

    /// Group tile ids into clusters where every pair has cosine similarity >= threshold.
    pub fn cluster_by_similarity(&self, threshold: f64) -> Vec<Vec<Uuid>> {
        let ids: Vec<Uuid> = self.embeddings.keys().copied().collect();
        let mut visited = vec![false; ids.len()];
        let mut clusters: Vec<Vec<Uuid>> = Vec::new();

        for i in 0..ids.len() {
            if visited[i] {
                continue;
            }
            let mut cluster = vec![ids[i]];
            visited[i] = true;
            let ei = &self.embeddings[&ids[i]];
            for j in (i + 1)..ids.len() {
                if visited[j] {
                    continue;
                }
                let ej = &self.embeddings[&ids[j]];
                if TileEmbedding::cosine_similarity(ei, ej) >= threshold {
                    cluster.push(ids[j]);
                    visited[j] = true;
                }
            }
            clusters.push(cluster);
        }
        clusters
    }
}

// ---------------------------------------------------------------------------
// Predictor
// ---------------------------------------------------------------------------

impl Predictor {
    pub fn new(config: JepaConfig) -> Self {
        let dim = config.embedding_dim;
        let ctx = config.context_window;
        // Simple linear predictor: dim×ctx weights applied to flattened context
        let mut weights = Vec::with_capacity(dim);
        for _ in 0..dim {
            let row = vec![1.0 / (ctx as f64); ctx];
            weights.push(row);
        }
        Self {
            weights,
            config,
            trained: false,
        }
    }

    /// Predict next embedding by averaging past context (each dimension independently).
    pub fn predict(&self, context: &PredictionContext) -> PredictionResult {
        let dim = self.config.embedding_dim;
        let past: Vec<&TileEmbedding> = context
            .past_tiles
            .iter()
            .rev()
            .take(self.config.context_window)
            .collect();

        let n = past.len().max(1) as f64;
        let mut predicted_vec = vec![0.0; dim];

        if past.is_empty() {
            // Zero prediction when no context
            return PredictionResult {
                predicted: TileEmbedding {
                    vector: predicted_vec,
                    dimension: dim,
                    tile_id: Uuid::new_v4(),
                },
                actual: context.target_tile.clone(),
                loss: 0.0,
                confidence: 0.0,
            };
        }

        // Weighted average using predictor weights (dim × ctx matrix × context vectors)
        for (d, row) in self.weights.iter().enumerate() {
            let mut sum = 0.0;
            for (t, te) in past.iter().enumerate() {
                let w = row.get(t).copied().unwrap_or(1.0 / n);
                let v = te.vector.get(d).copied().unwrap_or(0.0);
                sum += w * v;
            }
            predicted_vec[d] = sum;
        }

        let predicted = TileEmbedding {
            vector: predicted_vec,
            dimension: dim,
            tile_id: Uuid::new_v4(),
        };

        let (loss, confidence) = match &context.target_tile {
            Some(actual) => {
                let l = Self::compute_loss(&predicted, actual);
                let c = (1.0 / (1.0 + l)).clamp(0.0, 1.0);
                (l, c)
            }
            None => (0.0, 0.0),
        };

        PredictionResult {
            predicted,
            actual: context.target_tile.clone(),
            loss,
            confidence,
        }
    }

    /// MSE in embedding space.
    pub fn compute_loss(predicted: &TileEmbedding, actual: &TileEmbedding) -> f64 {
        let n = predicted.dimension.max(actual.dimension) as f64;
        predicted
            .vector
            .iter()
            .zip(&actual.vector)
            .map(|(p, a)| (p - a).powi(2))
            .sum::<f64>()
            / n
    }

    /// JEPA collapse prevention: average pairwise variance across dimensions.
    pub fn compute_variance_regulation(embeddings: &[TileEmbedding]) -> f64 {
        if embeddings.len() < 2 {
            return 0.0;
        }
        let dim = embeddings[0].dimension;
        let mut total_var = 0.0;
        for d in 0..dim {
            let vals: Vec<f64> = embeddings.iter().map(|e| e.vector[d]).collect();
            let mean = vals.iter().sum::<f64>() / vals.len() as f64;
            let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
            total_var += var;
        }
        total_var / dim as f64
    }
}

// ---------------------------------------------------------------------------
// JEPA-specific free functions
// ---------------------------------------------------------------------------

/// 0.0 = diverse representations, 1.0 = all collapsed to the same point.
pub fn collapse_score(embeddings: &[TileEmbedding]) -> f64 {
    if embeddings.len() < 2 {
        return 0.0;
    }
    // Average pairwise cosine similarity → collapse proxy
    let mut total_sim = 0.0;
    let mut count = 0usize;
    for i in 0..embeddings.len() {
        for j in (i + 1)..embeddings.len() {
            total_sim += TileEmbedding::cosine_similarity(&embeddings[i], &embeddings[j]);
            count += 1;
        }
    }
    // Map [-1,1] similarity to [0,1] collapse: (sim+1)/2 then average
    // But collapse is really about all being identical, so just use sim directly
    // shifted: (sim - (-1)) / 2 gives 0..1, but we want 1.0 when all same
    total_sim / count as f64
}

/// Entropy estimate of the representation distribution.
pub fn information_content(embeddings: &[TileEmbedding]) -> f64 {
    if embeddings.is_empty() {
        return 0.0;
    }
    let dim = embeddings[0].dimension;
    // Per-dimension entropy estimate using variance
    let mut total_entropy = 0.0;
    for d in 0..dim {
        let vals: Vec<f64> = embeddings.iter().map(|e| e.vector.get(d).copied().unwrap_or(0.0)).collect();
        let mean = vals.iter().sum::<f64>() / vals.len() as f64;
        let var: f64 = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
        // Entropy ∝ log(σ² + ε) — higher variance = more information
        total_entropy += (var + 1e-10).ln();
    }
    total_entropy / dim as f64
}

/// How hard is the prediction: increases with distance/noise between context and target.
pub fn prediction_difficulty(context: &[TileEmbedding], target: &TileEmbedding) -> f64 {
    if context.is_empty() {
        return 1.0;
    }
    let avg_dist: f64 = context
        .iter()
        .map(|c| TileEmbedding::euclidean_distance(c, target))
        .sum::<f64>()
        / context.len() as f64;
    // Normalize to [0,1] via sigmoid-like mapping
    1.0 - 1.0 / (1.0 + avg_dist)
}
