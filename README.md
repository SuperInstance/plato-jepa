# plato-jepa

> Joint-Embedding Predictive Architecture primitives for tile representation learning in PLATO rooms

## What This Does

plato-jepa implements the building blocks for JEPA-style self-supervised representation learning on PLATO tiles. Instead of predicting pixel-level reconstructions, JEPA predicts *latent representations* of future tiles from past context — learning the structure of the data without labels and without representation collapse.

## The Key Idea

Traditional autoencoders compress and reconstruct. JEPA says: don't reconstruct — *predict*. Given tiles from the last 5 minutes, predict the latent representation of the next tile. The future is your supervision.

The hard part is "representation collapse" — if the model maps everything to the same point, loss is zero but you've learned nothing. plato-jepa provides `collapse_score`, variance regulation, and information content metrics to keep representations diverse and useful.

## Install

```bash
cargo add plato-jepa
```

## Quick Start

```rust
use plato_jepa::{TileEmbedding, Predictor, JepaConfig, PredictionContext, collapse_score};

let past_tiles: Vec<TileEmbedding> = (0..5)
    .map(|i| TileEmbedding::from_tile_values(&[i as f64, (i as f64 * 0.5).sin()], 2))
    .collect();
let target = TileEmbedding::from_tile_values(&[5.0, (5.0 * 0.5).sin()], 2);

let predictor = Predictor::new(JepaConfig::default());
let ctx = PredictionContext { past_tiles, target_tile: Some(target) };
let result = predictor.predict(&ctx);
println!("Loss: {:.4}, Confidence: {:.1}%", result.loss, result.confidence * 100.0);

// Check representation health
let embeddings = vec![
    TileEmbedding::from_tile_values(&[1.0, 0.0], 2),
    TileEmbedding::from_tile_values(&[0.0, 1.0], 2),
];
println!("Collapse score: {:.2} (0=diverse, 1=collapsed)", collapse_score(&embeddings));
```

## API Reference

### Core Types

| Type | Description |
|---|---|
| `TileEmbedding { vector, dimension, tile_id }` | A tile's latent representation with unique UUID |
| `PredictionContext { past_tiles, target_tile }` | Context window + optional target |
| `PredictionResult { predicted, actual, loss, confidence }` | Prediction output |
| `JepaConfig` | Config: `embedding_dim`(16), `context_window`(5), `prediction_horizon`(1), `learning_rate`(0.01) |
| `LatentSpace` | Indexed embeddings with k-NN search and similarity clustering |
| `Predictor` | Linear predictor mapping context windows to next-tile predictions |

### TileEmbedding

```rust
let e = TileEmbedding::from_tile_values(&[1.0, 2.0, 3.0], 5);
// vector: [1.0, 2.0, 3.0, 0.0, 0.0] — pads or truncates to dim

TileEmbedding::cosine_similarity(&a, &b);  // -1.0 to 1.0
TileEmbedding::euclidean_distance(&a, &b); // 0.0 to ∞
```

### LatentSpace

```rust
let mut space = LatentSpace::new(16);
space.insert(embedding);
let neighbors = space.nearest_neighbors(&query, 5);
let clusters = space.cluster_by_similarity(0.9);
```

### Predictor

```rust
let result = predictor.predict(&ctx);
Predictor::compute_loss(&predicted, &actual); // MSE
Predictor::compute_variance_regulation(&embeddings); // avg per-dim variance
```

### Free Functions

| Function | Description |
|---|---|
| `collapse_score(embeddings) -> f64` | Avg pairwise cosine similarity. 0=diverse, 1=collapsed, -1=opposite. |
| `information_content(embeddings) -> f64` | Per-dimension entropy estimate via log(σ²+ε). Higher = more info. |
| `prediction_difficulty(context, target) -> f64` | Difficulty [0,1] via sigmoid of avg distance. |

## How It Works

The predictor is a linear model: weight matrix (dim × context_window) × flattened context. Each output dimension is a weighted average of that dimension across past tiles. Loss is MSE; confidence is 1/(1+loss). Collapse detection uses average pairwise cosine similarity — all identical vectors score 1.0.

## Testing

37 tests: embedding creation/padding/truncation, cosine similarity (identical/orthogonal/opposite/zero), Euclidean distance, k-NN, clustering, predictor with/without context, loss, variance regulation, collapse scores, information content, prediction difficulty, context window slicing, 512-d embeddings, serde round-trips.

## License

Apache-2.0
