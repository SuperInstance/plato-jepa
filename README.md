# plato-jepa

JEPA (Joint-Embedding Predictive Architecture) primitives for tile representation learning in PLATO rooms.

## Overview

This crate provides the core building blocks for a JEPA-style self-supervised learning pipeline:

- **TileEmbedding** — dense vector representations of spatial tiles with cosine similarity and Euclidean distance
- **LatentSpace** — an in-memory embedding store with nearest-neighbor search and similarity clustering
- **Predictor** — a linear predictor that forecasts the next tile embedding from a context window of past tiles
- **Collapse detection** — utilities (`collapse_score`, `information_content`, `prediction_difficulty`) to monitor representation quality and prevent mode collapse

## Usage

```rust
use plato_jepa::*;

let config = JepaConfig::default();
let predictor = Predictor::new(config);

let past = vec![
    TileEmbedding::from_tile_values(&[1.0, 2.0, 3.0], 3),
    TileEmbedding::from_tile_values(&[1.1, 2.1, 3.1], 3),
];

let ctx = PredictionContext {
    past_tiles: past,
    target_tile: Some(TileEmbedding::from_tile_values(&[1.2, 2.2, 3.2], 3)),
};

let result = predictor.predict(&ctx);
println!("predicted: {:?}", result.predicted.vector);
println!("loss: {:.4}, confidence: {:.4}", result.loss, result.confidence);
```

## Features

- Serde serialization for all core types
- Collapse score and variance regulation for JEPA training monitoring
- Prediction difficulty estimation
- Configurable embedding dimension, context window, and prediction horizon

## License

Apache-2.0
