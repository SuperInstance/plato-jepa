use plato_jepa::*;
use uuid::Uuid;

// Helper: make an embedding with known vector
fn make_embedding(vals: &[f64]) -> TileEmbedding {
    let dim = vals.len();
    TileEmbedding {
        vector: vals.to_vec(),
        dimension: dim,
        tile_id: Uuid::new_v4(),
    }
}

fn make_embedding_with_id(vals: &[f64], id: Uuid) -> TileEmbedding {
    TileEmbedding {
        vector: vals.to_vec(),
        dimension: vals.len(),
        tile_id: id,
    }
}

// ── Embedding creation ──────────────────────────────────────────────────────

#[test]
fn from_tile_values_basic() {
    let e = TileEmbedding::from_tile_values(&[1.0, 2.0, 3.0], 3);
    assert_eq!(e.vector, vec![1.0, 2.0, 3.0]);
    assert_eq!(e.dimension, 3);
}

#[test]
fn from_tile_values_pads_with_zeros() {
    let e = TileEmbedding::from_tile_values(&[1.0, 2.0], 5);
    assert_eq!(e.vector, vec![1.0, 2.0, 0.0, 0.0, 0.0]);
    assert_eq!(e.dimension, 5);
}

#[test]
fn from_tile_values_truncates() {
    let e = TileEmbedding::from_tile_values(&[1.0, 2.0, 3.0, 4.0], 2);
    assert_eq!(e.vector, vec![1.0, 2.0]);
}

#[test]
fn embedding_has_unique_id() {
    let a = TileEmbedding::from_tile_values(&[1.0], 1);
    let b = TileEmbedding::from_tile_values(&[1.0], 1);
    assert_ne!(a.tile_id, b.tile_id);
}

// ── Cosine similarity ──────────────────────────────────────────────────────

#[test]
fn cosine_identical_is_one() {
    let a = make_embedding(&[1.0, 0.0, 0.0]);
    let b = make_embedding(&[1.0, 0.0, 0.0]);
    let sim = TileEmbedding::cosine_similarity(&a, &b);
    assert!((sim - 1.0).abs() < 1e-10);
}

#[test]
fn cosine_orthogonal_is_zero() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[0.0, 1.0]);
    let sim = TileEmbedding::cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-10);
}

#[test]
fn cosine_opposite_is_minus_one() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[-1.0, 0.0]);
    let sim = TileEmbedding::cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 1e-10);
}

#[test]
fn cosine_zero_vector_is_zero() {
    let a = make_embedding(&[0.0, 0.0]);
    let b = make_embedding(&[1.0, 0.0]);
    assert_eq!(TileEmbedding::cosine_similarity(&a, &b), 0.0);
}

// ── Euclidean distance ──────────────────────────────────────────────────────

#[test]
fn euclidean_identical_is_zero() {
    let a = make_embedding(&[1.0, 2.0, 3.0]);
    assert_eq!(TileEmbedding::euclidean_distance(&a, &a), 0.0);
}

#[test]
fn euclidean_known_distance() {
    let a = make_embedding(&[0.0, 0.0]);
    let b = make_embedding(&[3.0, 4.0]);
    assert!((TileEmbedding::euclidean_distance(&a, &b) - 5.0).abs() < 1e-10);
}

// ── Latent space ────────────────────────────────────────────────────────────

#[test]
fn latent_space_insert_and_retrieve() {
    let mut space = LatentSpace::new(3);
    let id = Uuid::new_v4();
    let e = make_embedding_with_id(&[1.0, 2.0, 3.0], id);
    space.insert(e);
    assert!(space.embeddings.contains_key(&id));
    assert_eq!(space.embeddings.len(), 1);
}

#[test]
fn latent_space_nearest_neighbors() {
    let mut space = LatentSpace::new(2);
    let origin = make_embedding_with_id(&[0.0, 0.0], Uuid::new_v4());
    let near = make_embedding_with_id(&[1.0, 0.0], Uuid::new_v4());
    let far = make_embedding_with_id(&[10.0, 0.0], Uuid::new_v4());
    let nid = near.tile_id;
    space.insert(origin);
    space.insert(near);
    space.insert(far);

    let query = make_embedding_with_id(&[0.9, 0.0], Uuid::new_v4());
    let nn = space.nearest_neighbors(&query, 2);
    assert_eq!(nn.len(), 2);
    assert_eq!(nn[0].0.tile_id, nid); // near is closest (dist=0.1)
    assert!(nn[0].1 < nn[1].1); // sorted ascending
}

#[test]
fn latent_space_nearest_neighbors_empty() {
    let space = LatentSpace::new(2);
    let query = make_embedding(&[1.0, 2.0]);
    let nn = space.nearest_neighbors(&query, 3);
    assert!(nn.is_empty());
}

#[test]
fn latent_space_clustering() {
    let mut space = LatentSpace::new(2);
    let a = make_embedding_with_id(&[1.0, 0.0], Uuid::new_v4());
    let b = make_embedding_with_id(&[0.99, 0.0], Uuid::new_v4());
    let c = make_embedding_with_id(&[0.0, 1.0], Uuid::new_v4());
    space.insert(a.clone());
    space.insert(b.clone());
    space.insert(c.clone());

    let clusters = space.cluster_by_similarity(0.99);
    // a and b should cluster together, c separate
    assert_eq!(clusters.len(), 2);
    let ids: Vec<Uuid> = clusters.iter().flatten().copied().collect();
    assert!(ids.contains(&a.tile_id));
    assert!(ids.contains(&b.tile_id));
    assert!(ids.contains(&c.tile_id));
}

#[test]
fn latent_space_single_embedding_cluster() {
    let mut space = LatentSpace::new(2);
    let e = make_embedding_with_id(&[1.0, 0.0], Uuid::new_v4());
    space.insert(e);
    let clusters = space.cluster_by_similarity(0.5);
    assert_eq!(clusters.len(), 1);
    assert_eq!(clusters[0].len(), 1);
}

// ── Predictor ───────────────────────────────────────────────────────────────

#[test]
fn predictor_creation() {
    let config = JepaConfig::default();
    let p = Predictor::new(config.clone());
    assert_eq!(p.weights.len(), config.embedding_dim);
    assert!(!p.trained);
}

#[test]
fn predictor_predict_basic() {
    let config = JepaConfig {
        embedding_dim: 4,
        context_window: 3,
        ..JepaConfig::default()
    };
    let p = Predictor::new(config);
    let past = vec![
        make_embedding(&[1.0, 2.0, 3.0, 4.0]),
        make_embedding(&[1.0, 2.0, 3.0, 4.0]),
    ];
    let target = make_embedding(&[1.0, 2.0, 3.0, 4.0]);
    let ctx = PredictionContext {
        past_tiles: past,
        target_tile: Some(target.clone()),
    };
    let result = p.predict(&ctx);
    assert_eq!(result.predicted.dimension, 4);
    assert!(result.loss >= 0.0);
    assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
}

#[test]
fn predictor_predict_no_context() {
    let p = Predictor::new(JepaConfig::default());
    let ctx = PredictionContext {
        past_tiles: vec![],
        target_tile: None,
    };
    let result = p.predict(&ctx);
    assert!(result.predicted.vector.iter().all(|v| *v == 0.0));
    assert_eq!(result.confidence, 0.0);
}

#[test]
fn compute_loss_identical_is_zero() {
    let a = make_embedding(&[1.0, 2.0, 3.0]);
    assert_eq!(Predictor::compute_loss(&a, &a), 0.0);
}

#[test]
fn compute_loss_positive_for_different() {
    let a = make_embedding(&[0.0, 0.0, 0.0]);
    let b = make_embedding(&[1.0, 1.0, 1.0]);
    let loss = Predictor::compute_loss(&a, &b);
    assert!(loss > 0.0);
    assert!((loss - 1.0).abs() < 1e-10); // 3/3 = 1.0
}

// ── Variance regulation ─────────────────────────────────────────────────────

#[test]
fn variance_regulation_collapsed_is_zero() {
    let e = make_embedding(&[1.0, 2.0]);
    let embeddings = vec![e.clone(), e.clone(), e.clone()];
    assert_eq!(Predictor::compute_variance_regulation(&embeddings), 0.0);
}

#[test]
fn variance_regulation_diverse_positive() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[0.0, 1.0]);
    let var = Predictor::compute_variance_regulation(&[a, b]);
    assert!(var > 0.0);
}

#[test]
fn variance_regulation_single_is_zero() {
    let e = make_embedding(&[5.0, 5.0]);
    assert_eq!(Predictor::compute_variance_regulation(&[e]), 0.0);
}

// ── JEPA free functions ─────────────────────────────────────────────────────

#[test]
fn collapse_score_all_same() {
    let e = make_embedding(&[1.0, 2.0, 3.0]);
    let score = collapse_score(&[e.clone(), e.clone(), e.clone()]);
    assert!((score - 1.0).abs() < 1e-10);
}

#[test]
fn collapse_score_diverse() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[0.0, 1.0]);
    let score = collapse_score(&[a, b]);
    assert!(score < 0.1); // orthogonal → sim ~0
}

#[test]
fn collapse_score_opposite() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[-1.0, 0.0]);
    let score = collapse_score(&[a, b]);
    assert!((score + 1.0).abs() < 1e-10); // sim = -1.0
}

#[test]
fn collapse_score_empty() {
    assert_eq!(collapse_score(&[]), 0.0);
}

#[test]
fn information_content_collapsed_low() {
    let e = make_embedding(&[1.0, 1.0]);
    let ic = information_content(&[e.clone(), e.clone()]);
    assert!(ic < -20.0); // near-zero variance → very negative log
}

#[test]
fn information_content_diverse_high() {
    let a = make_embedding(&[1.0, 0.0]);
    let b = make_embedding(&[0.0, 1.0]);
    let ic = information_content(&[a, b]);
    assert!(ic > -5.0); // non-trivial variance
}

#[test]
fn information_content_empty() {
    assert_eq!(information_content(&[]), 0.0);
}

#[test]
fn prediction_difficulty_empty_context() {
    let target = make_embedding(&[1.0, 2.0]);
    assert_eq!(prediction_difficulty(&[], &target), 1.0);
}

#[test]
fn prediction_difficulty_increases_with_distance() {
    let context = vec![make_embedding(&[0.0, 0.0])];
    let near = make_embedding(&[0.1, 0.0]);
    let far = make_embedding(&[100.0, 0.0]);
    let d_near = prediction_difficulty(&context, &near);
    let d_far = prediction_difficulty(&context, &far);
    assert!(d_far > d_near);
}

// ── Serialization ───────────────────────────────────────────────────────────

#[test]
fn serde_roundtrip_embedding() {
    let e = TileEmbedding::from_tile_values(&[1.0, 2.0, 3.0], 3);
    let json = serde_json::to_string(&e).unwrap();
    let de: TileEmbedding = serde_json::from_str(&json).unwrap();
    assert_eq!(de.vector, e.vector);
    assert_eq!(de.tile_id, e.tile_id);
}

#[test]
fn serde_roundtrip_jepa_config() {
    let c = JepaConfig::default();
    let json = serde_json::to_string(&c).unwrap();
    let de: JepaConfig = serde_json::from_str(&json).unwrap();
    assert_eq!(de.embedding_dim, c.embedding_dim);
    assert_eq!(de.learning_rate, c.learning_rate);
}

#[test]
fn serde_roundtrip_prediction_result() {
    let r = PredictionResult {
        predicted: make_embedding(&[1.0, 2.0]),
        actual: Some(make_embedding(&[1.5, 2.5])),
        loss: 0.25,
        confidence: 0.8,
    };
    let json = serde_json::to_string(&r).unwrap();
    let de: PredictionResult = serde_json::from_str(&json).unwrap();
    assert!((de.loss - 0.25).abs() < 1e-10);
}

// ── Context window slicing ──────────────────────────────────────────────────

#[test]
fn context_window_respected() {
    let config = JepaConfig {
        embedding_dim: 2,
        context_window: 2,
        ..JepaConfig::default()
    };
    let p = Predictor::new(config);
    let past = vec![
        make_embedding(&[1.0, 1.0]),
        make_embedding(&[2.0, 2.0]),
        make_embedding(&[3.0, 3.0]),
        make_embedding(&[4.0, 4.0]),
    ];
    let ctx = PredictionContext {
        past_tiles: past,
        target_tile: Some(make_embedding(&[5.0, 5.0])),
    };
    let result = p.predict(&ctx);
    // With window=2 and equal weights, prediction should be avg of last 2 tiles: [3.5, 3.5]
    assert!((result.predicted.vector[0] - 3.5).abs() < 1e-10);
    assert!((result.predicted.vector[1] - 3.5).abs() < 1e-10);
}

// ── High-dimensional edge case ──────────────────────────────────────────────

#[test]
fn high_dimensional_embeddings() {
    let dim = 512;
    let vals: Vec<f64> = (0..dim).map(|i| (i as f64).sin()).collect();
    let e = TileEmbedding::from_tile_values(&vals, dim);
    assert_eq!(e.dimension, 512);
    assert!((TileEmbedding::cosine_similarity(&e, &e) - 1.0).abs() < 1e-10);
}
