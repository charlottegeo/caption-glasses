use ndarray::Array1;
use ort::Session;
use std::collections::HashMap;
use std::sync::Mutex;

const BASELINE_EMA_ALPHA: f32 = 0.15;
const SPIKE_MARGIN: f32 = 0.18;

const CATEGORY_RANGES: &[(&str, std::ops::Range<usize>)] = &[
    ("human", 0..67),
    ("animal", 67..132),
    ("music", 132..277),
    ("natural", 277..294),
    ("vehicle", 294..348),
    ("domestic", 348..412),
    ("tools", 412..420),
    ("explosive", 420..456),
    ("tones", 456..499),
    ("ambient", 499..521),
];

pub struct YamnetClassifier {
    session: Session,
    labels: Vec<String>,
    category_baseline: Mutex<HashMap<&'static str, f32>>,
}

pub struct Detection {
    pub label: String,
    pub category: &'static str,
    pub score: f32,
}

impl YamnetClassifier {
    pub fn load(model_path: &str, class_map_path: &str) -> anyhow::Result<Self> {
        let session = Session::builder()?.commit_from_file(model_path)?;
        let labels = load_class_names(class_map_path)?;
        Ok(Self {
            session,
            labels,
            category_baseline: Mutex::new(HashMap::new()),
        })
    }

    pub fn classify(&self, pcm: &[f32], threshold: f32) -> Option<Detection> {
        let input = Array1::from_vec(pcm.to_vec());
        let outputs = self
            .session
            .run(ort::inputs!["waveform" => input.view()].ok()?)
            .ok()?;
        let scores = outputs.get("output_0")?.try_extract_tensor::<f32>().ok()?;
        let scores = scores.view();
        let shape = scores.shape();
        let (frames, classes) = (shape[0], shape[1]);
        if frames == 0 {
            return None;
        }

        let mut averaged = vec![0f32; classes];
        for frame in 0..frames {
            for class in 0..classes {
                averaged[class] += scores[[frame, class]];
            }
        }
        for v in &mut averaged {
            *v /= frames as f32;
        }

        let (best_idx, &best_score) = averaged
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())?;
        if best_score < threshold {
            return None;
        }
        let category = category_for_index(best_idx);

        let mut baselines = self.category_baseline.lock().unwrap();
        let baseline = baselines.entry(category).or_insert(0.0);
        let prior_baseline = *baseline;
        *baseline = BASELINE_EMA_ALPHA * best_score + (1.0 - BASELINE_EMA_ALPHA) * prior_baseline;
        drop(baselines);

        if best_score < prior_baseline + SPIKE_MARGIN {
            return None;
        }

        Some(Detection {
            label: self.labels.get(best_idx).cloned().unwrap_or_default(),
            category,
            score: best_score,
        })
    }
}

fn category_for_index(idx: usize) -> &'static str {
    CATEGORY_RANGES
        .iter()
        .find(|(_, range)| range.contains(&idx))
        .map(|(name, _)| *name)
        .unwrap_or("other")
}

fn load_class_names(path: &str) -> anyhow::Result<Vec<String>> {
    let content = std::fs::read_to_string(path)?;
    Ok(content
        .lines()
        .skip(1)
        .filter_map(parse_csv_display_name)
        .collect())
}

fn parse_csv_display_name(line: &str) -> Option<String> {
    let mut fields = Vec::new();
    let mut field = String::new();
    let mut in_quotes = false;
    for c in line.chars() {
        match c {
            '"' => in_quotes = !in_quotes,
            ',' if !in_quotes => fields.push(std::mem::take(&mut field)),
            _ => field.push(c),
        }
    }
    fields.push(field);
    fields.into_iter().nth(2)
}
