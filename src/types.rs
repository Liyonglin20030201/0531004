#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TimeSeriesPoint {
    pub timestamp: i64,
    pub value: f64,
}

pub type SeriesId = u64;
