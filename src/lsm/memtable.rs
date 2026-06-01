use std::collections::BTreeMap;
use crate::types::TimeSeriesPoint;

pub struct MemTable {
    entries: BTreeMap<i64, f64>,
    size_bytes: usize,
    capacity: usize,
}

impl MemTable {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: BTreeMap::new(),
            size_bytes: 0,
            capacity,
        }
    }

    pub fn insert(&mut self, point: TimeSeriesPoint) {
        let is_new = self.entries.insert(point.timestamp, point.value).is_none();
        if is_new {
            // BTreeMap entry overhead: ~64 bytes (key + value + node pointers)
            self.size_bytes += 16 + 48;
        }
    }

    pub fn is_full(&self) -> bool {
        self.size_bytes >= self.capacity
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn drain_sorted(&mut self) -> Vec<TimeSeriesPoint> {
        let points: Vec<TimeSeriesPoint> = self
            .entries
            .iter()
            .map(|(&timestamp, &value)| TimeSeriesPoint { timestamp, value })
            .collect();
        self.entries.clear();
        self.size_bytes = 0;
        points
    }

    pub fn scan_range(&self, start: i64, end: i64) -> Vec<TimeSeriesPoint> {
        use std::ops::RangeInclusive;
        self.entries
            .range(RangeInclusive::new(start, end))
            .map(|(&timestamp, &value)| TimeSeriesPoint { timestamp, value })
            .collect()
    }

    pub fn scan_all(&self) -> Vec<TimeSeriesPoint> {
        self.entries
            .iter()
            .map(|(&timestamp, &value)| TimeSeriesPoint { timestamp, value })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_drain_sorted() {
        let mut mt = MemTable::new(1024 * 1024);
        mt.insert(TimeSeriesPoint { timestamp: 300, value: 3.0 });
        mt.insert(TimeSeriesPoint { timestamp: 100, value: 1.0 });
        mt.insert(TimeSeriesPoint { timestamp: 200, value: 2.0 });

        let points = mt.drain_sorted();
        assert_eq!(points.len(), 3);
        assert_eq!(points[0].timestamp, 100);
        assert_eq!(points[1].timestamp, 200);
        assert_eq!(points[2].timestamp, 300);
        assert!(mt.is_empty());
    }

    #[test]
    fn is_full_triggers_at_capacity() {
        let mut mt = MemTable::new(256);
        assert!(!mt.is_full());
        for i in 0..10 {
            mt.insert(TimeSeriesPoint { timestamp: i, value: i as f64 });
        }
        assert!(mt.is_full());
    }

    #[test]
    fn duplicate_timestamps_overwrite() {
        let mut mt = MemTable::new(1024 * 1024);
        mt.insert(TimeSeriesPoint { timestamp: 100, value: 1.0 });
        mt.insert(TimeSeriesPoint { timestamp: 100, value: 2.0 });
        assert_eq!(mt.len(), 1);
        let points = mt.drain_sorted();
        assert_eq!(points[0].value, 2.0);
    }
}
