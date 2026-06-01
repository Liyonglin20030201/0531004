use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use crate::types::TimeSeriesPoint;

use super::memtable::MemTable;
use super::sstable::{SSTableReader, SSTableWriter};
use super::wal::Wal;

pub const COMPACTION_THRESHOLD: usize = 4;

pub struct TsdbEngine {
    memtable: MemTable,
    wal: Wal,
    data_dir: PathBuf,
    sstable_seq: u64,
}

impl TsdbEngine {
    pub fn open(data_dir: &Path, memtable_capacity: usize) -> io::Result<Self> {
        fs::create_dir_all(data_dir)?;
        let wal = Wal::open(data_dir)?;
        let sstable_seq = Self::find_max_seq(data_dir);

        Ok(Self {
            memtable: MemTable::new(memtable_capacity),
            wal,
            data_dir: data_dir.to_path_buf(),
            sstable_seq,
        })
    }

    pub fn write(&mut self, point: TimeSeriesPoint) -> io::Result<()> {
        self.wal.append(point)?;
        self.memtable.insert(point);
        if self.memtable.is_full() {
            self.flush()?;
        }
        Ok(())
    }

    pub fn write_batch(&mut self, points: &[TimeSeriesPoint]) -> io::Result<()> {
        for &point in points {
            self.wal.append(point)?;
            self.memtable.insert(point);
        }
        if self.memtable.is_full() {
            self.flush()?;
        }
        Ok(())
    }

    pub fn flush(&mut self) -> io::Result<()> {
        if self.memtable.is_empty() {
            return Ok(());
        }

        let points = self.memtable.drain_sorted();
        self.sstable_seq += 1;
        let sst_path = self.data_dir.join(format!("{:06}.sst", self.sstable_seq));

        let mut writer = SSTableWriter::create(&sst_path)?;
        writer.write_all(&points)?;
        writer.finish()?;

        // WAL can be discarded after successful flush
        let old_wal = std::mem::replace(&mut self.wal, Wal::open(&self.data_dir)?);
        old_wal.discard()?;

        // Auto-compact if too many SSTables
        if self.sst_count() >= COMPACTION_THRESHOLD {
            self.compact()?;
        }

        Ok(())
    }

    pub fn recover(data_dir: &Path, memtable_capacity: usize) -> io::Result<Self> {
        fs::create_dir_all(data_dir)?;
        let points = Wal::replay(data_dir)?;
        let sstable_seq = Self::find_max_seq(data_dir);

        let mut memtable = MemTable::new(memtable_capacity);
        for point in points {
            memtable.insert(point);
        }

        let wal = Wal::open(data_dir)?;

        Ok(Self {
            memtable,
            wal,
            data_dir: data_dir.to_path_buf(),
            sstable_seq,
        })
    }

    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    fn find_max_seq(data_dir: &Path) -> u64 {
        let mut max_seq = 0u64;
        if let Ok(entries) = fs::read_dir(data_dir) {
            for entry in entries.flatten() {
                if let Some(name) = entry.file_name().to_str() {
                    if let Some(stem) = name.strip_suffix(".sst") {
                        if let Ok(seq) = stem.parse::<u64>() {
                            max_seq = max_seq.max(seq);
                        }
                    }
                }
            }
        }
        max_seq
    }

    fn list_sst_files(data_dir: &Path) -> Vec<PathBuf> {
        let mut files: Vec<PathBuf> = fs::read_dir(data_dir)
            .into_iter()
            .flatten()
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|ext| ext == "sst").unwrap_or(false))
            .collect();
        files.sort();
        files
    }

    /// Query a time range, merging results from MemTable and all SSTables.
    /// Results are returned sorted by timestamp (ascending).
    pub fn scan_range(&self, start: i64, end: i64) -> io::Result<Vec<TimeSeriesPoint>> {
        let mem_points = self.memtable.scan_range(start, end);
        let sst_files = Self::list_sst_files(&self.data_dir);

        let mut sst_points = Vec::new();
        for path in &sst_files {
            let reader = SSTableReader::open(path)?;
            sst_points.extend(reader.scan_range(start, end)?);
        }

        Ok(Self::merge_sorted(mem_points, sst_points))
    }

    /// Read all data points from MemTable and all SSTables.
    pub fn scan_all(&self) -> io::Result<Vec<TimeSeriesPoint>> {
        let mem_points = self.memtable.scan_all();
        let sst_files = Self::list_sst_files(&self.data_dir);

        let mut sst_points = Vec::new();
        for path in &sst_files {
            let reader = SSTableReader::open(path)?;
            sst_points.extend(reader.read_all()?);
        }

        Ok(Self::merge_sorted(mem_points, sst_points))
    }

    /// Merge two sorted streams. MemTable entries win on timestamp collision
    /// (they are newer).
    fn merge_sorted(mem: Vec<TimeSeriesPoint>, sst: Vec<TimeSeriesPoint>) -> Vec<TimeSeriesPoint> {
        let mut result = Vec::with_capacity(mem.len() + sst.len());
        let mut mi = 0;
        let mut si = 0;

        while mi < mem.len() && si < sst.len() {
            if mem[mi].timestamp < sst[si].timestamp {
                result.push(mem[mi]);
                mi += 1;
            } else if mem[mi].timestamp > sst[si].timestamp {
                result.push(sst[si]);
                si += 1;
            } else {
                // MemTable wins (newer data)
                result.push(mem[mi]);
                mi += 1;
                si += 1;
            }
        }
        result.extend_from_slice(&mem[mi..]);
        result.extend_from_slice(&sst[si..]);
        result
    }

    /// Trigger compaction: merge all SSTable files into one.
    /// Called automatically when SSTable count exceeds threshold.
    pub fn compact(&mut self) -> io::Result<()> {
        let sst_files = Self::list_sst_files(&self.data_dir);
        if sst_files.len() < 2 {
            return Ok(());
        }

        // Read all points from all SSTables
        let mut all_points = Vec::new();
        for path in &sst_files {
            let reader = SSTableReader::open(path)?;
            all_points.extend(reader.read_all()?);
        }

        // Sort and deduplicate (keep latest for duplicate timestamps)
        all_points.sort_by_key(|p| p.timestamp);
        all_points.dedup_by_key(|p| p.timestamp);

        // Write merged SSTable
        self.sstable_seq += 1;
        let merged_path = self.data_dir.join(format!("{:06}.sst", self.sstable_seq));
        let mut writer = SSTableWriter::create(&merged_path)?;
        writer.write_all(&all_points)?;
        writer.finish()?;

        // Remove old SSTable files
        for path in &sst_files {
            fs::remove_file(path)?;
        }

        Ok(())
    }

    /// Number of SSTable files on disk.
    pub fn sst_count(&self) -> usize {
        Self::list_sst_files(&self.data_dir).len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_and_flush() {
        let dir = TempDir::new().unwrap();
        let mut engine = TsdbEngine::open(dir.path(), 512).unwrap();

        for i in 0..100 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 1_000_000,
                    value: 20.0 + i as f64 * 0.1,
                })
                .unwrap();
        }

        let sst_files: Vec<_> = fs::read_dir(dir.path())
            .unwrap()
            .filter_map(|e| e.ok())
            .filter(|e| e.path().extension().map(|ext| ext == "sst").unwrap_or(false))
            .collect();
        assert!(!sst_files.is_empty());
    }

    #[test]
    fn crash_recovery() {
        let dir = TempDir::new().unwrap();
        {
            let mut engine = TsdbEngine::open(dir.path(), 1024 * 1024).unwrap();
            for i in 0..50 {
                engine
                    .write(TimeSeriesPoint {
                        timestamp: i * 1000,
                        value: i as f64,
                    })
                    .unwrap();
            }
            engine.wal.sync().unwrap();
        }

        let engine = TsdbEngine::recover(dir.path(), 1024 * 1024).unwrap();
        assert_eq!(engine.memtable_len(), 50);
    }

    #[test]
    fn scan_range_merges_memtable_and_sstable() {
        let dir = TempDir::new().unwrap();
        let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

        // Write first batch and flush to SSTable
        for i in 0..50 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 1000,
                    value: i as f64,
                })
                .unwrap();
        }
        engine.flush().unwrap();

        // Write second batch (stays in MemTable)
        for i in 50..100 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 1000,
                    value: i as f64,
                })
                .unwrap();
        }

        // Query across both SSTable and MemTable
        let results = engine.scan_range(25_000, 75_000).unwrap();
        assert!(!results.is_empty());
        for p in &results {
            assert!(p.timestamp >= 25_000 && p.timestamp <= 75_000);
        }
        // Should include points from both SSTable (25..50) and MemTable (50..75)
        assert_eq!(results.first().unwrap().timestamp, 25_000);
        assert_eq!(results.last().unwrap().timestamp, 75_000);
    }

    #[test]
    fn scan_all_returns_complete_dataset() {
        let dir = TempDir::new().unwrap();
        let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

        for i in 0..30 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 100,
                    value: i as f64,
                })
                .unwrap();
        }
        engine.flush().unwrap();

        for i in 30..60 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 100,
                    value: i as f64,
                })
                .unwrap();
        }

        let all = engine.scan_all().unwrap();
        assert_eq!(all.len(), 60);
        // Verify sorted
        for w in all.windows(2) {
            assert!(w[0].timestamp < w[1].timestamp);
        }
    }

    #[test]
    fn memtable_wins_on_timestamp_collision() {
        let dir = TempDir::new().unwrap();
        let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

        // Write old value and flush to SSTable
        engine
            .write(TimeSeriesPoint { timestamp: 1000, value: 1.0 })
            .unwrap();
        engine.flush().unwrap();

        // Write newer value to same timestamp (stays in MemTable)
        engine
            .write(TimeSeriesPoint { timestamp: 1000, value: 99.0 })
            .unwrap();

        let results = engine.scan_range(0, 2000).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].value, 99.0);
    }

    #[test]
    fn compaction_merges_sstables() {
        let dir = TempDir::new().unwrap();
        // Very small capacity: each write triggers flush
        let mut engine = TsdbEngine::open(dir.path(), 128).unwrap();

        // Write enough to create multiple SSTables (before auto-compact threshold)
        for i in 0..20 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 1000,
                    value: i as f64,
                })
                .unwrap();
        }

        // After auto-compaction, should have at most 1 SSTable
        // (compaction triggers when >= COMPACTION_THRESHOLD)
        let sst_count = engine.sst_count();
        assert!(
            sst_count < COMPACTION_THRESHOLD,
            "after compaction sst count should be < threshold, got {}",
            sst_count
        );

        // Data should still be fully readable
        let all = engine.scan_all().unwrap();
        assert_eq!(all.len(), 20);
    }

    #[test]
    fn manual_compact_deduplicates() {
        let dir = TempDir::new().unwrap();
        let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

        // First flush
        for i in 0..10 {
            engine
                .write(TimeSeriesPoint { timestamp: i * 100, value: i as f64 })
                .unwrap();
        }
        engine.flush().unwrap();

        // Second flush with overlapping timestamps (updated values)
        for i in 5..15 {
            engine
                .write(TimeSeriesPoint { timestamp: i * 100, value: (i * 10) as f64 })
                .unwrap();
        }
        engine.flush().unwrap();

        assert_eq!(engine.sst_count(), 2);
        engine.compact().unwrap();
        assert_eq!(engine.sst_count(), 1);

        // Verify deduplication: 15 unique timestamps (0..15)
        let all = engine.scan_all().unwrap();
        assert_eq!(all.len(), 15);
        // Value at timestamp 500 should be from second write (50.0)
        let p500 = all.iter().find(|p| p.timestamp == 500).unwrap();
        assert_eq!(p500.value, 50.0);
    }
}
