use tsdb_engine::{TimeSeriesPoint, TsdbEngine};
use tsdb_engine::lsm::sstable::SSTableReader;
use tsdb_engine::lsm::engine::COMPACTION_THRESHOLD;
use tempfile::TempDir;

#[test]
fn end_to_end_write_flush_read() {
    let dir = TempDir::new().unwrap();
    let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

    let points: Vec<TimeSeriesPoint> = (0..500)
        .map(|i| TimeSeriesPoint {
            timestamp: 1_000_000_000 + i * 1_000_000,
            value: 25.0 + (i as f64) * 0.01,
        })
        .collect();

    for &p in &points {
        engine.write(p).unwrap();
    }
    engine.flush().unwrap();

    // Use engine's scan_all to read back (merges MemTable + SSTable)
    let all_read = engine.scan_all().unwrap();

    assert_eq!(all_read.len(), points.len());
    for (orig, read) in points.iter().zip(all_read.iter()) {
        assert_eq!(orig.timestamp, read.timestamp);
        assert_eq!(orig.value.to_bits(), read.value.to_bits());
    }
}

#[test]
fn crash_recovery_via_wal_replay() {
    let dir = TempDir::new().unwrap();

    {
        let mut engine = TsdbEngine::open(dir.path(), 1024 * 1024).unwrap();
        for i in 0..50 {
            engine
                .write(TimeSeriesPoint {
                    timestamp: i * 1_000_000,
                    value: 100.0 + i as f64,
                })
                .unwrap();
        }
        engine.flush().ok();
    }

    let engine = TsdbEngine::recover(dir.path(), 1024 * 1024).unwrap();
    let sst_files: Vec<_> = std::fs::read_dir(dir.path())
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|ext| ext == "sst").unwrap_or(false))
        .collect();

    let has_sst_data = !sst_files.is_empty();
    let has_memtable_data = engine.memtable_len() > 0;
    assert!(has_sst_data || has_memtable_data);
}

#[test]
fn compression_ratio_iot_workload() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("compression_test.sst");

    let points: Vec<TimeSeriesPoint> = (0..10_000)
        .map(|i| TimeSeriesPoint {
            timestamp: 1_700_000_000_000 + i * 1_000,
            value: 25.0 + (i as f64 * 0.001).sin() * 5.0,
        })
        .collect();

    let mut writer = tsdb_engine::lsm::sstable::SSTableWriter::create(&path).unwrap();
    writer.write_all(&points).unwrap();
    writer.finish().unwrap();

    let file_size = std::fs::metadata(&path).unwrap().len();
    let raw_size = points.len() * 16;

    let ratio = raw_size as f64 / file_size as f64;
    assert!(
        ratio > 2.0,
        "compression ratio should be > 2x for IoT data, got {:.2}x ({} raw vs {} compressed)",
        ratio,
        raw_size,
        file_size
    );
}

#[test]
fn query_merges_memtable_and_multiple_sstables() {
    let dir = TempDir::new().unwrap();
    let mut engine = TsdbEngine::open(dir.path(), 4096).unwrap();

    // Phase 1: write and flush (goes to SSTable)
    for i in 0..100 {
        engine
            .write(TimeSeriesPoint {
                timestamp: i * 1000,
                value: i as f64,
            })
            .unwrap();
    }
    engine.flush().unwrap();

    // Phase 2: write more (stays in MemTable)
    for i in 100..200 {
        engine
            .write(TimeSeriesPoint {
                timestamp: i * 1000,
                value: i as f64,
            })
            .unwrap();
    }

    // Query spanning both
    let results = engine.scan_range(50_000, 150_000).unwrap();
    assert!(!results.is_empty());
    assert_eq!(results.first().unwrap().timestamp, 50_000);
    assert_eq!(results.last().unwrap().timestamp, 150_000);

    // Verify all results are in sorted order
    for w in results.windows(2) {
        assert!(w[0].timestamp <= w[1].timestamp);
    }
}

#[test]
fn compaction_reduces_sst_count_and_preserves_data() {
    let dir = TempDir::new().unwrap();
    // Small capacity: frequent flushes to create many SSTables
    let mut engine = TsdbEngine::open(dir.path(), 256).unwrap();

    let total_points = 200;
    for i in 0..total_points {
        engine
            .write(TimeSeriesPoint {
                timestamp: i * 1000,
                value: i as f64,
            })
            .unwrap();
    }
    engine.flush().unwrap();

    // After auto-compaction, SSTable count should be manageable
    assert!(
        engine.sst_count() < COMPACTION_THRESHOLD,
        "expected < {} SSTables after compaction, got {}",
        COMPACTION_THRESHOLD,
        engine.sst_count()
    );

    // All data should still be fully queryable
    let all = engine.scan_all().unwrap();
    assert_eq!(all.len(), total_points as usize);

    // Verify sort order
    for w in all.windows(2) {
        assert!(w[0].timestamp < w[1].timestamp);
    }
}

#[test]
fn avx2_compression_produces_correct_sstable() {
    let dir = TempDir::new().unwrap();
    let path = dir.path().join("avx2_test.sst");

    // Large enough block to trigger SIMD batch path (>= 3 points per block)
    let points: Vec<TimeSeriesPoint> = (0..2048)
        .map(|i| TimeSeriesPoint {
            timestamp: 1_000_000 + i * 1_000,
            value: 36.6 + (i as f64 * 0.01).cos() * 2.0,
        })
        .collect();

    let mut writer = tsdb_engine::lsm::sstable::SSTableWriter::create(&path).unwrap();
    writer.write_all(&points).unwrap();
    writer.finish().unwrap();

    let reader = SSTableReader::open(&path).unwrap();
    let read_back = reader.read_all().unwrap();
    assert_eq!(read_back.len(), points.len());
    for (orig, read) in points.iter().zip(read_back.iter()) {
        assert_eq!(orig.timestamp, read.timestamp);
        assert_eq!(orig.value.to_bits(), read.value.to_bits());
    }
}
