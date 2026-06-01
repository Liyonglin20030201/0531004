use criterion::{criterion_group, criterion_main, Criterion, Throughput};
use tsdb_engine::compression::delta::{DeltaOfDeltaDecoder, DeltaOfDeltaEncoder};
use tsdb_engine::compression::simd::{batch_delta_of_delta, batch_xor_values};
use tsdb_engine::compression::xor_float::{XorFloatDecoder, XorFloatEncoder};
use tsdb_engine::{TimeSeriesPoint, TsdbEngine};

fn bench_write_throughput(c: &mut Criterion) {
    let mut group = c.benchmark_group("write");
    group.throughput(Throughput::Elements(100_000));
    group.bench_function("sequential_timestamps", |b| {
        b.iter(|| {
            let dir = tempfile::tempdir().unwrap();
            let mut engine = TsdbEngine::open(dir.path(), 4 * 1024 * 1024).unwrap();
            for i in 0..100_000u64 {
                engine
                    .write(TimeSeriesPoint {
                        timestamp: i as i64 * 1_000_000_000,
                        value: 25.0 + (i as f64 * 0.01),
                    })
                    .unwrap();
            }
        });
    });
    group.finish();
}

fn bench_delta_of_delta(c: &mut Criterion) {
    let timestamps: Vec<i64> = (0..10_000).map(|i| i * 1_000_000_000).collect();

    let mut group = c.benchmark_group("compression");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("dod_encode", |b| {
        b.iter(|| {
            let mut encoder = DeltaOfDeltaEncoder::new();
            for &ts in &timestamps {
                encoder.encode(ts);
            }
            encoder.finish()
        });
    });

    let mut encoder = DeltaOfDeltaEncoder::new();
    for &ts in &timestamps {
        encoder.encode(ts);
    }
    let encoded = encoder.finish();

    group.bench_function("dod_decode", |b| {
        b.iter(|| {
            let mut decoder = DeltaOfDeltaDecoder::new(&encoded);
            decoder.decode_all(10_000)
        });
    });

    group.bench_function("simd_batch_dod", |b| {
        b.iter(|| batch_delta_of_delta(&timestamps));
    });

    group.bench_function("dod_encode_batch", |b| {
        b.iter(|| {
            let mut encoder = DeltaOfDeltaEncoder::new();
            encoder.encode_batch(&timestamps);
            encoder.finish()
        });
    });

    group.finish();
}

fn bench_xor_float(c: &mut Criterion) {
    let values: Vec<f64> = (0..10_000)
        .map(|i| 25.0 + (i as f64 * 0.001).sin() * 5.0)
        .collect();

    let mut group = c.benchmark_group("xor_float");
    group.throughput(Throughput::Elements(10_000));

    group.bench_function("encode", |b| {
        b.iter(|| {
            let mut encoder = XorFloatEncoder::new();
            for &v in &values {
                encoder.encode(v);
            }
            encoder.finish()
        });
    });

    let mut encoder = XorFloatEncoder::new();
    for &v in &values {
        encoder.encode(v);
    }
    let encoded = encoder.finish();

    group.bench_function("decode", |b| {
        b.iter(|| {
            let mut decoder = XorFloatDecoder::new(&encoded);
            decoder.decode_all(10_000)
        });
    });

    group.bench_function("simd_batch_xor", |b| {
        b.iter(|| batch_xor_values(&values));
    });

    group.bench_function("xor_encode_batch", |b| {
        b.iter(|| {
            let mut encoder = XorFloatEncoder::new();
            encoder.encode_batch(&values);
            encoder.finish()
        });
    });

    group.finish();
}

criterion_group!(benches, bench_write_throughput, bench_delta_of_delta, bench_xor_float);
criterion_main!(benches);
