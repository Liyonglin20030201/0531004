#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// Compute deltas between consecutive i64 timestamps using AVX2.
/// Processes 4 elements at a time (256-bit registers hold 4x i64).
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_delta_i64_inner(input: &[i64], output: &mut Vec<i64>) {
    let len = input.len();
    if len < 2 {
        return;
    }

    let chunks = (len - 1) / 4;
    for chunk in 0..chunks {
        let i = chunk * 4;
        let curr = _mm256_loadu_si256(input[i + 1..].as_ptr() as *const __m256i);
        let prev = _mm256_loadu_si256(input[i..].as_ptr() as *const __m256i);
        let delta = _mm256_sub_epi64(curr, prev);
        let mut buf = [0i64; 4];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, delta);
        output.extend_from_slice(&buf);
    }

    let processed = chunks * 4;
    for i in processed..(len - 1) {
        output.push(input[i + 1] - input[i]);
    }
}

/// Compute XOR between consecutive f64 values (as u64 bit patterns) using AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn avx2_xor_f64_inner(input: &[f64], output: &mut Vec<u64>) {
    let len = input.len();
    if len < 2 {
        return;
    }

    let ptr = input.as_ptr() as *const i64;
    let chunks = (len - 1) / 4;
    for chunk in 0..chunks {
        let i = chunk * 4;
        let curr = _mm256_loadu_si256(ptr.add(i + 1) as *const __m256i);
        let prev = _mm256_loadu_si256(ptr.add(i) as *const __m256i);
        let xor = _mm256_xor_si256(curr, prev);
        let mut buf = [0u64; 4];
        _mm256_storeu_si256(buf.as_mut_ptr() as *mut __m256i, xor);
        output.extend_from_slice(&buf);
    }

    let processed = chunks * 4;
    for i in processed..(len - 1) {
        let curr_bits = input[i + 1].to_bits();
        let prev_bits = input[i].to_bits();
        output.push(curr_bits ^ prev_bits);
    }
}

fn scalar_delta_i64(input: &[i64]) -> Vec<i64> {
    if input.len() < 2 {
        return Vec::new();
    }
    input.windows(2).map(|w| w[1] - w[0]).collect()
}

fn scalar_xor_f64(input: &[f64]) -> Vec<u64> {
    if input.len() < 2 {
        return Vec::new();
    }
    input
        .windows(2)
        .map(|w| w[1].to_bits() ^ w[0].to_bits())
        .collect()
}

fn scalar_batch_dod(timestamps: &[i64]) -> Vec<i64> {
    if timestamps.len() < 3 {
        return Vec::new();
    }
    let deltas = scalar_delta_i64(timestamps);
    scalar_delta_i64(&deltas)
}

/// Compute all delta-of-deltas for a timestamp slice.
/// Uses AVX2 when available, falls back to scalar.
pub fn batch_delta_of_delta(timestamps: &[i64]) -> Vec<i64> {
    if timestamps.len() < 3 {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut deltas = Vec::with_capacity(timestamps.len() - 1);
            unsafe { avx2_delta_i64_inner(timestamps, &mut deltas) };
            let mut dods = Vec::with_capacity(deltas.len() - 1);
            unsafe { avx2_delta_i64_inner(&deltas, &mut dods) };
            return dods;
        }
    }

    scalar_batch_dod(timestamps)
}

/// Compute XOR of consecutive f64 values, SIMD-accelerated where possible.
pub fn batch_xor_values(values: &[f64]) -> Vec<u64> {
    if values.len() < 2 {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut result = Vec::with_capacity(values.len() - 1);
            unsafe { avx2_xor_f64_inner(values, &mut result) };
            return result;
        }
    }

    scalar_xor_f64(values)
}

/// Compute first-order deltas. SIMD-accelerated where possible.
pub fn batch_delta(timestamps: &[i64]) -> Vec<i64> {
    if timestamps.len() < 2 {
        return Vec::new();
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            let mut result = Vec::with_capacity(timestamps.len() - 1);
            unsafe { avx2_delta_i64_inner(timestamps, &mut result) };
            return result;
        }
    }

    scalar_delta_i64(timestamps)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn delta_matches_scalar() {
        let timestamps: Vec<i64> = (0..1000).map(|i| i * 1_000_000 + i * i).collect();
        let scalar = scalar_delta_i64(&timestamps);
        let result = batch_delta(&timestamps);
        assert_eq!(scalar, result);
    }

    #[test]
    fn dod_matches_scalar() {
        let timestamps: Vec<i64> = (0..1000).map(|i| i * 1_000_000 + i * i * 3).collect();
        let scalar = scalar_batch_dod(&timestamps);
        let result = batch_delta_of_delta(&timestamps);
        assert_eq!(scalar, result);
    }

    #[test]
    fn xor_matches_scalar() {
        let values: Vec<f64> = (0..1000).map(|i| 25.0 + i as f64 * 0.01).collect();
        let scalar = scalar_xor_f64(&values);
        let result = batch_xor_values(&values);
        assert_eq!(scalar, result);
    }

    #[test]
    fn handles_small_inputs() {
        assert_eq!(batch_delta_of_delta(&[1, 2]), Vec::<i64>::new());
        assert_eq!(batch_delta_of_delta(&[]), Vec::<i64>::new());
        assert_eq!(batch_xor_values(&[1.0]), Vec::<u64>::new());
        assert_eq!(batch_delta(&[1]), Vec::<i64>::new());
    }

    #[test]
    fn constant_interval_dod_is_zero() {
        let timestamps: Vec<i64> = (0..100).map(|i| i * 1000).collect();
        let dods = batch_delta_of_delta(&timestamps);
        assert!(dods.iter().all(|&d| d == 0));
    }
}
