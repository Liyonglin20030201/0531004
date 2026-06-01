use super::delta::{BitBuffer, BitReader};

pub struct XorFloatEncoder {
    prev_bits: u64,
    prev_leading: u8,
    prev_trailing: u8,
    buf: BitBuffer,
    first: bool,
}

impl XorFloatEncoder {
    pub fn new() -> Self {
        Self {
            prev_bits: 0,
            prev_leading: 0,
            prev_trailing: 0,
            buf: BitBuffer::new(),
            first: true,
        }
    }

    pub fn encode(&mut self, value: f64) {
        let bits = value.to_bits();
        if self.first {
            self.buf.write_bits(bits, 64);
            self.prev_bits = bits;
            self.first = false;
            return;
        }

        let xor = bits ^ self.prev_bits;
        if xor == 0 {
            self.buf.write_bit(false);
        } else {
            self.buf.write_bit(true);
            let leading = xor.leading_zeros() as u8;
            let trailing = xor.trailing_zeros() as u8;

            if leading >= self.prev_leading && trailing >= self.prev_trailing {
                self.buf.write_bit(false);
                let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
                let value_shifted = xor >> self.prev_trailing;
                self.buf.write_bits(value_shifted, meaningful_bits);
            } else {
                self.buf.write_bit(true);
                self.buf.write_bits(leading as u64, 5);
                let meaningful_bits = 64 - leading - trailing;
                self.buf.write_bits(meaningful_bits as u64, 6);
                let value_shifted = xor >> trailing;
                self.buf.write_bits(value_shifted, meaningful_bits);
                self.prev_leading = leading;
                self.prev_trailing = trailing;
            }
        }
        self.prev_bits = bits;
    }

    pub fn encode_batch(&mut self, values: &[f64]) {
        if values.is_empty() {
            return;
        }
        if self.first && values.len() >= 2 {
            // First value stored raw
            let first_bits = values[0].to_bits();
            self.buf.write_bits(first_bits, 64);
            self.prev_bits = first_bits;
            self.first = false;

            // Use SIMD to pre-compute all XOR bit patterns
            let xors = crate::compression::simd::batch_xor_values(values);
            for &xor in &xors {
                self.encode_xor(xor);
            }
            self.prev_bits = values[values.len() - 1].to_bits();
        } else {
            for &v in values {
                self.encode(v);
            }
        }
    }

    fn encode_xor(&mut self, xor: u64) {
        if xor == 0 {
            self.buf.write_bit(false);
        } else {
            self.buf.write_bit(true);
            let leading = xor.leading_zeros() as u8;
            let trailing = xor.trailing_zeros() as u8;

            if leading >= self.prev_leading && trailing >= self.prev_trailing {
                self.buf.write_bit(false);
                let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
                let value_shifted = xor >> self.prev_trailing;
                self.buf.write_bits(value_shifted, meaningful_bits);
            } else {
                self.buf.write_bit(true);
                self.buf.write_bits(leading as u64, 5);
                let meaningful_bits = 64 - leading - trailing;
                self.buf.write_bits(meaningful_bits as u64, 6);
                let value_shifted = xor >> trailing;
                self.buf.write_bits(value_shifted, meaningful_bits);
                self.prev_leading = leading;
                self.prev_trailing = trailing;
            }
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf.finish()
    }
}

pub struct XorFloatDecoder<'a> {
    reader: BitReader<'a>,
    prev_bits: u64,
    prev_leading: u8,
    prev_trailing: u8,
    first: bool,
}

impl<'a> XorFloatDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            prev_bits: 0,
            prev_leading: 0,
            prev_trailing: 0,
            first: true,
        }
    }

    pub fn decode_all(&mut self, count: usize) -> Vec<f64> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.decode_one());
        }
        result
    }

    fn decode_one(&mut self) -> f64 {
        if self.first {
            self.prev_bits = self.reader.read_bits(64);
            self.first = false;
            return f64::from_bits(self.prev_bits);
        }

        if !self.reader.read_bit() {
            return f64::from_bits(self.prev_bits);
        }

        let (leading, meaningful_bits, trailing) = if !self.reader.read_bit() {
            let meaningful_bits = 64 - self.prev_leading - self.prev_trailing;
            (self.prev_leading, meaningful_bits, self.prev_trailing)
        } else {
            let leading = self.reader.read_bits(5) as u8;
            let meaningful_bits = self.reader.read_bits(6) as u8;
            let trailing = 64 - leading - meaningful_bits;
            self.prev_leading = leading;
            self.prev_trailing = trailing;
            (leading, meaningful_bits, trailing)
        };

        let _ = leading;
        let value_shifted = self.reader.read_bits(meaningful_bits);
        let xor = value_shifted << trailing;
        self.prev_bits ^= xor;
        f64::from_bits(self.prev_bits)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_constant_values() {
        let values: Vec<f64> = vec![25.0; 100];
        let mut encoder = XorFloatEncoder::new();
        for &v in &values {
            encoder.encode(v);
        }
        let data = encoder.finish();
        let mut decoder = XorFloatDecoder::new(&data);
        let decoded = decoder.decode_all(values.len());
        assert_eq!(values, decoded);
    }

    #[test]
    fn round_trip_slowly_changing() {
        let values: Vec<f64> = (0..100).map(|i| 25.0 + i as f64 * 0.01).collect();
        let mut encoder = XorFloatEncoder::new();
        for &v in &values {
            encoder.encode(v);
        }
        let data = encoder.finish();
        let mut decoder = XorFloatDecoder::new(&data);
        let decoded = decoder.decode_all(values.len());
        assert_eq!(values, decoded);
    }

    #[test]
    fn round_trip_special_values() {
        let values = vec![0.0, -0.0, f64::INFINITY, f64::NEG_INFINITY, 1.0, -1.0];
        let mut encoder = XorFloatEncoder::new();
        for &v in &values {
            encoder.encode(v);
        }
        let data = encoder.finish();
        let mut decoder = XorFloatDecoder::new(&data);
        let decoded = decoder.decode_all(values.len());
        for (original, decoded_val) in values.iter().zip(decoded.iter()) {
            assert_eq!(original.to_bits(), decoded_val.to_bits());
        }
    }

    #[test]
    fn compression_ratio_constant() {
        let values: Vec<f64> = vec![42.0; 1000];
        let mut encoder = XorFloatEncoder::new();
        for &v in &values {
            encoder.encode(v);
        }
        let data = encoder.finish();
        let raw_size = values.len() * 8;
        assert!(
            data.len() < raw_size / 10,
            "constant values should compress extremely well: {} vs {}",
            data.len(),
            raw_size
        );
    }

    #[test]
    fn encode_batch_round_trip() {
        let values: Vec<f64> = (0..1000).map(|i| 25.0 + (i as f64 * 0.001).sin() * 5.0).collect();
        let mut encoder = XorFloatEncoder::new();
        encoder.encode_batch(&values);
        let data = encoder.finish();
        let mut decoder = XorFloatDecoder::new(&data);
        let decoded = decoder.decode_all(values.len());
        for (orig, dec) in values.iter().zip(decoded.iter()) {
            assert_eq!(orig.to_bits(), dec.to_bits());
        }
    }

    #[test]
    fn encode_batch_matches_sequential() {
        let values: Vec<f64> = (0..200).map(|i| 100.0 + i as f64 * 0.5).collect();

        let mut enc_seq = XorFloatEncoder::new();
        for &v in &values {
            enc_seq.encode(v);
        }
        let data_seq = enc_seq.finish();

        let mut enc_batch = XorFloatEncoder::new();
        enc_batch.encode_batch(&values);
        let data_batch = enc_batch.finish();

        assert_eq!(data_seq, data_batch);
    }
}
