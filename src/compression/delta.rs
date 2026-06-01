pub(crate) struct BitBuffer {
    bytes: Vec<u8>,
    current_byte: u8,
    bit_pos: u8,
}

impl BitBuffer {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current_byte: 0,
            bit_pos: 0,
        }
    }

    pub(crate) fn write_bit(&mut self, bit: bool) {
        if bit {
            self.current_byte |= 1 << (7 - self.bit_pos);
        }
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.bytes.push(self.current_byte);
            self.current_byte = 0;
            self.bit_pos = 0;
        }
    }

    pub(crate) fn write_bits(&mut self, value: u64, num_bits: u8) {
        for i in (0..num_bits).rev() {
            self.write_bit((value >> i) & 1 == 1);
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.bit_pos > 0 {
            self.bytes.push(self.current_byte);
        }
        self.bytes
    }
}

pub(crate) struct BitReader<'a> {
    data: &'a [u8],
    byte_pos: usize,
    bit_pos: u8,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(data: &'a [u8]) -> Self {
        Self {
            data,
            byte_pos: 0,
            bit_pos: 0,
        }
    }

    pub(crate) fn read_bit(&mut self) -> bool {
        let bit = (self.data[self.byte_pos] >> (7 - self.bit_pos)) & 1 == 1;
        self.bit_pos += 1;
        if self.bit_pos == 8 {
            self.byte_pos += 1;
            self.bit_pos = 0;
        }
        bit
    }

    pub(crate) fn read_bits(&mut self, num_bits: u8) -> u64 {
        let mut value: u64 = 0;
        for _ in 0..num_bits {
            value = (value << 1) | (self.read_bit() as u64);
        }
        value
    }

    pub(crate) fn read_bits_signed(&mut self, num_bits: u8) -> i64 {
        let raw = self.read_bits(num_bits);
        let sign_bit = 1u64 << (num_bits - 1);
        if raw & sign_bit != 0 {
            (raw | !((1u64 << num_bits) - 1)) as i64
        } else {
            raw as i64
        }
    }
}

pub struct DeltaOfDeltaEncoder {
    prev_timestamp: i64,
    prev_delta: i64,
    buf: BitBuffer,
    count: usize,
}

impl DeltaOfDeltaEncoder {
    pub fn new() -> Self {
        Self {
            prev_timestamp: 0,
            prev_delta: 0,
            buf: BitBuffer::new(),
            count: 0,
        }
    }

    pub fn encode(&mut self, timestamp: i64) {
        if self.count == 0 {
            self.buf.write_bits(timestamp as u64, 64);
            self.prev_timestamp = timestamp;
        } else if self.count == 1 {
            let delta = timestamp - self.prev_timestamp;
            self.buf.write_bits(delta as u64, 64);
            self.prev_delta = delta;
            self.prev_timestamp = timestamp;
        } else {
            let delta = timestamp - self.prev_timestamp;
            let dod = delta - self.prev_delta;
            self.encode_dod(dod);
            self.prev_delta = delta;
            self.prev_timestamp = timestamp;
        }
        self.count += 1;
    }

    fn encode_dod(&mut self, dod: i64) {
        if dod == 0 {
            self.buf.write_bit(false);
        } else if dod >= -63 && dod <= 64 {
            self.buf.write_bits(0b10, 2);
            self.buf.write_bits((dod as u64) & 0x7F, 7);
        } else if dod >= -255 && dod <= 256 {
            self.buf.write_bits(0b110, 3);
            self.buf.write_bits((dod as u64) & 0x1FF, 9);
        } else if dod >= -2047 && dod <= 2048 {
            self.buf.write_bits(0b1110, 4);
            self.buf.write_bits((dod as u64) & 0xFFF, 12);
        } else {
            self.buf.write_bits(0b1111, 4);
            self.buf.write_bits(dod as u64, 64);
        }
    }

    pub fn encode_batch(&mut self, timestamps: &[i64]) {
        if timestamps.is_empty() {
            return;
        }
        if self.count == 0 && timestamps.len() >= 3 {
            self.buf.write_bits(timestamps[0] as u64, 64);
            let delta0 = timestamps[1] - timestamps[0];
            self.buf.write_bits(delta0 as u64, 64);
            self.prev_timestamp = timestamps[1];
            self.prev_delta = delta0;
            self.count = 2;

            let dods = crate::compression::simd::batch_delta_of_delta(timestamps);
            for &dod in &dods {
                self.encode_dod(dod);
                self.prev_delta += dod;
                self.prev_timestamp += self.prev_delta;
                self.count += 1;
            }
        } else {
            for &ts in timestamps {
                self.encode(ts);
            }
        }
    }

    pub fn finish(self) -> Vec<u8> {
        self.buf.finish()
    }
}

pub struct DeltaOfDeltaDecoder<'a> {
    reader: BitReader<'a>,
    prev_timestamp: i64,
    prev_delta: i64,
    count: usize,
}

impl<'a> DeltaOfDeltaDecoder<'a> {
    pub fn new(data: &'a [u8]) -> Self {
        Self {
            reader: BitReader::new(data),
            prev_timestamp: 0,
            prev_delta: 0,
            count: 0,
        }
    }

    pub fn decode_all(&mut self, count: usize) -> Vec<i64> {
        let mut result = Vec::with_capacity(count);
        for _ in 0..count {
            result.push(self.decode_one());
        }
        result
    }

    fn decode_one(&mut self) -> i64 {
        if self.count == 0 {
            let ts = self.reader.read_bits(64) as i64;
            self.prev_timestamp = ts;
            self.count += 1;
            return ts;
        }
        if self.count == 1 {
            let delta = self.reader.read_bits(64) as i64;
            self.prev_delta = delta;
            self.prev_timestamp += delta;
            self.count += 1;
            return self.prev_timestamp;
        }

        let dod = self.decode_dod();
        self.prev_delta += dod;
        self.prev_timestamp += self.prev_delta;
        self.count += 1;
        self.prev_timestamp
    }

    fn decode_dod(&mut self) -> i64 {
        if !self.reader.read_bit() {
            return 0;
        }
        if !self.reader.read_bit() {
            return self.reader.read_bits_signed(7);
        }
        if !self.reader.read_bit() {
            return self.reader.read_bits_signed(9);
        }
        if !self.reader.read_bit() {
            return self.reader.read_bits_signed(12);
        }
        self.reader.read_bits(64) as i64
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_constant_interval() {
        let timestamps: Vec<i64> = (0..100).map(|i| 1_000_000_000 + i * 1_000_000).collect();
        let mut encoder = DeltaOfDeltaEncoder::new();
        for &ts in &timestamps {
            encoder.encode(ts);
        }
        let data = encoder.finish();
        let mut decoder = DeltaOfDeltaDecoder::new(&data);
        let decoded = decoder.decode_all(timestamps.len());
        assert_eq!(timestamps, decoded);
    }

    #[test]
    fn round_trip_irregular_intervals() {
        let timestamps = vec![100, 160, 200, 400, 405, 500, 700, 710, 1000];
        let mut encoder = DeltaOfDeltaEncoder::new();
        for &ts in &timestamps {
            encoder.encode(ts);
        }
        let data = encoder.finish();
        let mut decoder = DeltaOfDeltaDecoder::new(&data);
        let decoded = decoder.decode_all(timestamps.len());
        assert_eq!(timestamps, decoded);
    }

    #[test]
    fn round_trip_negative_timestamps() {
        let timestamps = vec![-1000, -500, -200, 0, 100, 500];
        let mut encoder = DeltaOfDeltaEncoder::new();
        for &ts in &timestamps {
            encoder.encode(ts);
        }
        let data = encoder.finish();
        let mut decoder = DeltaOfDeltaDecoder::new(&data);
        let decoded = decoder.decode_all(timestamps.len());
        assert_eq!(timestamps, decoded);
    }

    #[test]
    fn compression_ratio_constant_interval() {
        let timestamps: Vec<i64> = (0..1000).map(|i| i * 1_000_000_000).collect();
        let mut encoder = DeltaOfDeltaEncoder::new();
        for &ts in &timestamps {
            encoder.encode(ts);
        }
        let data = encoder.finish();
        let raw_size = timestamps.len() * 8;
        assert!(
            data.len() < raw_size / 4,
            "compressed {} bytes should be much less than raw {} bytes",
            data.len(),
            raw_size
        );
    }

    #[test]
    fn bit_buffer_basic() {
        let mut buf = BitBuffer::new();
        buf.write_bit(true);
        buf.write_bit(false);
        buf.write_bit(true);
        buf.write_bits(0b1010, 4);
        let data = buf.finish();
        let mut reader = BitReader::new(&data);
        assert!(reader.read_bit());
        assert!(!reader.read_bit());
        assert!(reader.read_bit());
        assert_eq!(reader.read_bits(4), 0b1010);
    }

    #[test]
    fn encode_batch_round_trip() {
        let timestamps: Vec<i64> = (0..1000).map(|i| 1_000_000_000 + i * 1_000_000).collect();
        let mut encoder = DeltaOfDeltaEncoder::new();
        encoder.encode_batch(&timestamps);
        let data = encoder.finish();
        let mut decoder = DeltaOfDeltaDecoder::new(&data);
        let decoded = decoder.decode_all(timestamps.len());
        assert_eq!(timestamps, decoded);
    }

    #[test]
    fn encode_batch_irregular_round_trip() {
        let timestamps: Vec<i64> = (0..500).map(|i| i * 1000 + i * i).collect();
        let mut encoder = DeltaOfDeltaEncoder::new();
        encoder.encode_batch(&timestamps);
        let data = encoder.finish();
        let mut decoder = DeltaOfDeltaDecoder::new(&data);
        let decoded = decoder.decode_all(timestamps.len());
        assert_eq!(timestamps, decoded);
    }

    #[test]
    fn encode_batch_matches_sequential() {
        let timestamps: Vec<i64> = (0..200).map(|i| i * 3000 + i * i * 7).collect();

        let mut enc_seq = DeltaOfDeltaEncoder::new();
        for &ts in &timestamps {
            enc_seq.encode(ts);
        }
        let data_seq = enc_seq.finish();

        let mut enc_batch = DeltaOfDeltaEncoder::new();
        enc_batch.encode_batch(&timestamps);
        let data_batch = enc_batch.finish();

        assert_eq!(data_seq, data_batch);
    }
}
