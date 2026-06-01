use std::fs::File;
use std::io::{self, BufReader, BufWriter, Read, Seek, SeekFrom, Write};
use std::path::Path;

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;

use crate::compression::delta::{DeltaOfDeltaDecoder, DeltaOfDeltaEncoder};
use crate::compression::xor_float::{XorFloatDecoder, XorFloatEncoder};
use crate::types::TimeSeriesPoint;

const MAGIC: &[u8; 4] = b"TSDB";
const VERSION: u16 = 1;
const FLAG_DELTA_OF_DELTA: u16 = 0x01;
const FLAG_XOR_FLOAT: u16 = 0x02;
const HEADER_SIZE: usize = 32;
pub const BLOCK_SIZE: usize = 1024;

#[derive(Debug, Clone)]
struct IndexEntry {
    first_timestamp: i64,
    file_offset: u64,
    compressed_size: u32,
    point_count: u32,
}

pub struct SSTableWriter {
    writer: BufWriter<File>,
    index_entries: Vec<IndexEntry>,
    min_timestamp: i64,
    max_timestamp: i64,
    total_points: u64,
}

impl SSTableWriter {
    pub fn create(path: &Path) -> io::Result<Self> {
        let file = File::create(path)?;
        let mut writer = BufWriter::new(file);

        // Write placeholder header (will be overwritten in finish())
        let placeholder = [0u8; HEADER_SIZE];
        writer.write_all(&placeholder)?;

        Ok(Self {
            writer,
            index_entries: Vec::new(),
            min_timestamp: i64::MAX,
            max_timestamp: i64::MIN,
            total_points: 0,
        })
    }

    pub fn write_all(&mut self, points: &[TimeSeriesPoint]) -> io::Result<()> {
        for chunk in points.chunks(BLOCK_SIZE) {
            self.write_block(chunk)?;
        }
        Ok(())
    }

    fn write_block(&mut self, points: &[TimeSeriesPoint]) -> io::Result<()> {
        if points.is_empty() {
            return Ok(());
        }

        let first_ts = points[0].timestamp;
        let last_ts = points[points.len() - 1].timestamp;
        self.min_timestamp = self.min_timestamp.min(first_ts);
        self.max_timestamp = self.max_timestamp.max(last_ts);
        self.total_points += points.len() as u64;

        // Compress timestamps using SIMD-accelerated batch path
        let mut ts_encoder = DeltaOfDeltaEncoder::new();
        let timestamps: Vec<i64> = points.iter().map(|p| p.timestamp).collect();
        ts_encoder.encode_batch(&timestamps);
        let ts_compressed = ts_encoder.finish();

        // Compress values using SIMD-accelerated batch path
        let mut val_encoder = XorFloatEncoder::new();
        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        val_encoder.encode_batch(&values);
        let val_compressed = val_encoder.finish();

        let block_start = self.current_offset();

        // Write: ts_len(u32) | ts_data | val_len(u32) | val_data
        self.writer.write_u32::<LittleEndian>(ts_compressed.len() as u32)?;
        self.writer.write_all(&ts_compressed)?;
        self.writer.write_u32::<LittleEndian>(val_compressed.len() as u32)?;
        self.writer.write_all(&val_compressed)?;

        let block_size = 4 + ts_compressed.len() + 4 + val_compressed.len();

        self.index_entries.push(IndexEntry {
            first_timestamp: first_ts,
            file_offset: block_start,
            compressed_size: block_size as u32,
            point_count: points.len() as u32,
        });

        Ok(())
    }

    fn current_offset(&self) -> u64 {
        let data_written: u64 = self
            .index_entries
            .iter()
            .map(|e| e.compressed_size as u64)
            .sum();
        HEADER_SIZE as u64 + data_written
    }

    pub fn finish(mut self) -> io::Result<()> {
        let index_offset = self.current_offset();

        // Write index block
        for entry in &self.index_entries {
            self.writer.write_i64::<LittleEndian>(entry.first_timestamp)?;
            self.writer.write_u64::<LittleEndian>(entry.file_offset)?;
            self.writer.write_u32::<LittleEndian>(entry.compressed_size)?;
            self.writer.write_u32::<LittleEndian>(entry.point_count)?;
        }

        let index_len = (self.index_entries.len() * 24) as u32;

        // Write footer
        self.writer.write_u64::<LittleEndian>(index_offset)?;
        self.writer.write_u32::<LittleEndian>(index_len)?;
        self.writer.write_u64::<LittleEndian>(self.total_points)?;
        self.writer.write_u32::<LittleEndian>(0)?; // CRC placeholder

        // Seek back and write the real header
        self.writer.flush()?;
        let mut file = self.writer.into_inner()?;
        file.seek(SeekFrom::Start(0))?;

        let block_count = self.index_entries.len() as u32;
        let flags = FLAG_DELTA_OF_DELTA | FLAG_XOR_FLOAT;

        file.write_all(MAGIC)?;
        file.write_u16::<LittleEndian>(VERSION)?;
        file.write_u16::<LittleEndian>(flags)?;
        file.write_u32::<LittleEndian>(block_count)?;
        file.write_i64::<LittleEndian>(self.min_timestamp)?;
        file.write_i64::<LittleEndian>(self.max_timestamp)?;
        file.write_all(&[0u8; 4])?; // reserved

        // Compute CRC over entire file (except last 4 bytes)
        file.seek(SeekFrom::Start(0))?;
        let file_len = file.seek(SeekFrom::End(0))?;
        file.seek(SeekFrom::Start(0))?;
        let crc_range = file_len - 4;
        let mut hasher = Hasher::new();
        let mut buf = vec![0u8; 8192];
        let mut read_total = 0u64;
        loop {
            let to_read = ((crc_range - read_total) as usize).min(buf.len());
            if to_read == 0 {
                break;
            }
            let n = file.read(&mut buf[..to_read])?;
            if n == 0 {
                break;
            }
            hasher.update(&buf[..n]);
            read_total += n as u64;
        }
        let crc = hasher.finalize();

        file.seek(SeekFrom::End(-4))?;
        file.write_u32::<LittleEndian>(crc)?;
        file.sync_all()?;

        Ok(())
    }
}

pub struct SSTableReader {
    data: Vec<u8>,
    index_entries: Vec<IndexEntry>,
    #[allow(dead_code)]
    min_timestamp: i64,
    #[allow(dead_code)]
    max_timestamp: i64,
}

impl SSTableReader {
    pub fn open(path: &Path) -> io::Result<Self> {
        let mut file = BufReader::new(File::open(path)?);
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        if data.len() < HEADER_SIZE + 24 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        // Validate magic
        if &data[..4] != MAGIC {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "invalid magic"));
        }

        // Parse header
        let mut cursor = &data[4..];
        let _version = cursor.read_u16::<LittleEndian>()?;
        let _flags = cursor.read_u16::<LittleEndian>()?;
        let block_count = cursor.read_u32::<LittleEndian>()?;
        let min_timestamp = cursor.read_i64::<LittleEndian>()?;
        let max_timestamp = cursor.read_i64::<LittleEndian>()?;
        let _reserved = cursor.read_u32::<LittleEndian>()?;

        // Parse footer (last 24 bytes)
        let footer_start = data.len() - 24;
        let mut footer = &data[footer_start..];
        let index_offset = footer.read_u64::<LittleEndian>()?;
        let index_len = footer.read_u32::<LittleEndian>()?;
        let _total_points = footer.read_u64::<LittleEndian>()?;
        let _crc = footer.read_u32::<LittleEndian>()?;

        // Parse index block
        let index_start = index_offset as usize;
        let index_end = index_start + index_len as usize;
        let mut index_cursor = &data[index_start..index_end];
        let mut index_entries = Vec::with_capacity(block_count as usize);

        for _ in 0..block_count {
            let first_timestamp = index_cursor.read_i64::<LittleEndian>()?;
            let file_offset = index_cursor.read_u64::<LittleEndian>()?;
            let compressed_size = index_cursor.read_u32::<LittleEndian>()?;
            let point_count = index_cursor.read_u32::<LittleEndian>()?;
            index_entries.push(IndexEntry {
                first_timestamp,
                file_offset,
                compressed_size,
                point_count,
            });
        }

        Ok(Self {
            data,
            index_entries,
            min_timestamp,
            max_timestamp,
        })
    }

    pub fn scan_range(&self, start: i64, end: i64) -> io::Result<Vec<TimeSeriesPoint>> {
        let mut result = Vec::new();

        for entry in &self.index_entries {
            // Skip blocks that are entirely before or after the range
            if entry.first_timestamp > end {
                break;
            }

            let block_data = &self.data[entry.file_offset as usize
                ..(entry.file_offset as usize + entry.compressed_size as usize)];
            let points = self.decode_block(block_data, entry.point_count as usize)?;

            for p in points {
                if p.timestamp >= start && p.timestamp <= end {
                    result.push(p);
                }
            }
        }

        Ok(result)
    }

    pub fn read_all(&self) -> io::Result<Vec<TimeSeriesPoint>> {
        let mut result = Vec::new();
        for entry in &self.index_entries {
            let block_data = &self.data[entry.file_offset as usize
                ..(entry.file_offset as usize + entry.compressed_size as usize)];
            let points = self.decode_block(block_data, entry.point_count as usize)?;
            result.extend(points);
        }
        Ok(result)
    }

    fn decode_block(&self, block_data: &[u8], count: usize) -> io::Result<Vec<TimeSeriesPoint>> {
        let mut cursor = block_data;

        let ts_len = cursor.read_u32::<LittleEndian>()? as usize;
        let ts_data = &cursor[..ts_len];
        cursor = &cursor[ts_len..];

        let val_len = cursor.read_u32::<LittleEndian>()? as usize;
        let val_data = &cursor[..val_len];

        let mut ts_decoder = DeltaOfDeltaDecoder::new(ts_data);
        let timestamps = ts_decoder.decode_all(count);

        let mut val_decoder = XorFloatDecoder::new(val_data);
        let values = val_decoder.decode_all(count);

        Ok(timestamps
            .into_iter()
            .zip(values.into_iter())
            .map(|(timestamp, value)| TimeSeriesPoint { timestamp, value })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_and_read_back() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.sst");

        let points: Vec<TimeSeriesPoint> = (0..2500)
            .map(|i| TimeSeriesPoint {
                timestamp: 1_000_000_000 + i * 1_000_000,
                value: 25.0 + (i as f64) * 0.01,
            })
            .collect();

        let mut writer = SSTableWriter::create(&path).unwrap();
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

    #[test]
    fn range_scan() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("test.sst");

        let points: Vec<TimeSeriesPoint> = (0..5000)
            .map(|i| TimeSeriesPoint {
                timestamp: i * 1000,
                value: i as f64,
            })
            .collect();

        let mut writer = SSTableWriter::create(&path).unwrap();
        writer.write_all(&points).unwrap();
        writer.finish().unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        let range = reader.scan_range(1000_000, 2000_000).unwrap();
        for p in &range {
            assert!(p.timestamp >= 1000_000 && p.timestamp <= 2000_000);
        }
        assert!(!range.is_empty());
    }

    #[test]
    fn validates_magic() {
        let dir = TempDir::new().unwrap();
        let path = dir.path().join("bad.sst");
        std::fs::write(&path, &[0u8; 100]).unwrap();
        assert!(SSTableReader::open(&path).is_err());
    }
}
