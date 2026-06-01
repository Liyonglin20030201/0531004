use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use crc32fast::Hasher;

use crate::types::TimeSeriesPoint;

const RECORD_LEN: u32 = 20; // 4 bytes crc + 8 bytes timestamp + 8 bytes value

pub struct Wal {
    writer: BufWriter<File>,
    path: PathBuf,
}

impl Wal {
    pub fn open(dir: &Path) -> io::Result<Self> {
        fs::create_dir_all(dir)?;
        let path = dir.join("wal.bin");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            path,
        })
    }

    pub fn append(&mut self, point: TimeSeriesPoint) -> io::Result<()> {
        let mut hasher = Hasher::new();
        let mut payload = [0u8; 16];
        (&mut payload[..8]).write_i64::<LittleEndian>(point.timestamp)?;
        (&mut payload[8..]).write_f64::<LittleEndian>(point.value)?;
        hasher.update(&payload);
        let crc = hasher.finalize();

        self.writer.write_u32::<LittleEndian>(RECORD_LEN)?;
        self.writer.write_u32::<LittleEndian>(crc)?;
        self.writer.write_all(&payload)?;
        Ok(())
    }

    pub fn sync(&mut self) -> io::Result<()> {
        self.writer.flush()?;
        self.writer.get_ref().sync_all()
    }

    pub fn discard(self) -> io::Result<()> {
        drop(self.writer);
        fs::remove_file(&self.path)
    }

    pub fn replay(dir: &Path) -> io::Result<Vec<TimeSeriesPoint>> {
        let path = dir.join("wal.bin");
        if !path.exists() {
            return Ok(Vec::new());
        }

        let mut file = File::open(&path)?;
        let mut data = Vec::new();
        file.read_to_end(&mut data)?;

        let mut points = Vec::new();
        let mut cursor = &data[..];

        loop {
            if cursor.len() < 24 {
                break;
            }

            let record_len = cursor.read_u32::<LittleEndian>()?;
            if record_len != RECORD_LEN {
                break;
            }

            if cursor.len() < 20 {
                break;
            }

            let stored_crc = cursor.read_u32::<LittleEndian>()?;
            let mut payload = [0u8; 16];
            io::Read::read_exact(&mut cursor, &mut payload)?;

            let mut hasher = Hasher::new();
            hasher.update(&payload);
            let computed_crc = hasher.finalize();

            if stored_crc != computed_crc {
                break;
            }

            let timestamp = (&payload[..8]).read_i64::<LittleEndian>()?;
            let value = (&payload[8..]).read_f64::<LittleEndian>()?;
            points.push(TimeSeriesPoint { timestamp, value });
        }

        Ok(points)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::TempDir;

    #[test]
    fn write_and_replay() {
        let dir = TempDir::new().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();

        let points = vec![
            TimeSeriesPoint { timestamp: 1000, value: 25.5 },
            TimeSeriesPoint { timestamp: 2000, value: 26.0 },
            TimeSeriesPoint { timestamp: 3000, value: 24.8 },
        ];

        for &p in &points {
            wal.append(p).unwrap();
        }
        wal.sync().unwrap();
        drop(wal);

        let replayed = Wal::replay(dir.path()).unwrap();
        assert_eq!(replayed.len(), 3);
        for (orig, rep) in points.iter().zip(replayed.iter()) {
            assert_eq!(orig.timestamp, rep.timestamp);
            assert_eq!(orig.value.to_bits(), rep.value.to_bits());
        }
    }

    #[test]
    fn replay_truncated_wal() {
        let dir = TempDir::new().unwrap();
        let mut wal = Wal::open(dir.path()).unwrap();

        for i in 0..5 {
            wal.append(TimeSeriesPoint { timestamp: i * 1000, value: i as f64 }).unwrap();
        }
        wal.sync().unwrap();
        drop(wal);

        // Truncate last few bytes to simulate crash
        let path = dir.path().join("wal.bin");
        let data = fs::read(&path).unwrap();
        fs::write(&path, &data[..data.len() - 10]).unwrap();

        let replayed = Wal::replay(dir.path()).unwrap();
        assert_eq!(replayed.len(), 4); // Last record is incomplete
    }

    #[test]
    fn replay_nonexistent() {
        let dir = TempDir::new().unwrap();
        let replayed = Wal::replay(dir.path()).unwrap();
        assert!(replayed.is_empty());
    }

    #[test]
    fn discard_removes_file() {
        let dir = TempDir::new().unwrap();
        let wal = Wal::open(dir.path()).unwrap();
        let path = dir.path().join("wal.bin");
        assert!(path.exists());
        wal.discard().unwrap();
        assert!(!path.exists());
    }
}
