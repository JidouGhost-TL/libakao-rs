// SPDX-License-Identifier: MIT

//! AKAO companion wave & articulation bank parser, SPU RAM relocation, and upload.

use alloc::vec::Vec;

use crate::sequence::AKAO_MAGIC;
use crate::version::AkaoVersion;
use crate::Error;

/// Unified AKAO wave bank header size in bytes (`V2a`, `V2b`, `V3`).
pub const WAVE_BANK_HEADER_SIZE: usize = 0x40;

/// Hardware articulation descriptor for a single AKAO instrument sample.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Articulation {
    /// Byte address of the ADPCM sample start in PS1 SPU RAM.
    pub start_addr: u32,
    /// Byte address of the ADPCM loop point in PS1 SPU RAM.
    pub loop_addr: u32,
    /// Fine-tune / pitch scaling parameter (`0x1000` = unity).
    pub fine_tune: u16,
    /// PS1 SPU `ADSR1` register value (attack/decay/sustain level).
    pub adsr1: u16,
    /// PS1 SPU `ADSR2` register value (sustain/release rates and modes).
    pub adsr2: u16,
}

impl Articulation {
    /// Parses a 64-byte (`0x40`) AKAO `V1`/`V2a`/`V2b` articulation entry.
    ///
    /// Layout:
    /// - `+0x00`: `start_addr: u32`
    /// - `+0x04`: `loop_addr: u32`
    /// - `+0x08`: `fine_tune: u16`
    /// - `+0x0C`: `adsr1: u16`
    /// - `+0x0E`: `adsr2: u16`
    #[must_use]
    pub fn from_64b(entry: &[u8; 64]) -> Self {
        Self {
            start_addr: u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]),
            loop_addr: u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]),
            fine_tune: u16::from_le_bytes([entry[8], entry[9]]),
            adsr1: u16::from_le_bytes([entry[12], entry[13]]),
            adsr2: u16::from_le_bytes([entry[14], entry[15]]),
        }
    }

    /// Parses a compacted 16-byte (`0x10`) AKAO `V3` articulation entry.
    ///
    /// Layout:
    /// - `+0x00`: `start_addr: u32` (relative to `spu_base_addr` before relocation)
    /// - `+0x04`: `loop_addr: u32` (relative to `spu_base_addr` before relocation)
    /// - `+0x08`: `fine_tune: u16`
    /// - `+0x0C`: `adsr1: u16`
    /// - `+0x0E`: `adsr2: u16`
    #[must_use]
    pub fn from_16b(entry: &[u8; 16]) -> Self {
        Self {
            start_addr: u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]),
            loop_addr: u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]),
            fine_tune: u16::from_le_bytes([entry[8], entry[9]]),
            adsr1: u16::from_le_bytes([entry[12], entry[13]]),
            adsr2: u16::from_le_bytes([entry[14], entry[15]]),
        }
    }
}

/// Parsed AKAO wave bank containing articulation entries and PS1 SPU ADPCM sample data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkaoWaveBank {
    /// Generation format of this wave bank.
    pub version: AkaoVersion,
    /// Target byte address in PS1 SPU RAM where [`Self::adpcm_data`] is uploaded.
    pub spu_base_addr: u32,
    /// Starting instrument index in the global instrument table.
    pub start_index: u32,
    /// Articulation descriptors indexed relative to [`Self::start_index`].
    pub articulations: Vec<Articulation>,
    /// Raw PS1 SPU ADPCM sample payload.
    pub adpcm_data: Vec<u8>,
}

impl AkaoWaveBank {
    /// Constructs a wave bank directly from its components, validating SPU RAM bounds.
    pub fn from_parts(
        version: AkaoVersion,
        spu_base_addr: u32,
        start_index: u32,
        articulations: Vec<Articulation>,
        adpcm_data: Vec<u8>,
    ) -> Result<Self, Error> {
        validate_spu_range(spu_base_addr, adpcm_data.len())?;
        if articulations.len() > version.max_wave_entries() {
            return Err(Error::OutOfRange(
                "wave bank entry count exceeds version limit",
            ));
        }
        Ok(Self {
            version,
            spu_base_addr,
            start_index,
            articulations,
            adpcm_data,
        })
    }

    /// Parses an AKAO wave bank for the given [`AkaoVersion`].
    ///
    /// - **`V1`**: Supports both unified `"AKAO"` 64-byte headers and raw `V1` 16-byte
    ///   headers (`+0x00 spu_addr`, `+0x04 adpcm_len`, `+0x08 art_count`, followed by
    ///   `art_count * 64` bytes of articulations and `adpcm_len` bytes of ADPCM). For
    ///   separate `.DAT` + `.ALL` buffers, use [`Self::parse_v1_separate`].
    /// - **`V2a`**: Unified `"AKAO"` 64-byte header (`+0x10 spu_addr`, `+0x14 adpcm_len`,
    ///   `+0x18 start_idx`), 64-byte entries (up to 128).
    /// - **`V2b`**: Unified `"AKAO"` 64-byte header (`+0x10 spu_addr`, `+0x14 adpcm_len`,
    ///   `+0x18 start_idx`, `+0x1C end_idx` with `0 => 256`), 64-byte entries (up to 256).
    /// - **`V3`**: Unified `"AKAO"` 64-byte header (`+0x10 spu_addr`, `+0x14 adpcm_len`,
    ///   `+0x18 start_idx`, `+0x1C count`), compacted 16-byte entries whose relative
    ///   `start_addr` and `loop_addr` are relocated by `+ spu_base_addr`.
    pub fn parse(bytes: &[u8], version: AkaoVersion) -> Result<Self, Error> {
        match version {
            AkaoVersion::V1 => {
                if bytes.len() >= 4 && bytes[0..4] == AKAO_MAGIC {
                    parse_unified_bank(bytes, AkaoVersion::V1)
                } else {
                    parse_v1_raw_combined(bytes)
                }
            }
            AkaoVersion::V2a | AkaoVersion::V2b | AkaoVersion::V3 => {
                parse_unified_bank(bytes, version)
            }
        }
    }

    /// Parses separate AKAO `V1` articulation (`.DAT`, 64 B entries) and sample (`.ALL`,
    /// 16 B header `+0x00 spu_addr`, `+0x04 adpcm_len`, data at `+0x10`) buffers.
    pub fn parse_v1_separate(art_bytes: &[u8], all_bytes: &[u8]) -> Result<Self, Error> {
        if all_bytes.len() < 0x10 {
            return Err(Error::UnexpectedEof("V1 .ALL header"));
        }
        let spu_base_addr = read_u32_le(all_bytes, 0x00);
        let adpcm_len = read_u32_le(all_bytes, 0x04) as usize;
        if 0x10 + adpcm_len > all_bytes.len() {
            return Err(Error::UnexpectedEof("V1 .ALL ADPCM payload"));
        }
        if art_bytes.len() % 64 != 0 {
            return Err(Error::InvalidHeader(
                "V1 .DAT articulation table must be a multiple of 64 bytes",
            ));
        }
        let count = art_bytes.len() / 64;
        if count > AkaoVersion::V1.max_wave_entries() {
            return Err(Error::OutOfRange("V1 articulation count exceeds 128"));
        }
        validate_spu_range(spu_base_addr, adpcm_len)?;

        let mut articulations = Vec::with_capacity(count);
        for chunk in art_bytes.chunks_exact(64) {
            let mut arr = [0u8; 64];
            arr.copy_from_slice(chunk);
            articulations.push(Articulation::from_64b(&arr));
        }

        Ok(Self {
            version: AkaoVersion::V1,
            spu_base_addr,
            start_index: 0,
            articulations,
            adpcm_data: all_bytes[0x10..0x10 + adpcm_len].to_vec(),
        })
    }

    /// Looks up an articulation by global instrument index (`index >= start_index`).
    #[inline]
    #[must_use]
    pub fn articulation(&self, index: usize) -> Option<&Articulation> {
        let start = self.start_index as usize;
        let rel = index.checked_sub(start)?;
        self.articulations.get(rel)
    }

    /// Dynamically relocates this wave bank to `new_spu_base` in PS1 SPU RAM.
    ///
    /// Adjusts [`Self::spu_base_addr`] and shifts every articulation's `start_addr`
    /// and `loop_addr` by `new_spu_base - old_spu_base` (preserving zero addresses).
    pub fn relocate(&mut self, new_spu_base: u32) -> Result<(), Error> {
        if (new_spu_base & 0x7) != 0 {
            return Err(Error::InvalidArgument(
                "SPU base address must be 8-byte aligned",
            ));
        }
        validate_spu_range(new_spu_base, self.adpcm_data.len())?;

        let old_base = i64::from(self.spu_base_addr);
        let delta = i64::from(new_spu_base) - old_base;

        for art in &mut self.articulations {
            if art.start_addr != 0 {
                let shifted = i64::from(art.start_addr) + delta;
                if shifted < 0 || shifted as usize >= libspu2::PS1_RAM_SIZE {
                    return Err(Error::OutOfRange("relocated start_addr out of SPU RAM"));
                }
                art.start_addr = shifted as u32;
            }
            if art.loop_addr != 0 {
                let shifted = i64::from(art.loop_addr) + delta;
                if shifted < 0 || shifted as usize >= libspu2::PS1_RAM_SIZE {
                    return Err(Error::OutOfRange("relocated loop_addr out of SPU RAM"));
                }
                art.loop_addr = shifted as u32;
            }
        }

        self.spu_base_addr = new_spu_base;
        Ok(())
    }

    /// Uploads [`Self::adpcm_data`] into `device` at [`Self::spu_base_addr`].
    pub fn upload_to_spu(&self, device: &mut libspu2::Device) -> Result<(), Error> {
        let end = (self.spu_base_addr as usize)
            .checked_add(self.adpcm_data.len())
            .ok_or(Error::OutOfRange("SPU RAM upload range overflow"))?;
        if end > device.ram_size() {
            return Err(Error::OutOfRange("wave bank exceeds SPU RAM size"));
        }
        device.write_ram(self.spu_base_addr, &self.adpcm_data)?;
        Ok(())
    }
}

fn parse_v1_raw_combined(bytes: &[u8]) -> Result<AkaoWaveBank, Error> {
    if bytes.len() < 0x10 {
        return Err(Error::UnexpectedEof("V1 wave bank header"));
    }
    let spu_base_addr = read_u32_le(bytes, 0x00);
    let adpcm_len = read_u32_le(bytes, 0x04) as usize;
    let art_count = read_u32_le(bytes, 0x08) as usize;
    if art_count > AkaoVersion::V1.max_wave_entries() {
        return Err(Error::OutOfRange("V1 wave bank entry count exceeds 128"));
    }
    let art_bytes_len = art_count
        .checked_mul(64)
        .ok_or(Error::OutOfRange("V1 articulation table size overflow"))?;
    let adpcm_offset = 0x10usize
        .checked_add(art_bytes_len)
        .ok_or(Error::OutOfRange("V1 ADPCM offset overflow"))?;
    let total_needed = adpcm_offset
        .checked_add(adpcm_len)
        .ok_or(Error::OutOfRange("V1 total size overflow"))?;
    if total_needed > bytes.len() {
        return Err(Error::UnexpectedEof("V1 wave bank payload"));
    }
    validate_spu_range(spu_base_addr, adpcm_len)?;

    let mut articulations = Vec::with_capacity(art_count);
    for chunk in bytes[0x10..adpcm_offset].chunks_exact(64) {
        let mut arr = [0u8; 64];
        arr.copy_from_slice(chunk);
        articulations.push(Articulation::from_64b(&arr));
    }

    Ok(AkaoWaveBank {
        version: AkaoVersion::V1,
        spu_base_addr,
        start_index: 0,
        articulations,
        adpcm_data: bytes[adpcm_offset..adpcm_offset + adpcm_len].to_vec(),
    })
}

fn parse_unified_bank(bytes: &[u8], version: AkaoVersion) -> Result<AkaoWaveBank, Error> {
    if bytes.len() < WAVE_BANK_HEADER_SIZE {
        return Err(Error::UnexpectedEof("AKAO wave bank 64-byte header"));
    }
    if bytes[0..4] != AKAO_MAGIC {
        return Err(Error::InvalidMagic);
    }

    let spu_base_addr = read_u32_le(bytes, 0x10);
    let adpcm_len = read_u32_le(bytes, 0x14) as usize;
    let start_index = read_u32_le(bytes, 0x18);
    let field_1c = read_u32_le(bytes, 0x1C);
    let stride = version.articulation_stride();
    let max_entries = version.max_wave_entries();

    let count = match version {
        AkaoVersion::V1 | AkaoVersion::V2a => {
            if start_index as usize > max_entries {
                return Err(Error::OutOfRange("wave bank start_index exceeds 128"));
            }
            // If field_1c is explicitly set and valid, respect it; otherwise infer from remaining bytes
            // or `128 - start_index`.
            let max_by_index = max_entries - start_index as usize;
            let avail_for_art = bytes
                .len()
                .checked_sub(WAVE_BANK_HEADER_SIZE)
                .and_then(|rem| rem.checked_sub(adpcm_len))
                .ok_or(Error::UnexpectedEof("wave bank ADPCM payload"))?;
            if field_1c != 0 {
                let c = field_1c as usize;
                if c > max_by_index {
                    return Err(Error::OutOfRange("wave bank entry count exceeds limit"));
                }
                c
            } else {
                let by_size = avail_for_art / stride;
                by_size.min(max_by_index)
            }
        }
        AkaoVersion::V2b => {
            let end_index = if field_1c == 0 {
                max_entries as u32
            } else {
                field_1c
            };
            if end_index < start_index || end_index as usize > max_entries {
                return Err(Error::OutOfRange(
                    "V2b wave bank start/end index out of range",
                ));
            }
            (end_index - start_index) as usize
        }
        AkaoVersion::V3 => {
            let c = field_1c as usize;
            if c > max_entries {
                return Err(Error::OutOfRange("V3 wave bank entry count exceeds 256"));
            }
            c
        }
    };

    let art_bytes_len = count
        .checked_mul(stride)
        .ok_or(Error::OutOfRange("articulation table size overflow"))?;
    let adpcm_offset = WAVE_BANK_HEADER_SIZE
        .checked_add(art_bytes_len)
        .ok_or(Error::OutOfRange("ADPCM offset overflow"))?;
    let total_needed = adpcm_offset
        .checked_add(adpcm_len)
        .ok_or(Error::OutOfRange("wave bank total size overflow"))?;
    if total_needed > bytes.len() {
        return Err(Error::UnexpectedEof("AKAO wave bank payload"));
    }

    validate_spu_range(spu_base_addr, adpcm_len)?;

    let mut articulations = Vec::with_capacity(count);
    let art_slice = &bytes[WAVE_BANK_HEADER_SIZE..adpcm_offset];
    match version {
        AkaoVersion::V1 | AkaoVersion::V2a | AkaoVersion::V2b => {
            for chunk in art_slice.chunks_exact(64) {
                let mut arr = [0u8; 64];
                arr.copy_from_slice(chunk);
                articulations.push(Articulation::from_64b(&arr));
            }
        }
        AkaoVersion::V3 => {
            for chunk in art_slice.chunks_exact(16) {
                let mut arr = [0u8; 16];
                arr.copy_from_slice(chunk);
                let mut art = Articulation::from_16b(&arr);
                // V3 wave bank entries store SPU addresses relative to spu_base_addr.
                art.start_addr = art
                    .start_addr
                    .checked_add(spu_base_addr)
                    .ok_or(Error::OutOfRange("V3 start_addr relocation overflow"))?;
                art.loop_addr = art
                    .loop_addr
                    .checked_add(spu_base_addr)
                    .ok_or(Error::OutOfRange("V3 loop_addr relocation overflow"))?;
                if art.start_addr as usize >= libspu2::PS1_RAM_SIZE
                    || art.loop_addr as usize >= libspu2::PS1_RAM_SIZE
                {
                    return Err(Error::OutOfRange(
                        "V3 relocated address exceeds PS1 SPU RAM",
                    ));
                }
                articulations.push(art);
            }
        }
    }

    Ok(AkaoWaveBank {
        version,
        spu_base_addr,
        start_index,
        articulations,
        adpcm_data: bytes[adpcm_offset..adpcm_offset + adpcm_len].to_vec(),
    })
}

fn validate_spu_range(spu_base_addr: u32, adpcm_len: usize) -> Result<(), Error> {
    let end = (spu_base_addr as usize)
        .checked_add(adpcm_len)
        .ok_or(Error::OutOfRange("SPU RAM range overflow"))?;
    if end > libspu2::PS1_RAM_SIZE {
        return Err(Error::OutOfRange("wave bank exceeds 512 KiB PS1 SPU RAM"));
    }
    Ok(())
}

#[inline]
fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
