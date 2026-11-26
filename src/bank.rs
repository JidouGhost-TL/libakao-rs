// SPDX-License-Identifier: MIT OR Apache-2.0

//! AKAO companion wave & articulation bank parser, SPU RAM relocation, and upload.
//!
//! [`AkaoWaveBankView`] provides zero-allocation borrowed parsing usable in pure
//! `#![no_std]` environments without `alloc`. With the `alloc` feature enabled,
//! `AkaoWaveBank` provides an owned representation with relocation and upload.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::sequence::AKAO_MAGIC;
use crate::version::AkaoVersion;
use crate::{Error, Result};

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
    pub const fn from_64b(entry: &[u8; 64]) -> Self {
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
    pub const fn from_16b(entry: &[u8; 16]) -> Self {
        Self {
            start_addr: u32::from_le_bytes([entry[0], entry[1], entry[2], entry[3]]),
            loop_addr: u32::from_le_bytes([entry[4], entry[5], entry[6], entry[7]]),
            fine_tune: u16::from_le_bytes([entry[8], entry[9]]),
            adsr1: u16::from_le_bytes([entry[12], entry[13]]),
            adsr2: u16::from_le_bytes([entry[14], entry[15]]),
        }
    }
}

/// Zero-allocation borrowed view of a parsed AKAO wave & articulation bank.
///
/// Usable in pure `#![no_std]` builds without the `alloc` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkaoWaveBankView<'a> {
    /// Generation format of this wave bank.
    pub version: AkaoVersion,
    /// Target byte address in PS1 SPU RAM where ADPCM data is uploaded.
    pub spu_base_addr: u32,
    /// Starting instrument index in the global instrument table.
    pub start_index: u32,
    art_slice: &'a [u8],
    art_count: usize,
    adpcm_data: &'a [u8],
}

impl<'a> AkaoWaveBankView<'a> {
    /// Returns `true` if `bytes` has a structurally valid 64-byte unified `"AKAO"` wave bank
    /// header (`declared_length == 0`, zero reserved block `+0x20..+0x40`, 16-byte-aligned
    /// `spu_base_addr` and non-zero `adpcm_len` within PS1 SPU RAM, valid articulation bounds,
    /// and sufficient payload bytes — permitting trailing sector padding).
    #[must_use]
    pub fn is_unified_header(bytes: &[u8]) -> bool {
        if bytes.len() < WAVE_BANK_HEADER_SIZE + 0x20 {
            return false;
        }
        if bytes[0..4] != AKAO_MAGIC {
            return false;
        }
        let decl = u16::from_le_bytes([bytes[6], bytes[7]]);
        if decl != 0 {
            return false;
        }
        if bytes[0x20..WAVE_BANK_HEADER_SIZE].iter().any(|&b| b != 0) {
            return false;
        }
        let spu_base_addr = read_u32_le(bytes, 0x10);
        let adpcm_len = read_u32_le(bytes, 0x14) as usize;
        let start_index = read_u32_le(bytes, 0x18) as usize;
        let field_1c = read_u32_le(bytes, 0x1C) as usize;
        if (spu_base_addr & 0xF) != 0 || spu_base_addr as usize >= libspu2::PS1_RAM_SIZE {
            return false;
        }
        if adpcm_len == 0 || (adpcm_len & 0xF) != 0 || adpcm_len >= bytes.len() {
            return false;
        }
        if validate_spu_range(spu_base_addr, adpcm_len).is_err() {
            return false;
        }
        let max_v3 = AkaoVersion::V3.max_wave_entries();
        if start_index > max_v3 || field_1c > max_v3 {
            return false;
        }

        // Verify that at least one unified generation layout (V1/V2a, V2b, or V3) fits within
        // `bytes.len()`, matching `parse_unified_bank_view` (including sector-padded banks).
        let fits = |count: usize, stride: usize| -> bool {
            count > 0
                && count
                    .checked_mul(stride)
                    .and_then(|art_len| WAVE_BANK_HEADER_SIZE.checked_add(art_len))
                    .and_then(|off| off.checked_add(adpcm_len))
                    .is_some_and(|total| total <= bytes.len())
        };

        let v3_fits = field_1c > 0 && fits(field_1c, 16);
        let v1_v2a_fits = start_index <= 128
            && if field_1c != 0 {
                field_1c <= 128 - start_index && fits(field_1c, 64)
            } else {
                fits(1, 64)
            };
        let v2b_end = if field_1c == 0 { max_v3 } else { field_1c };
        let v2b_fits = v2b_end > start_index && fits(v2b_end - start_index, 64);

        v3_fits || v1_v2a_fits || v2b_fits
    }

    /// Parses a borrowed AKAO wave bank view for `version` without allocating.
    pub fn parse(bytes: &'a [u8], version: AkaoVersion) -> Result<Self> {
        match version {
            AkaoVersion::V1 => {
                if bytes.len() >= 4 && bytes[0..4] == AKAO_MAGIC {
                    parse_unified_bank_view(bytes, AkaoVersion::V1)
                } else {
                    parse_v1_raw_combined_view(bytes)
                }
            }
            AkaoVersion::V2a | AkaoVersion::V2b | AkaoVersion::V3 => {
                parse_unified_bank_view(bytes, version)
            }
        }
    }

    /// Parses separate AKAO `V1` articulation (`.DAT`, 64 B entries) and sample (`.ALL`)
    /// buffers into a borrowed view without allocating.
    pub fn parse_v1_separate(art_bytes: &'a [u8], all_bytes: &'a [u8]) -> Result<Self> {
        if all_bytes.len() < 0x10 {
            return Err(Error::UnexpectedEof("V1 .ALL header"));
        }
        let spu_base_addr = read_u32_le(all_bytes, 0x00);
        let adpcm_len = read_u32_le(all_bytes, 0x04) as usize;
        let adpcm_end = 0x10usize
            .checked_add(adpcm_len)
            .ok_or(Error::OutOfRange("V1 .ALL ADPCM size overflow"))?;
        if adpcm_end > all_bytes.len() {
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

        Ok(Self {
            version: AkaoVersion::V1,
            spu_base_addr,
            start_index: 0,
            art_slice: art_bytes,
            art_count: count,
            adpcm_data: &all_bytes[0x10..adpcm_end],
        })
    }

    /// Returns the number of articulation entries in this bank.
    #[inline]
    #[must_use]
    pub const fn articulation_count(&self) -> usize {
        self.art_count
    }

    /// Decodes an articulation by global instrument index (`index >= start_index`).
    #[must_use]
    pub fn articulation(&self, index: usize) -> Option<Articulation> {
        let rel = index.checked_sub(self.start_index as usize)?;
        if rel >= self.art_count {
            return None;
        }
        let stride = self.version.articulation_stride();
        let off = rel * stride;
        match self.version {
            AkaoVersion::V1 | AkaoVersion::V2a | AkaoVersion::V2b => {
                let arr: [u8; 64] = self.art_slice[off..off + 64].try_into().ok()?;
                Some(Articulation::from_64b(&arr))
            }
            AkaoVersion::V3 => {
                let arr: [u8; 16] = self.art_slice[off..off + 16].try_into().ok()?;
                let mut art = Articulation::from_16b(&arr);
                art.start_addr = art.start_addr.checked_add(self.spu_base_addr)?;
                art.loop_addr = art.loop_addr.checked_add(self.spu_base_addr)?;
                Some(art)
            }
        }
    }

    /// Returns the raw PS1 SPU ADPCM payload slice.
    #[inline]
    #[must_use]
    pub const fn adpcm_data(&self) -> &'a [u8] {
        self.adpcm_data
    }

    /// Converts this borrowed view into an owned [`AkaoWaveBank`].
    #[cfg(feature = "alloc")]
    pub fn to_owned_bank(&self) -> Result<AkaoWaveBank> {
        let mut articulations = Vec::with_capacity(self.art_count);
        let base = self.start_index as usize;
        for rel in 0..self.art_count {
            let art = self
                .articulation(base + rel)
                .ok_or(Error::OutOfRange("articulation decode failed"))?;
            articulations.push(art);
        }
        Ok(AkaoWaveBank {
            version: self.version,
            spu_base_addr: self.spu_base_addr,
            start_index: self.start_index,
            articulations,
            adpcm_data: self.adpcm_data.to_vec(),
        })
    }
}

/// Owned parsed AKAO wave bank containing articulation entries and PS1 SPU ADPCM sample data.
#[cfg(feature = "alloc")]
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

#[cfg(feature = "alloc")]
impl AkaoWaveBank {
    /// Constructs a wave bank directly from its components, validating SPU RAM bounds.
    pub fn from_parts(
        version: AkaoVersion,
        spu_base_addr: u32,
        start_index: u32,
        articulations: Vec<Articulation>,
        adpcm_data: Vec<u8>,
    ) -> Result<Self> {
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
    pub fn parse(bytes: &[u8], version: AkaoVersion) -> Result<Self> {
        let view = AkaoWaveBankView::parse(bytes, version)?;
        view.to_owned_bank()
    }

    /// Parses separate AKAO `V1` articulation (`.DAT`, 64 B entries) and sample (`.ALL`,
    /// 16 B header `+0x00 spu_addr`, `+0x04 adpcm_len`, data at `+0x10`) buffers.
    pub fn parse_v1_separate(art_bytes: &[u8], all_bytes: &[u8]) -> Result<Self> {
        let view = AkaoWaveBankView::parse_v1_separate(art_bytes, all_bytes)?;
        view.to_owned_bank()
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
    pub fn relocate(&mut self, new_spu_base: u32) -> Result<()> {
        if (new_spu_base & 0x7) != 0 {
            return Err(Error::InvalidArgument(
                "SPU base address must be 8-byte aligned",
            ));
        }
        validate_spu_range(new_spu_base, self.adpcm_data.len())?;

        let old_base = i64::from(self.spu_base_addr);
        let delta = i64::from(new_spu_base) - old_base;

        // First pass: validate all shifted addresses so errors never leave partial mutations.
        for art in &self.articulations {
            if art.start_addr != 0 {
                let shifted = i64::from(art.start_addr) + delta;
                if shifted < 0 || shifted as usize >= libspu2::PS1_RAM_SIZE {
                    return Err(Error::OutOfRange("relocated start_addr out of SPU RAM"));
                }
            }
            if art.loop_addr != 0 {
                let shifted = i64::from(art.loop_addr) + delta;
                if shifted < 0 || shifted as usize >= libspu2::PS1_RAM_SIZE {
                    return Err(Error::OutOfRange("relocated loop_addr out of SPU RAM"));
                }
            }
        }

        // Second pass: apply validated shifts atomically.
        for art in &mut self.articulations {
            if art.start_addr != 0 {
                art.start_addr = (i64::from(art.start_addr) + delta) as u32;
            }
            if art.loop_addr != 0 {
                art.loop_addr = (i64::from(art.loop_addr) + delta) as u32;
            }
        }

        self.spu_base_addr = new_spu_base;
        Ok(())
    }

    /// Uploads [`Self::adpcm_data`] into `device` at [`Self::spu_base_addr`].
    pub fn upload_to_spu(&self, device: &mut libspu2::Device) -> Result<()> {
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

fn parse_v1_raw_combined_view<'a>(bytes: &'a [u8]) -> Result<AkaoWaveBankView<'a>> {
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

    Ok(AkaoWaveBankView {
        version: AkaoVersion::V1,
        spu_base_addr,
        start_index: 0,
        art_slice: &bytes[0x10..adpcm_offset],
        art_count,
        adpcm_data: &bytes[adpcm_offset..adpcm_offset + adpcm_len],
    })
}

fn parse_unified_bank_view<'a>(
    bytes: &'a [u8],
    version: AkaoVersion,
) -> Result<AkaoWaveBankView<'a>> {
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

    let art_slice = &bytes[WAVE_BANK_HEADER_SIZE..adpcm_offset];
    if version == AkaoVersion::V3 {
        for chunk in art_slice.chunks_exact(16) {
            let mut arr = [0u8; 16];
            arr.copy_from_slice(chunk);
            let art = Articulation::from_16b(&arr);
            let start = art
                .start_addr
                .checked_add(spu_base_addr)
                .ok_or(Error::OutOfRange("V3 start_addr relocation overflow"))?;
            let loop_a = art
                .loop_addr
                .checked_add(spu_base_addr)
                .ok_or(Error::OutOfRange("V3 loop_addr relocation overflow"))?;
            if start as usize >= libspu2::PS1_RAM_SIZE || loop_a as usize >= libspu2::PS1_RAM_SIZE {
                return Err(Error::OutOfRange(
                    "V3 relocated address exceeds PS1 SPU RAM",
                ));
            }
        }
    }

    Ok(AkaoWaveBankView {
        version,
        spu_base_addr,
        start_index,
        art_slice,
        art_count: count,
        adpcm_data: &bytes[adpcm_offset..adpcm_offset + adpcm_len],
    })
}

fn validate_spu_range(spu_base_addr: u32, adpcm_len: usize) -> Result<()> {
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
