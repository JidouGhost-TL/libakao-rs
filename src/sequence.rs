// SPDX-License-Identifier: MIT OR Apache-2.0

//! AKAO sequence header, channel table, and V3 embedded drum/keysplit table parser.
//!
//! [`AkaoSequenceView`] provides zero-allocation borrowed parsing usable in pure
//! `#![no_std]` environments without `alloc`. With the `alloc` feature enabled,
//! `AkaoSequence` provides an owned representation.

#[cfg(feature = "alloc")]
use alloc::vec;
#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::version::{detect_version, AkaoVersion};
use crate::{Error, Result};

/// Four-byte ASCII magic signature at the start of every AKAO sequence (`"AKAO"`).
pub const AKAO_MAGIC: [u8; 4] = *b"AKAO";

/// An 8-byte AKAO `V3` embedded drum/percussion map entry (pointed to by header `+0x30`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DrumEntry {
    /// Index into the companion wave bank's 16-byte articulation table.
    pub articulation_index: u8,
    /// Fixed playback note / semitone index (`0..=127`).
    pub note: u8,
    /// ADSR1 register override (`0` preserves articulation default).
    pub adsr1: u16,
    /// ADSR2 register override (`0` preserves articulation default).
    pub adsr2: u16,
    /// Per-note volume scaling (`0..=127`).
    pub volume: u8,
    /// Per-note pan (`0..=127`, `64` = center).
    pub pan: u8,
}

impl DrumEntry {
    /// Decodes an 8-byte `V3` drum entry from a raw slice.
    #[inline]
    #[must_use]
    pub const fn from_bytes(chunk: &[u8; 8]) -> Self {
        Self {
            articulation_index: chunk[0],
            note: chunk[1],
            adsr1: u16::from_le_bytes([chunk[2], chunk[3]]),
            adsr2: u16::from_le_bytes([chunk[4], chunk[5]]),
            volume: chunk[6],
            pan: chunk[7],
        }
    }
}

/// An 8-byte AKAO `V3` embedded custom keysplit / articulation entry (pointed to by header `+0x34`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct KeysplitEntry {
    /// Index into the companion wave bank's 16-byte articulation table.
    pub articulation_index: u8,
    /// Lowest MIDI/AKAO note number covered by this split (inclusive).
    pub low_note: u8,
    /// Highest MIDI/AKAO note number covered by this split (inclusive).
    pub high_note: u8,
    /// Base/root note for pitch calculation.
    pub base_note: u8,
    /// ADSR1 register override (`0` preserves articulation default).
    pub adsr1: u16,
    /// ADSR2 register override (`0` preserves articulation default).
    pub adsr2: u16,
}

impl KeysplitEntry {
    /// Decodes an 8-byte `V3` keysplit entry from a raw slice.
    #[inline]
    #[must_use]
    pub const fn from_bytes(chunk: &[u8; 8]) -> Self {
        Self {
            articulation_index: chunk[0],
            low_note: chunk[1],
            high_note: chunk[2],
            base_note: chunk[3],
            adsr1: u16::from_le_bytes([chunk[4], chunk[5]]),
            adsr2: u16::from_le_bytes([chunk[6], chunk[7]]),
        }
    }
}

/// Zero-allocation borrowed view of a parsed AKAO music sequence header and tables.
///
/// Usable in pure `#![no_std]` builds without the `alloc` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkaoSequenceView<'a> {
    /// Sequence generation (`V1`, `V2a`, `V2b`, or `V3`).
    pub version: AkaoVersion,
    /// Song identifier from header `+0x04`.
    pub song_id: u16,
    /// Declared payload length from header `+0x06`.
    pub declared_length: u16,
    /// Reverb preset / mode identifier from header `+0x08`.
    pub reverb_type: u16,
    /// Initial sequence volume field (`V3` `+0x10`, `0` on `V1`/`V2`).
    pub initial_volume: u32,
    /// Additional sequence parameter field (`V3` `+0x14`, `0` on `V1`/`V2`).
    pub header_param: u32,
    /// Active channel bitmask (`24` bits on `V1`/`V2`, `32` bits on `V3`).
    pub channel_mask: u32,
    /// Rhythm / percussion channel bitmask (`V2` `+0x14`, `V3` `+0x24`, `0` on `V1`).
    pub rhythm_mask: u32,
    /// Secondary / alternate channel bitmask (`V2` `+0x18`, `V3` `+0x28`, `0` on `V1`).
    pub secondary_mask: u32,
    channel_starts: [Option<usize>; 32],
    drum_region: &'a [u8],
    drum_count: usize,
    keysplit_region: &'a [u8],
    keysplit_count: usize,
    data: &'a [u8],
}

impl<'a> AkaoSequenceView<'a> {
    /// Parses a borrowed AKAO sequence view from `bytes` without allocating.
    ///
    /// Pass `Some(version)` to force a specific [`AkaoVersion`], or `None` to
    /// auto-detect via [`detect_version`].
    pub fn parse(bytes: &'a [u8], version: Option<AkaoVersion>) -> Result<Self> {
        let version = match version {
            Some(v) => v,
            None => detect_version(bytes)?,
        };

        let hdr_size = version.header_size();
        if bytes.len() < hdr_size {
            return Err(Error::UnexpectedEof("AKAO sequence header"));
        }
        if bytes[0..4] != AKAO_MAGIC {
            return Err(Error::InvalidMagic);
        }

        let song_id = read_u16_le(bytes, 0x04);
        let declared_length = read_u16_le(bytes, 0x06);
        let reverb_type = read_u16_le(bytes, 0x08);

        let (initial_volume, header_param, channel_mask, rhythm_mask, secondary_mask) =
            match version {
                AkaoVersion::V1 => {
                    let ch_mask = read_u32_le(bytes, 0x10) & 0x00FF_FFFF;
                    (0, 0, ch_mask, 0, 0)
                }
                AkaoVersion::V2a | AkaoVersion::V2b => {
                    let ch_mask = read_u32_le(bytes, 0x10) & 0x00FF_FFFF;
                    let r_mask = read_u32_le(bytes, 0x14) & 0x00FF_FFFF;
                    let s_mask = read_u32_le(bytes, 0x18) & 0x00FF_FFFF;
                    (0, 0, ch_mask, r_mask, s_mask)
                }
                AkaoVersion::V3 => {
                    let vol = read_u32_le(bytes, 0x10);
                    let param = read_u32_le(bytes, 0x14);
                    let ch_mask = read_u32_le(bytes, 0x20);
                    let r_mask = read_u32_le(bytes, 0x24);
                    let s_mask = read_u32_le(bytes, 0x28);
                    (vol, param, ch_mask, r_mask, s_mask)
                }
            };

        let active_count = channel_mask.count_ones() as usize;
        if active_count == 0 {
            return Err(Error::InvalidHeader("channel mask has no active channels"));
        }

        let table_bytes = active_count
            .checked_mul(2)
            .ok_or(Error::OutOfRange("channel offset table size overflow"))?;
        let table_end = hdr_size
            .checked_add(table_bytes)
            .ok_or(Error::OutOfRange("channel offset table end overflow"))?;
        if table_end > bytes.len() {
            return Err(Error::UnexpectedEof("AKAO channel offset table"));
        }

        let max_ch = version.max_channels();
        let mut channel_starts = [None; 32];
        let mut entry_idx = 0usize;
        let mut min_channel_start = bytes.len();

        for (ch, slot) in channel_starts[..max_ch].iter_mut().enumerate() {
            if (channel_mask & (1u32 << ch)) == 0 {
                continue;
            }
            let entry_addr = hdr_size + entry_idx * 2;
            entry_idx += 1;
            let rel = read_u16_le(bytes, entry_addr) as usize;
            let abs = if version.uses_self_relative_offsets() {
                entry_addr
                    .checked_add(rel)
                    .ok_or(Error::OutOfRange("channel start offset overflow"))?
            } else {
                entry_addr
                    .checked_add(2)
                    .and_then(|base| base.checked_add(rel))
                    .ok_or(Error::OutOfRange("channel start offset overflow"))?
            };
            if abs < table_end || abs >= bytes.len() {
                return Err(Error::OutOfRange("channel start offset out of bounds"));
            }
            if abs < min_channel_start {
                min_channel_start = abs;
            }
            *slot = Some(abs);
        }

        let mut drum_region: &'a [u8] = &[];
        let mut drum_count = 0usize;
        let mut keysplit_region: &'a [u8] = &[];
        let mut keysplit_count = 0usize;

        if version == AkaoVersion::V3 {
            let drum_rel = read_i32_le(bytes, 0x30);
            let art_rel = read_i32_le(bytes, 0x34);

            let drum_addr = if drum_rel != 0 {
                Some(resolve_v3_rel_offset(
                    0x30,
                    drum_rel,
                    hdr_size,
                    bytes.len(),
                )?)
            } else {
                None
            };
            let art_addr = if art_rel != 0 {
                Some(resolve_v3_rel_offset(0x34, art_rel, hdr_size, bytes.len())?)
            } else {
                None
            };

            if let Some(start) = drum_addr {
                let end = table_region_end(start, art_addr, min_channel_start, bytes.len());
                drum_region = &bytes[start..end];
                drum_count = count_table_entries(drum_region);
            }
            if let Some(start) = art_addr {
                let end = table_region_end(start, drum_addr, min_channel_start, bytes.len());
                keysplit_region = &bytes[start..end];
                keysplit_count = count_table_entries(keysplit_region);
            }
        }

        Ok(Self {
            version,
            song_id,
            declared_length,
            reverb_type,
            initial_volume,
            header_param,
            channel_mask,
            rhythm_mask,
            secondary_mask,
            channel_starts,
            drum_region,
            drum_count,
            keysplit_region,
            keysplit_count,
            data: bytes,
        })
    }

    /// Returns the number of active channels in this sequence.
    #[inline]
    #[must_use]
    pub const fn active_channel_count(&self) -> usize {
        self.channel_mask.count_ones() as usize
    }

    /// Returns the absolute byte start offset for `channel`, or `None` if inactive/out of range.
    #[inline]
    #[must_use]
    pub const fn channel_start(&self, channel: usize) -> Option<usize> {
        if channel < self.version.max_channels() {
            self.channel_starts[channel]
        } else {
            None
        }
    }

    /// Returns the number of embedded `V3` drum entries.
    #[inline]
    #[must_use]
    pub const fn drum_count(&self) -> usize {
        self.drum_count
    }

    /// Decodes the embedded `V3` [`DrumEntry`] at `index`, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn drum_entry(&self, index: usize) -> Option<DrumEntry> {
        if index >= self.drum_count {
            return None;
        }
        let off = index * 8;
        let chunk: [u8; 8] = self.drum_region[off..off + 8].try_into().ok()?;
        Some(DrumEntry::from_bytes(&chunk))
    }

    /// Returns the number of embedded `V3` keysplit entries.
    #[inline]
    #[must_use]
    pub const fn keysplit_count(&self) -> usize {
        self.keysplit_count
    }

    /// Decodes the embedded `V3` [`KeysplitEntry`] at `index`, or `None` if out of bounds.
    #[inline]
    #[must_use]
    pub fn keysplit_entry(&self, index: usize) -> Option<KeysplitEntry> {
        if index >= self.keysplit_count {
            return None;
        }
        let off = index * 8;
        let chunk: [u8; 8] = self.keysplit_region[off..off + 8].try_into().ok()?;
        Some(KeysplitEntry::from_bytes(&chunk))
    }

    /// Returns the underlying raw sequence slice.
    #[inline]
    #[must_use]
    pub const fn data(&self) -> &'a [u8] {
        self.data
    }

    /// Converts this borrowed view into an owned [`AkaoSequence`].
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn to_owned_sequence(&self) -> AkaoSequence {
        let max_ch = self.version.max_channels();
        let mut channel_starts = vec![None; max_ch];
        channel_starts.copy_from_slice(&self.channel_starts[..max_ch]);

        let mut drum_table = Vec::with_capacity(self.drum_count);
        for i in 0..self.drum_count {
            if let Some(entry) = self.drum_entry(i) {
                drum_table.push(entry);
            }
        }

        let mut keysplit_table = Vec::with_capacity(self.keysplit_count);
        for i in 0..self.keysplit_count {
            if let Some(entry) = self.keysplit_entry(i) {
                keysplit_table.push(entry);
            }
        }

        AkaoSequence {
            version: self.version,
            song_id: self.song_id,
            declared_length: self.declared_length,
            reverb_type: self.reverb_type,
            initial_volume: self.initial_volume,
            header_param: self.header_param,
            channel_mask: self.channel_mask,
            rhythm_mask: self.rhythm_mask,
            secondary_mask: self.secondary_mask,
            channel_starts,
            drum_table,
            keysplit_table,
            data: self.data.to_vec(),
        }
    }
}

/// Owned parsed AKAO music sequence across all supported PlayStation 1 generations.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkaoSequence {
    /// Sequence generation (`V1`, `V2a`, `V2b`, or `V3`).
    pub version: AkaoVersion,
    /// Song identifier from header `+0x04`.
    pub song_id: u16,
    /// Declared payload length from header `+0x06`.
    pub declared_length: u16,
    /// Reverb preset / mode identifier from header `+0x08`.
    pub reverb_type: u16,
    /// Initial sequence volume field (`V3` `+0x10`, `0` on `V1`/`V2`).
    pub initial_volume: u32,
    /// Additional sequence parameter field (`V3` `+0x14`, `0` on `V1`/`V2`).
    pub header_param: u32,
    /// Active channel bitmask (`24` bits on `V1`/`V2`, `32` bits on `V3`).
    pub channel_mask: u32,
    /// Rhythm / percussion channel bitmask (`V2` `+0x14`, `V3` `+0x24`, `0` on `V1`).
    pub rhythm_mask: u32,
    /// Secondary / alternate channel bitmask (`V2` `+0x18`, `V3` `+0x28`, `0` on `V1`).
    pub secondary_mask: u32,
    /// Absolute byte offset in [`Self::data`] for each channel slot (`None` if inactive).
    pub channel_starts: Vec<Option<usize>>,
    /// Embedded `V3` 8-byte drum map entries (empty if absent or `V1`/`V2`).
    pub drum_table: Vec<DrumEntry>,
    /// Embedded `V3` 8-byte keysplit entries (empty if absent or `V1`/`V2`).
    pub keysplit_table: Vec<KeysplitEntry>,
    /// Raw sequence bytes.
    pub data: Vec<u8>,
}

#[cfg(feature = "alloc")]
impl AkaoSequence {
    /// Parses an AKAO sequence from `bytes`.
    ///
    /// Pass `Some(version)` to force a specific [`AkaoVersion`], or `None` to
    /// auto-detect via [`detect_version`].
    pub fn parse(bytes: &[u8], version: Option<AkaoVersion>) -> Result<Self> {
        let view = AkaoSequenceView::parse(bytes, version)?;
        Ok(view.to_owned_sequence())
    }

    /// Wraps a raw single-channel AKAO SFX bytecode slice into a playable 1-channel [`AkaoSequence`].
    ///
    /// Relative jumps inside `slot_bytes` resolve directly against offset `0`.
    pub fn from_sfx_slot(slot_bytes: &[u8], version: AkaoVersion) -> Result<Self> {
        if slot_bytes.is_empty() {
            return Err(Error::UnexpectedEof("AKAO SFX slot bytecode"));
        }
        let mut channel_starts = vec![None; version.max_channels()];
        channel_starts[0] = Some(0);
        Ok(Self {
            version,
            song_id: 0,
            declared_length: 0,
            reverb_type: 0,
            initial_volume: 0,
            header_param: 0,
            channel_mask: 1,
            rhythm_mask: 0,
            secondary_mask: 0,
            channel_starts,
            drum_table: Vec::new(),
            keysplit_table: Vec::new(),
            data: slot_bytes.to_vec(),
        })
    }

    /// Returns the number of active channels in this sequence.
    #[inline]
    #[must_use]
    pub fn active_channel_count(&self) -> usize {
        self.channel_mask.count_ones() as usize
    }

    /// Returns the absolute byte start offset for `channel`, or `None` if inactive/out of range.
    #[inline]
    #[must_use]
    pub fn channel_start(&self, channel: usize) -> Option<usize> {
        self.channel_starts.get(channel).copied().flatten()
    }
}

fn resolve_v3_rel_offset(
    field_offset: usize,
    rel: i32,
    min_addr: usize,
    len: usize,
) -> Result<usize> {
    let target = (field_offset as i64) + i64::from(rel);
    if target < min_addr as i64 || target as usize >= len {
        return Err(Error::OutOfRange("V3 embedded table offset out of bounds"));
    }
    Ok(target as usize)
}

fn table_region_end(
    start: usize,
    other_table: Option<usize>,
    min_channel_start: usize,
    total_len: usize,
) -> usize {
    let mut end = total_len;
    if let Some(other) = other_table {
        if other > start && other < end {
            end = other;
        }
    }
    if min_channel_start > start && min_channel_start < end {
        end = min_channel_start;
    }
    end
}

fn count_table_entries(region: &[u8]) -> usize {
    let mut count = 0usize;
    for chunk in region.chunks_exact(8) {
        if chunk == [0xFF; 8] || (chunk == [0x00; 8] && count > 0) {
            break;
        }
        count += 1;
    }
    count
}

#[inline]
fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
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

#[inline]
fn read_i32_le(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}
