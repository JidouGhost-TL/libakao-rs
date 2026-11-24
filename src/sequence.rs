// SPDX-License-Identifier: MIT

//! AKAO sequence header, channel table, and V3 embedded drum/keysplit table parser.

use alloc::vec;
use alloc::vec::Vec;

use crate::version::{detect_version, AkaoVersion};
use crate::Error;

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

/// Parsed AKAO music sequence across all supported PlayStation 1 generations.
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

impl AkaoSequence {
    /// Parses an AKAO sequence from `bytes`.
    ///
    /// Pass `Some(version)` to force a specific [`AkaoVersion`], or `None` to
    /// auto-detect via [`detect_version`].
    pub fn parse(bytes: &[u8], version: Option<AkaoVersion>) -> Result<Self, Error> {
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
        let mut channel_starts = vec![None; max_ch];
        let mut entry_idx = 0usize;
        let mut min_channel_start = bytes.len();

        for (ch, slot) in channel_starts.iter_mut().enumerate() {
            if (channel_mask & (1u32 << ch)) == 0 {
                continue;
            }
            let entry_addr = hdr_size + entry_idx * 2;
            entry_idx += 1;
            let rel = read_u16_le(bytes, entry_addr) as usize;
            let abs = if version.uses_self_relative_offsets() {
                // AKAO V3: self-relative (`0x40 + idx * 2 + val`).
                entry_addr
                    .checked_add(rel)
                    .ok_or(Error::OutOfRange("channel start offset overflow"))?
            } else {
                // AKAO V1/V2a/V2b: post-entry relative (`entry_addr + 2 + val`).
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

        let mut drum_table = Vec::new();
        let mut keysplit_table = Vec::new();

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
                parse_drum_table(&bytes[start..end], &mut drum_table);
            }
            if let Some(start) = art_addr {
                let end = table_region_end(start, drum_addr, min_channel_start, bytes.len());
                parse_keysplit_table(&bytes[start..end], &mut keysplit_table);
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
            drum_table,
            keysplit_table,
            data: bytes.to_vec(),
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
) -> Result<usize, Error> {
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

fn parse_drum_table(region: &[u8], out: &mut Vec<DrumEntry>) {
    for chunk in region.chunks_exact(8) {
        // Stop at an all-0xFF or all-zero terminator if encountered after entries.
        if chunk == [0xFF; 8] || (chunk == [0x00; 8] && !out.is_empty()) {
            break;
        }
        out.push(DrumEntry {
            articulation_index: chunk[0],
            note: chunk[1],
            adsr1: u16::from_le_bytes([chunk[2], chunk[3]]),
            adsr2: u16::from_le_bytes([chunk[4], chunk[5]]),
            volume: chunk[6],
            pan: chunk[7],
        });
    }
}

fn parse_keysplit_table(region: &[u8], out: &mut Vec<KeysplitEntry>) {
    for chunk in region.chunks_exact(8) {
        if chunk == [0xFF; 8] || (chunk == [0x00; 8] && !out.is_empty()) {
            break;
        }
        out.push(KeysplitEntry {
            articulation_index: chunk[0],
            low_note: chunk[1],
            high_note: chunk[2],
            base_note: chunk[3],
            adsr1: u16::from_le_bytes([chunk[4], chunk[5]]),
            adsr2: u16::from_le_bytes([chunk[6], chunk[7]]),
        });
    }
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
