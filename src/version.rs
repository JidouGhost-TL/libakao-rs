// SPDX-License-Identifier: MIT

//! AKAO sequencer generation metadata, hardware timer rates, and version auto-detection.

use crate::sequence::AKAO_MAGIC;
use crate::Error;

/// PlayStation 1 Root Counter 2 input clock (`SysClk / 8` = `4,233,600 Hz`).
pub const PS1_ROOT_COUNTER_2_CLOCK_HZ: f64 = 4_233_600.0;

/// Root Counter 2 reload value used by AKAO `V1` (`0x43D1` = `17,361`).
pub const V1_TIMER_DIVIDER: u32 = 17_361;

/// Root Counter 2 reload value used by AKAO `V2a`, `V2b`, and `V3` (`0x44E8` = `17,640`).
pub const V2_V3_TIMER_DIVIDER: u32 = 17_640;

/// Supported PlayStation 1 AKAO sequencer generations.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AkaoVersion {
    /// Generation 1 (`0x14` header, 24 channels, post-entry offsets, `243.8569 Hz` timer, 64 B bank entries, `0xFC` extended prefix).
    V1,
    /// Generation 2a (`0x20` header, 24 channels, post-entry offsets, `240.0000 Hz` timer, 64 B bank entries up to 128, `0xFC` extended prefix).
    V2a,
    /// Generation 2b (`0x20` header, 24 channels, post-entry offsets, `240.0000 Hz` timer, 64 B bank entries up to 256 with relocation, `0xFC` extended prefix).
    V2b,
    /// Generation 3 (`0x40` header, 32 channels, self-relative offsets, `240.0000 Hz` timer, 16 B bank entries, `0xF0..0xFD` explicit durations, `0xFE` extended prefix with subroutines).
    V3,
}

impl AkaoVersion {
    /// Returns the PS1 Root Counter 2 target divider for this generation.
    #[inline]
    #[must_use]
    pub const fn timer_divider(self) -> u32 {
        match self {
            Self::V1 => V1_TIMER_DIVIDER,
            Self::V2a | Self::V2b | Self::V3 => V2_V3_TIMER_DIVIDER,
        }
    }

    /// Returns the hardware sequencer timer frequency in Hz.
    ///
    /// - `V1`: `4,233,600 / 17,361 ≈ 243.8569 Hz`
    /// - `V2a`, `V2b`, `V3`: `4,233,600 / 17,640 = 240.0000 Hz`
    #[inline]
    #[must_use]
    pub const fn timer_hz(self) -> f64 {
        PS1_ROOT_COUNTER_2_CLOCK_HZ / (self.timer_divider() as f64)
    }

    /// Returns the sequence header size in bytes before the channel-start offset table.
    #[inline]
    #[must_use]
    pub const fn header_size(self) -> usize {
        match self {
            Self::V1 => 0x14,
            Self::V2a | Self::V2b => 0x20,
            Self::V3 => 0x40,
        }
    }

    /// Returns the maximum number of sequence channels supported by this generation.
    #[inline]
    #[must_use]
    pub const fn max_channels(self) -> usize {
        match self {
            Self::V1 | Self::V2a | Self::V2b => 24,
            Self::V3 => 32,
        }
    }

    /// Returns the byte offset of the primary `u32` channel mask in the sequence header.
    #[inline]
    #[must_use]
    pub const fn channel_mask_offset(self) -> usize {
        match self {
            Self::V1 | Self::V2a | Self::V2b => 0x10,
            Self::V3 => 0x20,
        }
    }

    /// Returns the valid bitmask for channel masks in this generation.
    #[inline]
    #[must_use]
    pub const fn channel_mask_bits(self) -> u32 {
        match self {
            Self::V1 | Self::V2a | Self::V2b => 0x00FF_FFFF,
            Self::V3 => 0xFFFF_FFFF,
        }
    }

    /// Returns `true` if channel-start `u16` offsets are self-relative (`entry_addr + val`, `V3`),
    /// or `false` if post-entry relative (`entry_addr + 2 + val`, `V1`/`V2a`/`V2b`).
    #[inline]
    #[must_use]
    pub const fn uses_self_relative_offsets(self) -> bool {
        matches!(self, Self::V3)
    }

    /// Returns the byte stride of each articulation entry in a companion wave bank.
    #[inline]
    #[must_use]
    pub const fn articulation_stride(self) -> usize {
        match self {
            Self::V1 | Self::V2a | Self::V2b => 64,
            Self::V3 => 16,
        }
    }

    /// Returns the maximum number of wave/articulation entries supported by this generation.
    #[inline]
    #[must_use]
    pub const fn max_wave_entries(self) -> usize {
        match self {
            Self::V1 | Self::V2a => 128,
            Self::V2b | Self::V3 => 256,
        }
    }

    /// Returns the extended-opcode prefix byte (`0xFC` for `V1`/`V2`, `0xFE` for `V3`).
    #[inline]
    #[must_use]
    pub const fn extended_prefix(self) -> u8 {
        match self {
            Self::V1 | Self::V2a | Self::V2b => 0xFC,
            Self::V3 => 0xFE,
        }
    }

    /// Returns `true` if `0xF0..=0xFD` explicit-duration note/tie/rest opcodes are supported (`V3`).
    #[inline]
    #[must_use]
    pub const fn has_explicit_duration_opcodes(self) -> bool {
        matches!(self, Self::V3)
    }

    /// Returns `true` if extended subroutine call (`0xFE 0x0E`) and return (`0xFE 0x0F`) are supported (`V3`).
    #[inline]
    #[must_use]
    pub const fn supports_subroutines(self) -> bool {
        matches!(self, Self::V3)
    }
}

/// Attempts to auto-detect the [`AkaoVersion`] of a raw AKAO sequence buffer.
///
/// Because `V2a` and `V2b` share an identical sequence header geometry (`0x20` header,
/// 24 channels, post-entry offsets, `0xFC` prefix), `detect_version` reports [`AkaoVersion::V2a`]
/// for generation-2 sequences unless the caller explicitly specifies [`AkaoVersion::V2b`].
pub fn detect_version(bytes: &[u8]) -> Result<AkaoVersion, Error> {
    if bytes.len() < 0x14 {
        return Err(Error::UnexpectedEof("AKAO sequence header"));
    }
    if bytes[0..4] != AKAO_MAGIC {
        return Err(Error::InvalidMagic);
    }

    // Score each candidate version by structural validity of its header and channel table.
    let v3_ok = probe_sequence_layout(bytes, AkaoVersion::V3);
    let v2_ok = probe_sequence_layout(bytes, AkaoVersion::V2a);
    let v1_ok = probe_sequence_layout(bytes, AkaoVersion::V1);

    // Prefer V3 when its 0x40 header + self-relative table validates and either:
    // - V3-only fields (+0x20..+0x38) are populated, or
    // - V1/V2 do not validate, or
    // - bytes at +0x10..+0x20 don't look like a V1/V2 channel mask.
    if v3_ok && (!v2_ok && !v1_ok) {
        return Ok(AkaoVersion::V3);
    }
    if v3_ok {
        let v3_ch_mask = read_u32_le(bytes, 0x20);
        let v3_drum = read_i32_le(bytes, 0x30);
        let v3_art = read_i32_le(bytes, 0x34);
        let v1_ch_mask = read_u32_le(bytes, 0x10) & 0x00FF_FFFF;
        // If V3 has upper channels (>23), embedded drum/art tables, or V1/V2 mask is 0, it's V3.
        if (v3_ch_mask & 0xFF00_0000) != 0 || v3_drum != 0 || v3_art != 0 || v1_ch_mask == 0 {
            return Ok(AkaoVersion::V3);
        }
        // Check if V3 channel 0 starts right at or immediately after the V3 table end.
        let v3_table_end = 0x40 + (v3_ch_mask.count_ones() as usize) * 2;
        if let Some(first_v3) = first_channel_start(bytes, AkaoVersion::V3) {
            if first_v3 == v3_table_end {
                return Ok(AkaoVersion::V3);
            }
        }
    }

    if v2_ok && !v1_ok {
        return Ok(AkaoVersion::V2a);
    }
    if v2_ok {
        let ch_mask = read_u32_le(bytes, 0x10) & 0x00FF_FFFF;
        let rhythm_mask = read_u32_le(bytes, 0x14) & 0x00FF_FFFF;
        let sec_mask = read_u32_le(bytes, 0x18) & 0x00FF_FFFF;
        let v2_table_end = 0x20 + (ch_mask.count_ones() as usize) * 2;
        if let Some(first_v2) = first_channel_start(bytes, AkaoVersion::V2a) {
            // In V2, the first channel almost always begins immediately after the u16 table,
            // and bytes 0x14..0x20 are mask fields (often 0 or subsets of channels) rather than u16 offsets.
            if first_v2 == v2_table_end
                || ((rhythm_mask != 0 || sec_mask != 0) && first_v2 >= v2_table_end)
            {
                return Ok(AkaoVersion::V2a);
            }
        }
    }

    if v1_ok {
        return Ok(AkaoVersion::V1);
    }
    if v2_ok {
        return Ok(AkaoVersion::V2a);
    }
    if v3_ok {
        return Ok(AkaoVersion::V3);
    }

    Err(Error::UnknownVersion)
}

fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

fn read_u32_le(bytes: &[u8], offset: usize) -> u32 {
    u32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn read_i32_le(bytes: &[u8], offset: usize) -> i32 {
    i32::from_le_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn first_channel_start(bytes: &[u8], version: AkaoVersion) -> Option<usize> {
    let hdr = version.header_size();
    if bytes.len() < hdr + 2 {
        return None;
    }
    let rel = read_u16_le(bytes, hdr) as usize;
    let abs = if version.uses_self_relative_offsets() {
        hdr.checked_add(rel)?
    } else {
        hdr.checked_add(2)?.checked_add(rel)?
    };
    Some(abs)
}

fn probe_sequence_layout(bytes: &[u8], version: AkaoVersion) -> bool {
    let hdr = version.header_size();
    if bytes.len() < hdr {
        return false;
    }
    let ch_mask = read_u32_le(bytes, version.channel_mask_offset()) & version.channel_mask_bits();
    let count = ch_mask.count_ones() as usize;
    if count == 0 {
        return false;
    }
    let table_end = match hdr.checked_add(count * 2) {
        Some(end) if end < bytes.len() => end,
        _ => return false,
    };

    for idx in 0..count {
        let entry_addr = hdr + idx * 2;
        let rel = read_u16_le(bytes, entry_addr) as usize;
        let abs = if version.uses_self_relative_offsets() {
            entry_addr.saturating_add(rel)
        } else {
            entry_addr.saturating_add(2).saturating_add(rel)
        };
        if abs < table_end || abs >= bytes.len() {
            return false;
        }
    }

    if version == AkaoVersion::V3 {
        let drum_rel = read_i32_le(bytes, 0x30);
        if drum_rel != 0 {
            let target = 0x30i64 + i64::from(drum_rel);
            if target < hdr as i64 || target as usize >= bytes.len() {
                return false;
            }
        }
        let art_rel = read_i32_le(bytes, 0x34);
        if art_rel != 0 {
            let target = 0x34i64 + i64::from(art_rel);
            if target < hdr as i64 || target as usize >= bytes.len() {
                return false;
            }
        }
    }

    true
}
