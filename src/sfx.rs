// SPDX-License-Identifier: MIT OR Apache-2.0

//! AKAO sound-effect (SFX) offset-table bank parser.
//!
//! Layout (all little-endian):
//! - `0x00..0x04`: `"AKAO"` magic
//! - `0x04..0x10`: 12 zero header bytes (including `declared_length == 0` at `+0x06`)
//! - `0x10..0x10 + slot_count * 2`: `u16` offset table where `0xFFFF` marks an unused slot
//!   and active offsets start at `0` and strictly increase
//! - Optional `slot_count`-byte per-slot attribute/channel-flags table immediately after
//!   the offset table (zero for unused slots)
//! - Contiguous AKAO SFX bytecode payload region starting at `data_base`, indexed by each
//!   active slot's `u16` offset.
//!
//! [`AkaoSfxBankView`] provides zero-allocation borrowed parsing usable in pure
//! `#![no_std]` environments without `alloc`. With the `alloc` feature enabled,
//! `AkaoSfxBank` provides an owned representation.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

#[cfg(feature = "alloc")]
use crate::sequence::AkaoSequence;
use crate::sequence::AKAO_MAGIC;
#[cfg(feature = "alloc")]
use crate::version::AkaoVersion;
use crate::{Error, Result};

/// Fixed header size in bytes before the `u16` SFX offset table.
pub const SFX_BANK_HEADER_SIZE: usize = 0x10;

/// Sentinel value marking an unused slot in the `u16` SFX offset table.
pub const SFX_UNUSED_SLOT: u16 = 0xFFFF;

/// Maximum number of `u16` slots scanned in an AKAO SFX offset table.
const MAX_SFX_SLOTS: usize = 4096;

/// Zero-allocation borrowed view of a parsed AKAO sound-effect offset-table bank.
///
/// Usable in pure `#![no_std]` builds without the `alloc` feature.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct AkaoSfxBankView<'a> {
    offset_table: &'a [u8],
    slot_count: usize,
    active_count: usize,
    attributes: Option<&'a [u8]>,
    payload: &'a [u8],
    data_base: usize,
}

impl<'a> AkaoSfxBankView<'a> {
    /// Parses a borrowed AKAO SFX offset-table bank from `bytes` without allocating.
    pub fn parse(bytes: &'a [u8]) -> Result<Self> {
        if bytes.len() < SFX_BANK_HEADER_SIZE + 3 {
            return Err(Error::UnexpectedEof("AKAO SFX bank header"));
        }
        if bytes[0..4] != AKAO_MAGIC {
            return Err(Error::InvalidMagic);
        }
        if bytes[4..SFX_BANK_HEADER_SIZE].iter().any(|&b| b != 0) {
            return Err(Error::InvalidHeader(
                "AKAO SFX bank header bytes +0x04..+0x10 must be zero",
            ));
        }

        // Pass 1: find the longest prefix of `u16` values that could form a valid offset table.
        let max_possible_slots = ((bytes.len() - SFX_BANK_HEADER_SIZE) / 2).min(MAX_SFX_SLOTS);
        let mut max_scan = 0usize;
        let mut prev_off: Option<u16> = None;
        let mut first_zero_idx: Option<usize> = None;

        for idx in 0..max_possible_slots {
            let pos = SFX_BANK_HEADER_SIZE + idx * 2;
            let val = read_u16_le(bytes, pos);
            if val == SFX_UNUSED_SLOT {
                max_scan = idx + 1;
                continue;
            }
            let remaining_after_entry = bytes.len() - (pos + 2);
            if usize::from(val) >= remaining_after_entry {
                break;
            }
            if let Some(prev) = prev_off {
                if val <= prev {
                    break;
                }
            } else if val == 0 {
                first_zero_idx = Some(idx);
            } else {
                return Err(Error::InvalidHeader(
                    "first active AKAO SFX slot offset must be 0",
                ));
            }
            prev_off = Some(val);
            max_scan = idx + 1;
        }

        let Some(first_zero) = first_zero_idx else {
            return Err(Error::InvalidHeader(
                "AKAO SFX bank has no valid active slots",
            ));
        };

        // Pass 2: score candidate `(slot_count, has_attrs)` boundaries so non-zero attribute bytes
        // on large payloads (>259 B) or leading payload bytes cannot be over-scanned as offset
        // entries, and plain note-opcode streams (`0x00..=0x1F`) are not misclassified as
        // attribute tables.
        struct SfxCandidate {
            score: (usize, usize, u8, u8, u8, usize),
            slot_count: usize,
            active_count: usize,
            has_attrs: bool,
            data_base: usize,
        }
        let mut best: Option<SfxCandidate> = None;

        for cand_slots in (first_zero + 1)..=max_scan {
            let Some(table_bytes) = cand_slots.checked_mul(2) else {
                break;
            };
            let Some(table_end) = SFX_BANK_HEADER_SIZE.checked_add(table_bytes) else {
                break;
            };
            if table_end >= bytes.len() {
                break;
            }

            let mut active_count = 0usize;
            let mut last_off = 0usize;
            for idx in 0..cand_slots {
                let val = read_u16_le(bytes, SFX_BANK_HEADER_SIZE + idx * 2);
                if val != SFX_UNUSED_SLOT {
                    active_count += 1;
                    last_off = usize::from(val);
                }
            }
            if active_count == 0 {
                continue;
            }

            for has_attrs in [false, true] {
                let data_base = if has_attrs {
                    let Some(end) = table_end.checked_add(cand_slots) else {
                        continue;
                    };
                    if end >= bytes.len() {
                        continue;
                    }
                    let attrs = &bytes[table_end..end];
                    // Attribute bytes must be small flags (`<= 0x1F`) and zero for unused slots.
                    if !attrs.iter().enumerate().all(|(i, &b)| {
                        if b > 0x1F {
                            return false;
                        }
                        let off = read_u16_le(bytes, SFX_BANK_HEADER_SIZE + i * 2);
                        off != SFX_UNUSED_SLOT || b == 0
                    }) {
                        continue;
                    }
                    end
                } else {
                    table_end
                };

                let payload = &bytes[data_base..];
                if last_off >= payload.len() {
                    continue;
                }

                let mut interior_total = 0usize;
                let mut interior_ok = 0usize;
                let mut prev_active: Option<usize> = None;
                for idx in 0..cand_slots {
                    let val = read_u16_le(bytes, SFX_BANK_HEADER_SIZE + idx * 2);
                    if val == SFX_UNUSED_SLOT {
                        continue;
                    }
                    let off = usize::from(val);
                    if let Some(prev) = prev_active {
                        if off > prev {
                            interior_total += 1;
                            if is_sfx_terminator(payload[off - 1]) {
                                interior_ok += 1;
                            }
                        }
                    }
                    prev_active = Some(off);
                }

                let final_ok = is_sfx_terminator(*payload.last().unwrap_or(&0));
                let starts_with_ctrl = matches!(payload[0], 0xA1..=0xFE);
                let interior_ratio = (interior_ok * 1024)
                    .checked_div(interior_total)
                    .unwrap_or(if final_ok { 512 } else { 0 });
                let score = (
                    interior_ratio,
                    interior_ok,
                    u8::from(final_ok),
                    u8::from(starts_with_ctrl),
                    u8::from(!has_attrs),
                    cand_slots,
                );
                if best.as_ref().map_or(true, |b| score > b.score) {
                    best = Some(SfxCandidate {
                        score,
                        slot_count: cand_slots,
                        active_count,
                        has_attrs,
                        data_base,
                    });
                }
            }
        }

        let Some(SfxCandidate {
            slot_count,
            active_count,
            has_attrs,
            data_base,
            ..
        }) = best
        else {
            return Err(Error::UnexpectedEof("AKAO SFX bank payload"));
        };
        let table_end = SFX_BANK_HEADER_SIZE + slot_count * 2;
        let attributes = if has_attrs {
            Some(&bytes[table_end..data_base])
        } else {
            None
        };
        let payload = &bytes[data_base..];

        Ok(Self {
            offset_table: &bytes[SFX_BANK_HEADER_SIZE..table_end],
            slot_count,
            active_count,
            attributes,
            payload,
            data_base,
        })
    }

    /// Returns `true` if `bytes` is a structurally valid AKAO SFX offset-table bank.
    #[inline]
    #[must_use]
    pub fn identifies(bytes: &[u8]) -> bool {
        AkaoSfxBankView::parse(bytes).is_ok()
    }

    /// Returns the total number of slots (active + unused) in the offset table.
    #[inline]
    #[must_use]
    pub const fn slot_count(&self) -> usize {
        self.slot_count
    }

    /// Returns the number of active (non-`0xFFFF`) SFX slots.
    #[inline]
    #[must_use]
    pub const fn active_count(&self) -> usize {
        self.active_count
    }

    /// Returns the file offset where the SFX bytecode payload begins.
    #[inline]
    #[must_use]
    pub const fn data_base(&self) -> usize {
        self.data_base
    }

    /// Returns the raw payload slice containing all SFX bytecode sequences.
    #[inline]
    #[must_use]
    pub const fn payload(&self) -> &'a [u8] {
        self.payload
    }

    /// Returns the raw payload-relative offset for `slot`, or `None` if unused or out of range.
    #[inline]
    #[must_use]
    pub fn slot_offset(&self, slot: usize) -> Option<usize> {
        if slot >= self.slot_count {
            return None;
        }
        let val = read_u16_le(self.offset_table, slot * 2);
        if val == SFX_UNUSED_SLOT {
            None
        } else {
            Some(usize::from(val))
        }
    }

    /// Returns the optional per-slot attribute/channel-flags byte for `slot`.
    #[inline]
    #[must_use]
    pub fn slot_attribute(&self, slot: usize) -> Option<u8> {
        self.attributes.and_then(|attrs| attrs.get(slot).copied())
    }

    /// Returns the bytecode slice for `slot` (up to the next active slot or end of payload),
    /// or `None` if `slot` is unused or out of range.
    #[must_use]
    pub fn slot_bytes(&self, slot: usize) -> Option<&'a [u8]> {
        let start = self.slot_offset(slot)?;
        let mut end = self.payload.len();
        for next in (slot + 1)..self.slot_count {
            if let Some(next_off) = self.slot_offset(next) {
                end = next_off;
                break;
            }
        }
        self.payload.get(start..end)
    }

    /// Converts this borrowed view into an owned [`AkaoSfxBank`].
    #[cfg(feature = "alloc")]
    #[must_use]
    pub fn to_owned_sfx_bank(&self) -> AkaoSfxBank {
        let mut slot_offsets = Vec::with_capacity(self.slot_count);
        for idx in 0..self.slot_count {
            slot_offsets.push(self.slot_offset(idx));
        }
        AkaoSfxBank {
            slot_offsets,
            active_count: self.active_count,
            attributes: self.attributes.map(|a| a.to_vec()),
            payload: self.payload.to_vec(),
            data_base: self.data_base,
        }
    }
}

/// Owned parsed AKAO sound-effect (SFX) offset-table bank.
#[cfg(feature = "alloc")]
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AkaoSfxBank {
    /// Payload-relative byte offset for each slot (`None` if unused).
    pub slot_offsets: Vec<Option<usize>>,
    /// Number of active slots.
    pub active_count: usize,
    /// Optional per-slot attribute/channel-flags table.
    pub attributes: Option<Vec<u8>>,
    /// Contiguous SFX bytecode payload.
    pub payload: Vec<u8>,
    /// Original file offset where [`Self::payload`] begins.
    pub data_base: usize,
}

#[cfg(feature = "alloc")]
impl AkaoSfxBank {
    /// Parses an owned AKAO SFX offset-table bank from `bytes`.
    pub fn parse(bytes: &[u8]) -> Result<Self> {
        let view = AkaoSfxBankView::parse(bytes)?;
        Ok(view.to_owned_sfx_bank())
    }

    /// Returns the total number of slots (active + unused) in the offset table.
    #[inline]
    #[must_use]
    pub fn slot_count(&self) -> usize {
        self.slot_offsets.len()
    }

    /// Returns the raw payload-relative offset for `slot`, or `None` if unused or out of range.
    #[inline]
    #[must_use]
    pub fn slot_offset(&self, slot: usize) -> Option<usize> {
        self.slot_offsets.get(slot).copied().flatten()
    }

    /// Returns the optional per-slot attribute/channel-flags byte for `slot`.
    #[inline]
    #[must_use]
    pub fn slot_attribute(&self, slot: usize) -> Option<u8> {
        self.attributes
            .as_deref()
            .and_then(|attrs| attrs.get(slot).copied())
    }

    /// Returns the bytecode slice for `slot`, or `None` if unused or out of range.
    #[must_use]
    pub fn slot_bytes(&self, slot: usize) -> Option<&[u8]> {
        let start = self.slot_offsets.get(slot).copied().flatten()?;
        let mut end = self.payload.len();
        if let Some(&next_off) = self.slot_offsets[slot + 1..].iter().flatten().next() {
            end = next_off;
        }
        self.payload.get(start..end)
    }

    /// Wraps the bytecode of `slot` into a playable 1-channel [`AkaoSequence`].
    pub fn to_sequence(&self, slot: usize, version: AkaoVersion) -> Result<AkaoSequence> {
        let bytes = self
            .slot_bytes(slot)
            .ok_or(Error::OutOfRange("AKAO SFX slot unused or out of range"))?;
        AkaoSequence::from_sfx_slot(bytes, version)
    }
}

#[inline]
fn is_sfx_terminator(byte: u8) -> bool {
    matches!(byte, 0xA0 | 0xCA | 0xFF)
}

#[inline]
fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_std_sfx_view_parses_tiny_single_slot() {
        let mut buf = [0u8; 0x10 + 2 + 2];
        buf[0..4].copy_from_slice(b"AKAO");
        buf[0x10..0x12].copy_from_slice(&0u16.to_le_bytes());
        buf[0x12..0x14].copy_from_slice(&[0x02, 0xA0]);
        let view = AkaoSfxBankView::parse(&buf).unwrap();
        assert_eq!(view.slot_count(), 1);
        assert_eq!(view.active_count(), 1);
        assert_eq!(view.slot_attribute(0), None);
        assert_eq!(view.slot_bytes(0), Some(&[0x02, 0xA0][..]));
    }
}
