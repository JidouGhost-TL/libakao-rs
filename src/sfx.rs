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
//! [`AkaoSfxBank`] provides an owned representation.

#[cfg(feature = "alloc")]
use alloc::vec::Vec;

use crate::sequence::AKAO_MAGIC;
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
        if bytes.len() < SFX_BANK_HEADER_SIZE + 8 {
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

        let max_possible_slots = ((bytes.len() - SFX_BANK_HEADER_SIZE) / 2).min(MAX_SFX_SLOTS);
        let mut slot_count = 0usize;
        let mut active_count = 0usize;
        let mut prev_off: Option<u16> = None;
        let mut saw_zero = false;

        for idx in 0..max_possible_slots {
            let pos = SFX_BANK_HEADER_SIZE + idx * 2;
            let val = read_u16_le(bytes, pos);
            if val == SFX_UNUSED_SLOT {
                slot_count = idx + 1;
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
                saw_zero = true;
            } else {
                return Err(Error::InvalidHeader(
                    "first active AKAO SFX slot offset must be 0",
                ));
            }
            prev_off = Some(val);
            active_count += 1;
            slot_count = idx + 1;
        }

        if !saw_zero || active_count == 0 {
            return Err(Error::InvalidHeader(
                "AKAO SFX bank has no valid active slots",
            ));
        }
        let last_off = usize::from(prev_off.unwrap_or(0));

        let table_bytes = slot_count
            .checked_mul(2)
            .ok_or(Error::OutOfRange("SFX offset table size overflow"))?;
        let table_end = SFX_BANK_HEADER_SIZE
            .checked_add(table_bytes)
            .ok_or(Error::OutOfRange("SFX offset table end overflow"))?;

        // Detect optional `slot_count`-byte attribute/flag table immediately after the offset table:
        // attribute bytes are small per-slot flags (`<= 0x1F`), whereas AKAO SFX bytecode streams
        // begin with channel control opcodes (`>= 0xA0`) or note commands.
        let attr_end = table_end.checked_add(slot_count);
        let has_attrs = attr_end.is_some_and(|end| {
            if end >= bytes.len() {
                return false;
            }
            let payload_len = bytes.len() - end;
            if last_off >= payload_len {
                return false;
            }
            let attrs = &bytes[table_end..end];
            attrs.iter().all(|&b| b <= 0x1F)
        });

        let (attributes, data_base) = if has_attrs {
            let end = table_end + slot_count;
            (Some(&bytes[table_end..end]), end)
        } else {
            (None, table_end)
        };

        if data_base >= bytes.len() {
            return Err(Error::UnexpectedEof("AKAO SFX bank payload"));
        }
        let payload = &bytes[data_base..];
        if last_off >= payload.len() {
            return Err(Error::OutOfRange("AKAO SFX slot offset exceeds payload"));
        }

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
    pub fn identifies(bytes: &'a [u8]) -> bool {
        Self::parse(bytes).is_ok()
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

    /// Returns the bytecode slice for `slot`, or `None` if unused or out of range.
    #[must_use]
    pub fn slot_bytes(&self, slot: usize) -> Option<&[u8]> {
        let start = self.slot_offsets.get(slot).copied().flatten()?;
        let mut end = self.payload.len();
        for next in &self.slot_offsets[slot + 1..] {
            if let Some(next_off) = next {
                end = *next_off;
                break;
            }
        }
        self.payload.get(start..end)
    }
}

#[inline]
fn read_u16_le(bytes: &[u8], offset: usize) -> u16 {
    u16::from_le_bytes([bytes[offset], bytes[offset + 1]])
}
