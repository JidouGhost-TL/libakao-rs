// SPDX-License-Identifier: MIT OR Apache-2.0

#![doc = include_str!("../README.md")]
#![no_std]
#![forbid(unsafe_code)]
#![warn(missing_docs)]

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "std")]
extern crate std;

use core::fmt;

pub mod bank;
pub mod opcodes;
#[cfg(feature = "alloc")]
pub mod player;
pub mod reverb;
pub mod sequence;
pub mod sfx;
pub mod version;

#[cfg(feature = "alloc")]
pub use bank::AkaoWaveBank;
pub use bank::{AkaoWaveBankView, Articulation, WAVE_BANK_HEADER_SIZE};
pub use opcodes::{decode_opcode, ExtendedOpcode, Opcode, DURATION_TABLE};
#[cfg(feature = "alloc")]
pub use player::{AkaoPlayer, ChannelSnapshot, FadeCurve, RenderConfig, RenderResult};
pub use reverb::{ReverbParams, ReverbPreset};
#[cfg(feature = "alloc")]
pub use sequence::AkaoSequence;
pub use sequence::{AkaoSequenceView, DrumEntry, KeysplitEntry, AKAO_MAGIC};
#[cfg(feature = "alloc")]
pub use sfx::AkaoSfxBank;
pub use sfx::{AkaoSfxBankView, SFX_BANK_HEADER_SIZE, SFX_UNUSED_SLOT};
pub use version::{detect_version, AkaoVersion, PS1_ROOT_COUNTER_2_CLOCK_HZ};

/// Crate-wide result type alias for `libakao`.
pub type Result<T> = core::result::Result<T, Error>;

/// Errors returned by `libakao` parsing, relocation, upload, and playback APIs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer was shorter than required for the header, table, or opcode operand.
    UnexpectedEof(&'static str),
    /// Input did not begin with the expected `"AKAO"` signature or header magic.
    InvalidMagic,
    /// Header field violated a structural invariant (e.g., empty channel mask, invalid size).
    InvalidHeader(&'static str),
    /// Channel offset, table pointer, jump target, or RAM address was out of range.
    OutOfRange(&'static str),
    /// Automatic AKAO version detection could not match any supported generation.
    UnknownVersion,
    /// An invalid configuration or argument was supplied.
    InvalidArgument(&'static str),
    /// An underlying `libspu2` device error occurred.
    Spu(libspu2::Error),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof(ctx) => write!(f, "unexpected end of data: {ctx}"),
            Self::InvalidMagic => write!(f, "invalid AKAO magic header"),
            Self::InvalidHeader(msg) => write!(f, "invalid AKAO header: {msg}"),
            Self::OutOfRange(msg) => write!(f, "out of range: {msg}"),
            Self::UnknownVersion => write!(f, "unable to detect AKAO version"),
            Self::InvalidArgument(msg) => write!(f, "invalid argument: {msg}"),
            Self::Spu(err) => write!(f, "SPU error: {err}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Spu(err) => Some(err),
            _ => None,
        }
    }
}

impl From<libspu2::Error> for Error {
    fn from(err: libspu2::Error) -> Self {
        Self::Spu(err)
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Error>();
    assert_send_sync::<AkaoVersion>();
    assert_send_sync::<AkaoSequenceView<'static>>();
    assert_send_sync::<AkaoWaveBankView<'static>>();
    assert_send_sync::<AkaoSfxBankView<'static>>();
    assert_send_sync::<DrumEntry>();
    assert_send_sync::<KeysplitEntry>();
    assert_send_sync::<Articulation>();
    assert_send_sync::<ReverbPreset>();
    assert_send_sync::<ReverbParams>();
    assert_send_sync::<Opcode>();
    assert_send_sync::<ExtendedOpcode>();
    #[cfg(feature = "alloc")]
    {
        assert_send_sync::<AkaoSequence>();
        assert_send_sync::<AkaoWaveBank>();
        assert_send_sync::<AkaoSfxBank>();
        assert_send_sync::<FadeCurve>();
        assert_send_sync::<RenderConfig>();
        assert_send_sync::<RenderResult>();
        assert_send_sync::<ChannelSnapshot>();
        assert_send_sync::<AkaoPlayer>();
    }
};
