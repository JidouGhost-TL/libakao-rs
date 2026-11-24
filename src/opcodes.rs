// SPDX-License-Identifier: MIT OR Apache-2.0

//! AKAO bytecode opcode definitions and version-aware decoder.
//!
//! Operand byte lengths are separated by AKAO generation (`V1`, `V2a`/`V2b`, `V3`)
//! so real PlayStation 1 streams never desynchronize across `0xCC..=0xFF` or
//! extended `0xFC`/`0xFE` sub-opcodes.

use crate::version::AkaoVersion;
use crate::{Error, Result};

/// Standard AKAO 11-entry fixed note/tie/rest duration table (in ticks at 48 PPQN).
///
/// Indices `0..=10` correspond to:
/// whole (`192`), half (`96`), quarter (`48`), 8th (`24`), 16th (`12`), 32nd (`6`),
/// 64th (`3`), quarter-triplet (`32`), 8th-triplet (`16`), 16th-triplet (`8`),
/// and 32nd-triplet (`4`).
pub const DURATION_TABLE: [u8; 11] = [192, 96, 48, 24, 12, 6, 3, 32, 16, 8, 4];

/// Extended AKAO opcodes dispatched under prefix `0xFC` (`V2a`/`V2b`) or `0xFE` (`V3`).
///
/// Relative jump/call `i16` offsets are relative to the byte address of the `i16` operand
/// itself, matching the hardware AKAO dispatcher.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ExtendedOpcode {
    /// `0x00 <u16 tempo>`: Set 16-bit sequence tempo (loaded into high half of 16.16 tempo).
    SetTempo(u16),
    /// `0x01 <u8 duration> <u16 target>`: Slide sequence tempo over `duration` musical ticks.
    TempoSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target 16-bit tempo value.
        target: u16,
    },
    /// `0x02 <i16 depth>`: Set SPU reverb return depth (applied to `EVOLL`/`EVOLR`).
    SetReverbDepth(i16),
    /// `0x03 <u8 duration> <i16 target>`: Slide SPU reverb return depth over `duration` ticks.
    ReverbDepthSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target signed 16-bit reverb depth.
        target: i16,
    },
    /// `0x04` (`V3`): Enable drum/percussion mapping mode on this channel.
    DrumModeOn,
    /// `0x04 <i16 offset>` (`V2a`/`V2b`): Enable drum/percussion mapping mode with relative table offset.
    DrumModeOnOffset(i16),
    /// `0x05`: Disable drum/percussion mapping mode on this channel.
    DrumModeOff,
    /// `0x06 <i16 offset>`: Unconditional relative jump (`pc = operand_addr + offset`).
    Jump(i16),
    /// `0x07 <u8 condition> <i16 offset>`: Jump if sequence condition `>= condition`.
    ConditionalJump {
        /// Minimum condition value required to take the jump.
        condition: u8,
        /// Relative signed 16-bit jump offset from the `i16` operand address.
        offset: i16,
    },
    /// `0x08 <u8 iteration> <i16 offset>`: Pop loop and jump when current loop iteration matches.
    LoopBreak {
        /// 1-based loop iteration count on which to break out.
        iteration: u8,
        /// Relative signed 16-bit jump offset from the `i16` operand address.
        offset: i16,
    },
    /// `0x09 <u8 iteration> <i16 offset>`: Alternate loop-break jump on matching iteration.
    LoopBreakAlt {
        /// 1-based loop iteration count on which to break out.
        iteration: u8,
        /// Relative signed 16-bit jump offset from the `i16` operand address.
        offset: i16,
    },
    /// `0x0A <u8>`: Select instrument / articulation without resetting ADSR overrides.
    SetInstrumentAlt(u8),
    /// `0x0E <u8>` (`V2`): Set channel expression / volume balance (`0..=127`).
    VolumeBalance(u8),
    /// `0x0F <u8 duration> <u8 target>` (`V2`): Slide channel volume balance.
    VolumeBalanceSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target volume balance (`0..=127`).
        target: u8,
    },
    /// `0x0E <i16 offset>` (`V3`): Call subroutine at `operand_addr + offset`.
    SubroutineCall(i16),
    /// `0x0F` (`V3`): Return from subroutine.
    SubroutineReturn,
    /// `0x10 <u8>`: Select keysplit / bank region on this channel.
    SetKeysplit(u8),
    /// `0x11`: Clear keysplit mode on this channel.
    ResetKeysplit,
    /// `0x12 <u8 duration> <u8 target>`: Slide channel master volume.
    MasterVolumeSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target master volume (`0..=127`).
        target: u8,
    },
    /// `0x14 <u8>` (`V3`): Select custom keysplit instrument index.
    SelectKeysplitInstrument(u8),
    /// `0x15 <u8 num> <u8 den>`: Time signature marker (2 operand bytes).
    TimeSignature(u8, u8),
    /// `0x16 <u16 measure>`: Measure / bar marker (2 operand bytes).
    MeasureNumber(u16),
    /// Known 1-byte extended opcode consumed to preserve stream alignment.
    Reserved1(u8, u8),
    /// Known 2-byte extended opcode consumed to preserve stream alignment.
    Reserved2(u8, u8, u8),
    /// Known 3-byte extended opcode consumed to preserve stream alignment.
    Reserved3(u8, [u8; 3]),
    /// Known 4-byte extended opcode consumed to preserve stream alignment.
    Reserved4(u8, [u8; 4]),
    /// Unrecognized extended sub-opcode byte (0 operand bytes).
    Unknown(u8),
}

/// Decoded AKAO sequence opcode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Opcode {
    /// Trigger a note (`semitone` in `0..=11`, `duration` in musical ticks).
    Note {
        /// Semitone offset within the current octave (`0 = C .. 11 = B`).
        semitone: u8,
        /// Duration in musical ticks.
        duration: u8,
    },
    /// Tie / sustain the currently sounding note without re-keying or keying off.
    Tie {
        /// Duration in musical ticks.
        duration: u8,
    },
    /// Rest (keys off active note and clears slur/portamento state).
    Rest {
        /// Duration in musical ticks.
        duration: u8,
    },
    /// No-operation (`0x9A..=0x9F`, and 0-byte reserved no-ops).
    Nop,
    /// `0xA0` (and `0xFF`, plus unused halt slots): Halt channel and key off voice.
    EndOfTrack,
    /// `0xA1 <u8>`: Select instrument / articulation index.
    SetInstrument(u8),
    /// `0xA2 <u8>`: Override duration of the next note/tie/rest (`0` clears override).
    OverrideDuration(u8),
    /// `0xA3 <u8>`: Set channel master volume (`0..=127`).
    SetMasterVolume(u8),
    /// `0xA4 <u8 duration> <i8 semitones>`: Slide pitch by `semitones` over `duration` ticks.
    PitchBendSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Signed semitone delta.
        semitones: i8,
    },
    /// `0xA5 <u8>`: Set channel octave (`0..=10`).
    SetOctave(u8),
    /// `0xA6`: Increment channel octave by 1.
    OctaveUp,
    /// `0xA7`: Decrement channel octave by 1.
    OctaveDown,
    /// `0xA8 <u8>`: Set channel volume (`0..=127`).
    SetVolume(u8),
    /// `0xA9 <u8 duration> <u8 target>`: Slide channel volume to `target` over `duration` ticks.
    VolumeSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target volume (`0..=127`).
        target: u8,
    },
    /// `0xAA <u8>`: Set channel stereo pan (`0 = left, 64 = center, 127 = right`).
    SetPan(u8),
    /// `0xAB <u8 duration> <u8 target>`: Slide channel pan to `target` over `duration` ticks.
    PanSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target pan (`0..=127`).
        target: u8,
    },
    /// `0xAC <u8>`: Set PS1 SPU noise generator clock frequency (`0..=63`).
    NoiseClock(u8),
    /// `0xAD <u8>`: Override ADSR1 attack rate (`0..=127`).
    SetAdsrAttackRate(u8),
    /// `0xAE <u8>`: Override ADSR1 decay rate (`0..=15`).
    SetAdsrDecayRate(u8),
    /// `0xAF <u8>`: Override ADSR1 sustain level (`0..=15`).
    SetAdsrSustainLevel(u8),
    /// `0xB0 <u8 decay> <u8 sustain_level>`: Override ADSR1 decay rate and sustain level.
    SetAdsrDecayAndSustain {
        /// Decay rate (`0..=15`).
        decay: u8,
        /// Sustain level (`0..=15`).
        sustain_level: u8,
    },
    /// `0xB1 <u8>`: Override ADSR2 sustain rate (`0..=127`).
    SetAdsrSustainRate(u8),
    /// `0xB2 <u8>`: Override ADSR2 release rate (`0..=31`).
    SetAdsrReleaseRate(u8),
    /// `0xB3`: Restore ADSR1/ADSR2 from the current instrument articulation.
    ResetAdsr,
    /// `0xB4 <u8 delay> <u8 rate> <u8 waveform>`: Enable pitch vibrato LFO.
    Vibrato {
        /// Delay in musical ticks before vibrato begins.
        delay: u8,
        /// Full-cycle LFO period in musical ticks (`0` treated as `256`).
        rate: u8,
        /// Waveform table selector (`0` sine, `1` triangle, `2` square, `3` sawtooth).
        waveform: u8,
    },
    /// `0xB5 <u8>`: Set pitch vibrato LFO depth (bit `0x80` selects coarse range).
    VibratoDepth(u8),
    /// `0xB6`: Disable pitch vibrato LFO.
    VibratoOff,
    /// `0xB7 <u8>`: Override ADSR1 attack mode (`0` = linear, non-zero = exponential).
    SetAdsrAttackMode(u8),
    /// `0xB8 <u8 delay> <u8 rate> <u8 waveform>`: Enable volume tremolo LFO.
    Tremolo {
        /// Delay in musical ticks before tremolo begins.
        delay: u8,
        /// Full-cycle LFO period in musical ticks (`0` treated as `256`).
        rate: u8,
        /// Waveform table selector.
        waveform: u8,
    },
    /// `0xB9 <u8>`: Set volume tremolo LFO depth.
    TremoloDepth(u8),
    /// `0xBA`: Disable volume tremolo LFO.
    TremoloOff,
    /// `0xBB <u8>`: Override ADSR2 sustain mode (`0..=3`: bit 1 = exp, bit 0 = decrease).
    SetAdsrSustainMode(u8),
    /// `0xBC <u8 rate> <u8 waveform>`: Enable stereo auto-pan LFO.
    AutoPan {
        /// Full-cycle LFO period in musical ticks (`0` treated as `256`).
        rate: u8,
        /// Waveform table selector.
        waveform: u8,
    },
    /// `0xBD <u8>`: Set stereo auto-pan LFO depth.
    AutoPanDepth(u8),
    /// `0xBE`: Disable stereo auto-pan LFO.
    AutoPanOff,
    /// `0xBF <u8>`: Override ADSR2 release mode (`0` = linear, non-zero = exponential).
    SetAdsrReleaseMode(u8),
    /// `0xC0 <i8>`: Set absolute semitone transpose.
    SetTranspose(i8),
    /// `0xC1 <i8>`: Add signed semitone delta to current transpose.
    AddTranspose(i8),
    /// `0xC2`: Enable SPU reverb send (`EON`) for this channel.
    ReverbOn,
    /// `0xC3`: Disable SPU reverb send (`EON`) for this channel.
    ReverbOff,
    /// `0xC4`: Enable SPU hardware noise (`NON`) for this channel.
    NoiseOn,
    /// `0xC5`: Disable SPU hardware noise (`NON`) for this channel.
    NoiseOff,
    /// `0xC6`: Enable SPU pitch modulation (`PMON`) for this channel.
    PitchModOn,
    /// `0xC7`: Disable SPU pitch modulation (`PMON`) for this channel.
    PitchModOff,
    /// `0xC8`: Push current stream position onto the 4-deep loop stack.
    LoopStart,
    /// `0xC9 <u8 count>`: Repeat loop (`count == 0` = 256 iterations; otherwise `count` total iterations).
    LoopEnd(u8),
    /// `0xCA`: Unconditionally repeat the innermost loop (infinite loop).
    LoopInfinite,
    /// `0xCB`: Reset voice modulation/noise/reverb/slur/portamento flags.
    ResetVoiceModes,
    /// `0xCC`: Enable slur (suppress key-off between consecutive notes).
    SlurOn,
    /// `0xCD`: Disable slur.
    SlurOff,
    /// `0xCE <u8>`: Enable noise switch timer (`1` operand byte).
    NoiseTimerOn(u8),
    /// `0xCF <u8>`: Set noise switch delay (`1` operand byte).
    NoiseTimerDelay(u8),
    /// `0xD0`: Enable legato / roll (suppress key-on retrigger on consecutive notes).
    LegatoOn,
    /// `0xD1`: Disable legato / roll.
    LegatoOff,
    /// `0xD2 <u8>`: Enable pitch-modulation switch timer (`1` operand byte).
    PitchModTimerOn(u8),
    /// `0xD3 <u8>`: Set pitch-modulation switch delay (`1` operand byte).
    PitchModTimerDelay(u8),
    /// `0xD4`: Enable right-channel stereo phase inversion (signed negative `V_VOLR`).
    PhaseInvertOn,
    /// `0xD5`: Disable right-channel stereo phase inversion.
    PhaseInvertOff,
    /// `0xD6`: Enable alternate stereo mode flag.
    StereoModeOn,
    /// `0xD7`: Disable alternate stereo mode flag.
    StereoModeOff,
    /// `0xD8 <i8>`: Set fine-tune pitch offset (signed cents-like step).
    FineTune(i8),
    /// `0xD9 <i8>`: Add signed delta to fine-tune pitch offset.
    AddFineTune(i8),
    /// `0xDA <u8>`: Enable portamento glide over `duration` ticks.
    PortamentoOn(u8),
    /// `0xDB`: Disable portamento glide.
    PortamentoOff,
    /// `0xDC <i8>`: Adjust fixed note gate length delta (`1` operand byte).
    FixNoteLength(i8),
    /// `0xDD <u8 duration> <u8 target>`: Slide pitch vibrato depth (`2` operand bytes).
    VibratoDepthSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target vibrato depth.
        target: u8,
    },
    /// `0xDE <u8 duration> <u8 target>`: Slide volume tremolo depth (`2` operand bytes).
    TremoloDepthSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target tremolo depth.
        target: u8,
    },
    /// `0xDF <u8 duration> <u8 target>`: Slide auto-pan depth (`2` operand bytes).
    AutoPanDepthSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target auto-pan depth.
        target: u8,
    },
    /// `0xE4 <u8 duration> <u8 target>` (`V3`): Slide vibrato LFO rate (`2` operand bytes).
    VibratoRateSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target vibrato rate.
        target: u8,
    },
    /// `0xE5 <u8 duration> <u8 target>` (`V3`): Slide tremolo LFO rate (`2` operand bytes).
    TremoloRateSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target tremolo rate.
        target: u8,
    },
    /// `0xE6 <u8 duration> <u8 target>` (`V3`): Slide auto-pan LFO rate (`2` operand bytes).
    AutoPanRateSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target auto-pan rate.
        target: u8,
    },
    /// `0xE8 <u16>` (`V1`): Set sequence tempo directly.
    SetTempo(u16),
    /// `0xE9 <u8 duration> <u16 target>` (`V1`): Slide sequence tempo directly.
    TempoSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target 16-bit tempo value.
        target: u16,
    },
    /// `0xEA <i16 depth>` (`V1`): Set SPU reverb return depth (`EVOLL`/`EVOLR`).
    SetReverbDepth(i16),
    /// `0xEB <u8 duration> <i16 target>` (`V1`): Slide SPU reverb return depth.
    ReverbDepthSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target signed 16-bit reverb depth.
        target: i16,
    },
    /// `0xEC <i16 offset>` (`V1`): Enable drum mode with relative table offset.
    DrumModeOnOffset(i16),
    /// `0xED` (`V1`): Disable drum mode.
    DrumModeOff,
    /// `0xEE <i16 offset>` (`V1`): Unconditional relative jump.
    Jump(i16),
    /// `0xEF <u8 condition> <i16 offset>` (`V1`): Conditional relative jump.
    ConditionalJump {
        /// Minimum condition value required to take the jump.
        condition: u8,
        /// Relative signed 16-bit jump offset from the `i16` operand address.
        offset: i16,
    },
    /// `0xF0 <u8 iteration> <i16 offset>` (`V1`): Loop break jump on matching iteration.
    LoopBreak {
        /// 1-based loop iteration count on which to break out.
        iteration: u8,
        /// Relative signed 16-bit jump offset.
        offset: i16,
    },
    /// `0xF1 <u8 iteration> <i16 offset>` (`V1`): Alternate loop break jump.
    LoopBreakAlt {
        /// 1-based loop iteration count on which to break out.
        iteration: u8,
        /// Relative signed 16-bit jump offset.
        offset: i16,
    },
    /// `0xF2 <u8>` (`V1`): Select instrument / articulation without resetting ADSR.
    SetInstrumentAlt(u8),
    /// `0xF6 <u8>` (`V1`): Set channel expression / volume balance (`0..=127`).
    VolumeBalance(u8),
    /// `0xF7 <u8 duration> <u8 target>` (`V1`): Slide channel volume balance.
    VolumeBalanceSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target volume balance (`0..=127`).
        target: u8,
    },
    /// `0xF8 <u8>` (`V1`): Select keysplit / bank region.
    SetKeysplit(u8),
    /// `0xF9` (`V1`): Reset keysplit mode.
    ResetKeysplit,
    /// Known 1-byte opcode consumed to preserve stream alignment.
    Reserved1(u8, u8),
    /// Known 2-byte opcode consumed to preserve stream alignment.
    Reserved2(u8, u8, u8),
    /// Version-prefixed extended opcode (`0xFC` on `V2a`/`V2b`, `0xFE` on `V3`).
    Extended(ExtendedOpcode),
    /// Unrecognized opcode byte (consumed as 1 byte).
    Unknown(u8),
}

/// Decodes a single AKAO opcode at `pc` in `data` for `version`, returning `(opcode, next_pc)`.
pub fn decode_opcode(data: &[u8], pc: usize, version: AkaoVersion) -> Result<(Opcode, usize)> {
    let op = *data
        .get(pc)
        .ok_or(Error::UnexpectedEof("AKAO opcode byte"))?;
    let mut cur = pc + 1;

    if op <= 0x83 {
        let semitone = op / 11;
        let duration = DURATION_TABLE[(op % 11) as usize];
        return Ok((Opcode::Note { semitone, duration }, cur));
    }
    if op <= 0x8E {
        let duration = DURATION_TABLE[(op - 0x84) as usize];
        return Ok((Opcode::Tie { duration }, cur));
    }
    if op <= 0x99 {
        let duration = DURATION_TABLE[(op - 0x8F) as usize];
        return Ok((Opcode::Rest { duration }, cur));
    }
    if op <= 0x9F {
        return Ok((Opcode::Nop, cur));
    }

    // Version-divergent range (`0xE0..=0xFF`).
    if op >= 0xE0 {
        return decode_version_specific(data, op, cur, version);
    }

    // Common range (`0xA0..=0xDF`), verified identical across V1, V2a/V2b, and V3.
    let decoded = match op {
        0xA0 => Opcode::EndOfTrack,
        0xA1 => Opcode::SetInstrument(read_u8(data, &mut cur)?),
        0xA2 => Opcode::OverrideDuration(read_u8(data, &mut cur)?),
        0xA3 => Opcode::SetMasterVolume(read_u8(data, &mut cur)?),
        0xA4 => {
            let duration = read_u8(data, &mut cur)?;
            let semitones = read_u8(data, &mut cur)? as i8;
            Opcode::PitchBendSlide {
                duration,
                semitones,
            }
        }
        0xA5 => Opcode::SetOctave(read_u8(data, &mut cur)?),
        0xA6 => Opcode::OctaveUp,
        0xA7 => Opcode::OctaveDown,
        0xA8 => Opcode::SetVolume(read_u8(data, &mut cur)?),
        0xA9 => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u8(data, &mut cur)?;
            Opcode::VolumeSlide { duration, target }
        }
        0xAA => Opcode::SetPan(read_u8(data, &mut cur)?),
        0xAB => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u8(data, &mut cur)?;
            Opcode::PanSlide { duration, target }
        }
        0xAC => Opcode::NoiseClock(read_u8(data, &mut cur)?),
        0xAD => Opcode::SetAdsrAttackRate(read_u8(data, &mut cur)?),
        0xAE => Opcode::SetAdsrDecayRate(read_u8(data, &mut cur)?),
        0xAF => Opcode::SetAdsrSustainLevel(read_u8(data, &mut cur)?),
        0xB0 => {
            let decay = read_u8(data, &mut cur)?;
            let sustain_level = read_u8(data, &mut cur)?;
            Opcode::SetAdsrDecayAndSustain {
                decay,
                sustain_level,
            }
        }
        0xB1 => Opcode::SetAdsrSustainRate(read_u8(data, &mut cur)?),
        0xB2 => Opcode::SetAdsrReleaseRate(read_u8(data, &mut cur)?),
        0xB3 => Opcode::ResetAdsr,
        0xB4 => {
            let delay = read_u8(data, &mut cur)?;
            let rate = read_u8(data, &mut cur)?;
            let waveform = read_u8(data, &mut cur)?;
            Opcode::Vibrato {
                delay,
                rate,
                waveform,
            }
        }
        0xB5 => Opcode::VibratoDepth(read_u8(data, &mut cur)?),
        0xB6 => Opcode::VibratoOff,
        0xB7 => Opcode::SetAdsrAttackMode(read_u8(data, &mut cur)?),
        0xB8 => {
            let delay = read_u8(data, &mut cur)?;
            let rate = read_u8(data, &mut cur)?;
            let waveform = read_u8(data, &mut cur)?;
            Opcode::Tremolo {
                delay,
                rate,
                waveform,
            }
        }
        0xB9 => Opcode::TremoloDepth(read_u8(data, &mut cur)?),
        0xBA => Opcode::TremoloOff,
        0xBB => Opcode::SetAdsrSustainMode(read_u8(data, &mut cur)?),
        0xBC => {
            let rate = read_u8(data, &mut cur)?;
            let waveform = read_u8(data, &mut cur)?;
            Opcode::AutoPan { rate, waveform }
        }
        0xBD => Opcode::AutoPanDepth(read_u8(data, &mut cur)?),
        0xBE => Opcode::AutoPanOff,
        0xBF => Opcode::SetAdsrReleaseMode(read_u8(data, &mut cur)?),
        0xC0 => Opcode::SetTranspose(read_u8(data, &mut cur)? as i8),
        0xC1 => Opcode::AddTranspose(read_u8(data, &mut cur)? as i8),
        0xC2 => Opcode::ReverbOn,
        0xC3 => Opcode::ReverbOff,
        0xC4 => Opcode::NoiseOn,
        0xC5 => Opcode::NoiseOff,
        0xC6 => Opcode::PitchModOn,
        0xC7 => Opcode::PitchModOff,
        0xC8 => Opcode::LoopStart,
        0xC9 => Opcode::LoopEnd(read_u8(data, &mut cur)?),
        0xCA => Opcode::LoopInfinite,
        0xCB => Opcode::ResetVoiceModes,
        0xCC => Opcode::SlurOn,
        0xCD => Opcode::SlurOff,
        0xCE => Opcode::NoiseTimerOn(read_u8(data, &mut cur)?),
        0xCF => Opcode::NoiseTimerDelay(read_u8(data, &mut cur)?),
        0xD0 => Opcode::LegatoOn,
        0xD1 => Opcode::LegatoOff,
        0xD2 => Opcode::PitchModTimerOn(read_u8(data, &mut cur)?),
        0xD3 => Opcode::PitchModTimerDelay(read_u8(data, &mut cur)?),
        0xD4 => Opcode::PhaseInvertOn,
        0xD5 => Opcode::PhaseInvertOff,
        0xD6 => Opcode::StereoModeOn,
        0xD7 => Opcode::StereoModeOff,
        0xD8 => Opcode::FineTune(read_u8(data, &mut cur)? as i8),
        0xD9 => Opcode::AddFineTune(read_u8(data, &mut cur)? as i8),
        0xDA => Opcode::PortamentoOn(read_u8(data, &mut cur)?),
        0xDB => Opcode::PortamentoOff,
        0xDC => Opcode::FixNoteLength(read_u8(data, &mut cur)? as i8),
        0xDD => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u8(data, &mut cur)?;
            Opcode::VibratoDepthSlide { duration, target }
        }
        0xDE => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u8(data, &mut cur)?;
            Opcode::TremoloDepthSlide { duration, target }
        }
        0xDF => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u8(data, &mut cur)?;
            Opcode::AutoPanDepthSlide { duration, target }
        }
        _ => Opcode::Unknown(op),
    };

    Ok((decoded, cur))
}

fn decode_version_specific(
    data: &[u8],
    op: u8,
    mut cur: usize,
    version: AkaoVersion,
) -> Result<(Opcode, usize)> {
    match version {
        AkaoVersion::V1 => {
            let decoded = match op {
                0xE0..=0xE7 => Opcode::EndOfTrack,
                0xE8 => Opcode::SetTempo(read_u16_le(data, &mut cur)?),
                0xE9 => {
                    let duration = read_u8(data, &mut cur)?;
                    let target = read_u16_le(data, &mut cur)?;
                    Opcode::TempoSlide { duration, target }
                }
                0xEA => Opcode::SetReverbDepth(read_i16_le(data, &mut cur)?),
                0xEB => {
                    let duration = read_u8(data, &mut cur)?;
                    let target = read_i16_le(data, &mut cur)?;
                    Opcode::ReverbDepthSlide { duration, target }
                }
                0xEC => Opcode::DrumModeOnOffset(read_i16_le(data, &mut cur)?),
                0xED => Opcode::DrumModeOff,
                0xEE => Opcode::Jump(read_i16_le(data, &mut cur)?),
                0xEF => {
                    let condition = read_u8(data, &mut cur)?;
                    let offset = read_i16_le(data, &mut cur)?;
                    Opcode::ConditionalJump { condition, offset }
                }
                0xF0 => {
                    let iteration = read_u8(data, &mut cur)?;
                    let offset = read_i16_le(data, &mut cur)?;
                    Opcode::LoopBreak { iteration, offset }
                }
                0xF1 => {
                    let iteration = read_u8(data, &mut cur)?;
                    let offset = read_i16_le(data, &mut cur)?;
                    Opcode::LoopBreakAlt { iteration, offset }
                }
                0xF2 => Opcode::SetInstrumentAlt(read_u8(data, &mut cur)?),
                0xF3 | 0xF5 => Opcode::Nop,
                0xF4 => {
                    let a = read_u8(data, &mut cur)?;
                    let b = read_u8(data, &mut cur)?;
                    Opcode::Reserved2(0xF4, a, b)
                }
                0xF6 => Opcode::VolumeBalance(read_u8(data, &mut cur)?),
                0xF7 => {
                    let duration = read_u8(data, &mut cur)?;
                    let target = read_u8(data, &mut cur)?;
                    Opcode::VolumeBalanceSlide { duration, target }
                }
                0xF8 => Opcode::SetKeysplit(read_u8(data, &mut cur)?),
                0xF9 => Opcode::ResetKeysplit,
                0xFA..=0xFC | 0xFF => Opcode::EndOfTrack,
                0xFD | 0xFE => {
                    let a = read_u8(data, &mut cur)?;
                    let b = read_u8(data, &mut cur)?;
                    Opcode::Reserved2(op, a, b)
                }
                _ => Opcode::Unknown(op),
            };
            Ok((decoded, cur))
        }
        AkaoVersion::V2a | AkaoVersion::V2b => match op {
            0xE0 => Ok((Opcode::Nop, cur)),
            0xE1..=0xFB | 0xFD..=0xFF => Ok((Opcode::EndOfTrack, cur)),
            0xFC => {
                let ext = decode_extended_v2(data, &mut cur)?;
                Ok((Opcode::Extended(ext), cur))
            }
            _ => Ok((Opcode::Unknown(op), cur)),
        },
        AkaoVersion::V3 => match op {
            0xE0 | 0xE2 => Ok((Opcode::Nop, cur)),
            0xE1 => {
                let b = read_u8(data, &mut cur)?;
                Ok((Opcode::Reserved1(0xE1, b), cur))
            }
            0xE4 => {
                let duration = read_u8(data, &mut cur)?;
                let target = read_u8(data, &mut cur)?;
                Ok((Opcode::VibratoRateSlide { duration, target }, cur))
            }
            0xE5 => {
                let duration = read_u8(data, &mut cur)?;
                let target = read_u8(data, &mut cur)?;
                Ok((Opcode::TremoloRateSlide { duration, target }, cur))
            }
            0xE6 => {
                let duration = read_u8(data, &mut cur)?;
                let target = read_u8(data, &mut cur)?;
                Ok((Opcode::AutoPanRateSlide { duration, target }, cur))
            }
            0xE3 | 0xE7..=0xEF | 0xFF => Ok((Opcode::EndOfTrack, cur)),
            0xF0..=0xFB => {
                let duration = read_u8(data, &mut cur)?;
                Ok((
                    Opcode::Note {
                        semitone: op - 0xF0,
                        duration,
                    },
                    cur,
                ))
            }
            0xFC => {
                let duration = read_u8(data, &mut cur)?;
                Ok((Opcode::Tie { duration }, cur))
            }
            0xFD => {
                let duration = read_u8(data, &mut cur)?;
                Ok((Opcode::Rest { duration }, cur))
            }
            0xFE => {
                let ext = decode_extended_v3(data, &mut cur)?;
                Ok((Opcode::Extended(ext), cur))
            }
            _ => Ok((Opcode::Unknown(op), cur)),
        },
    }
}

fn decode_extended_v2(data: &[u8], cur: &mut usize) -> Result<ExtendedOpcode> {
    let sub = read_u8(data, cur)?;
    let ext = match sub {
        0x00 => ExtendedOpcode::SetTempo(read_u16_le(data, cur)?),
        0x01 => {
            let duration = read_u8(data, cur)?;
            let target = read_u16_le(data, cur)?;
            ExtendedOpcode::TempoSlide { duration, target }
        }
        0x02 => ExtendedOpcode::SetReverbDepth(read_i16_le(data, cur)?),
        0x03 => {
            let duration = read_u8(data, cur)?;
            let target = read_i16_le(data, cur)?;
            ExtendedOpcode::ReverbDepthSlide { duration, target }
        }
        0x04 => ExtendedOpcode::DrumModeOnOffset(read_i16_le(data, cur)?),
        0x05 => ExtendedOpcode::DrumModeOff,
        0x06 => ExtendedOpcode::Jump(read_i16_le(data, cur)?),
        0x07 => {
            let condition = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::ConditionalJump { condition, offset }
        }
        0x08 => {
            let iteration = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::LoopBreak { iteration, offset }
        }
        0x09 => {
            let iteration = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::LoopBreakAlt { iteration, offset }
        }
        0x0A => ExtendedOpcode::SetInstrumentAlt(read_u8(data, cur)?),
        0x0B => {
            let b0 = read_u8(data, cur)?;
            let b1 = read_u8(data, cur)?;
            let b2 = read_u8(data, cur)?;
            let b3 = read_u8(data, cur)?;
            ExtendedOpcode::Reserved4(0x0B, [b0, b1, b2, b3])
        }
        0x0C | 0x14 | 0x18 => {
            let a = read_u8(data, cur)?;
            let b = read_u8(data, cur)?;
            ExtendedOpcode::Reserved2(sub, a, b)
        }
        0x0E => ExtendedOpcode::VolumeBalance(read_u8(data, cur)?),
        0x0F => {
            let duration = read_u8(data, cur)?;
            let target = read_u8(data, cur)?;
            ExtendedOpcode::VolumeBalanceSlide { duration, target }
        }
        0x10 => ExtendedOpcode::SetKeysplit(read_u8(data, cur)?),
        0x11 => ExtendedOpcode::ResetKeysplit,
        0x12 => {
            let duration = read_u8(data, cur)?;
            let target = read_u8(data, cur)?;
            ExtendedOpcode::MasterVolumeSlide { duration, target }
        }
        0x15 => {
            let a = read_u8(data, cur)?;
            let b = read_u8(data, cur)?;
            ExtendedOpcode::TimeSignature(a, b)
        }
        0x16 => ExtendedOpcode::MeasureNumber(read_u16_le(data, cur)?),
        0x17 => ExtendedOpcode::Reserved1(0x17, read_u8(data, cur)?),
        other => ExtendedOpcode::Unknown(other),
    };
    Ok(ext)
}

fn decode_extended_v3(data: &[u8], cur: &mut usize) -> Result<ExtendedOpcode> {
    let sub = read_u8(data, cur)?;
    let ext = match sub {
        0x00 => ExtendedOpcode::SetTempo(read_u16_le(data, cur)?),
        0x01 => {
            let duration = read_u8(data, cur)?;
            let target = read_u16_le(data, cur)?;
            ExtendedOpcode::TempoSlide { duration, target }
        }
        0x02 => ExtendedOpcode::SetReverbDepth(read_i16_le(data, cur)?),
        0x03 => {
            let duration = read_u8(data, cur)?;
            let target = read_i16_le(data, cur)?;
            ExtendedOpcode::ReverbDepthSlide { duration, target }
        }
        0x04 => ExtendedOpcode::DrumModeOn,
        0x05 => ExtendedOpcode::DrumModeOff,
        0x06 => ExtendedOpcode::Jump(read_i16_le(data, cur)?),
        0x07 => {
            let condition = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::ConditionalJump { condition, offset }
        }
        0x08 => {
            let iteration = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::LoopBreak { iteration, offset }
        }
        0x09 => {
            let iteration = read_u8(data, cur)?;
            let offset = read_i16_le(data, cur)?;
            ExtendedOpcode::LoopBreakAlt { iteration, offset }
        }
        0x0A => ExtendedOpcode::SetInstrumentAlt(read_u8(data, cur)?),
        0x0B | 0x1C => ExtendedOpcode::Reserved1(sub, read_u8(data, cur)?),
        0x0E => ExtendedOpcode::SubroutineCall(read_i16_le(data, cur)?),
        0x0F => ExtendedOpcode::SubroutineReturn,
        0x10 => ExtendedOpcode::SetKeysplit(read_u8(data, cur)?),
        0x11 => ExtendedOpcode::ResetKeysplit,
        0x12 => {
            let duration = read_u8(data, cur)?;
            let target = read_u8(data, cur)?;
            ExtendedOpcode::MasterVolumeSlide { duration, target }
        }
        0x14 => ExtendedOpcode::SelectKeysplitInstrument(read_u8(data, cur)?),
        0x15 => {
            let a = read_u8(data, cur)?;
            let b = read_u8(data, cur)?;
            ExtendedOpcode::TimeSignature(a, b)
        }
        0x16 => ExtendedOpcode::MeasureNumber(read_u16_le(data, cur)?),
        0x19 => {
            let b0 = read_u8(data, cur)?;
            let b1 = read_u8(data, cur)?;
            let b2 = read_u8(data, cur)?;
            ExtendedOpcode::Reserved3(0x19, [b0, b1, b2])
        }
        other => ExtendedOpcode::Unknown(other),
    };
    Ok(ext)
}

#[inline]
fn read_u8(data: &[u8], cur: &mut usize) -> Result<u8> {
    let b = *data
        .get(*cur)
        .ok_or(Error::UnexpectedEof("AKAO opcode operand"))?;
    *cur += 1;
    Ok(b)
}

#[inline]
fn read_u16_le(data: &[u8], cur: &mut usize) -> Result<u16> {
    let lo = read_u8(data, cur)?;
    let hi = read_u8(data, cur)?;
    Ok(u16::from_le_bytes([lo, hi]))
}

#[inline]
fn read_i16_le(data: &[u8], cur: &mut usize) -> Result<i16> {
    Ok(read_u16_le(data, cur)? as i16)
}
