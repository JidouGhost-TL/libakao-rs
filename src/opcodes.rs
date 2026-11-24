// SPDX-License-Identifier: MIT

//! AKAO bytecode opcode definitions and version-aware decoder.

use crate::version::AkaoVersion;
use crate::Error;

/// Standard AKAO 11-entry fixed note/tie/rest duration table (in ticks at 48 PPQN).
///
/// Indices `0..=10` correspond to:
/// whole (`192`), half (`96`), quarter (`48`), 8th (`24`), 16th (`12`), 32nd (`6`),
/// 64th (`3`), quarter-triplet (`32`), 8th-triplet (`16`), 16th-triplet (`8`),
/// and 32nd-triplet (`4`).
pub const DURATION_TABLE: [u8; 11] = [192, 96, 48, 24, 12, 6, 3, 32, 16, 8, 4];

/// Extended AKAO opcodes dispatched under prefix `0xFC` (`V1`/`V2a`/`V2b`) or `0xFE` (`V3`).
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
    /// `0x04`: Enable drum/percussion mapping mode on this channel.
    DrumModeOn,
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
    /// `0x0E <i16 offset>` (`V3` only): Call subroutine at `operand_addr + offset`.
    SubroutineCall(i16),
    /// `0x0F` (`V3` only): Return from subroutine.
    SubroutineReturn,
    /// Unrecognized extended sub-opcode byte.
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
    /// No-operation (`0x9A..=0x9F`).
    Nop,
    /// `0xA0` (and `0xFF` on `V3`): Halt channel and key off voice.
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
        /// LFO step period in ticks (`0` treated as `256`).
        rate: u8,
        /// Waveform table selector (`0` sine, `1` triangle, `2` square, `3` sawtooth).
        waveform: u8,
    },
    /// `0xB5 <u8>`: Set pitch vibrato LFO depth.
    VibratoDepth(u8),
    /// `0xB6`: Disable pitch vibrato LFO.
    VibratoOff,
    /// `0xB7 <u8>`: Override ADSR1 attack mode (`0` = linear, non-zero = exponential).
    SetAdsrAttackMode(u8),
    /// `0xB8 <u8 delay> <u8 rate> <u8 waveform>`: Enable volume tremolo LFO.
    Tremolo {
        /// Delay in musical ticks before tremolo begins.
        delay: u8,
        /// LFO step period in ticks (`0` treated as `256`).
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
        /// LFO step period in ticks (`0` treated as `256`).
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
    /// `0xC9 <u8 count>`: Repeat loop (`count == 0` = infinite; otherwise `count` total iterations).
    LoopEnd(u8),
    /// `0xCA`: Unconditionally repeat the innermost loop (infinite loop).
    LoopInfinite,
    /// `0xCB`: Reset voice modulation/noise/reverb/slur/portamento flags.
    ResetVoiceModes,
    /// `0xCC`: Enable slur (suppress key-off between consecutive notes).
    SlurOn,
    /// `0xCD`: Disable slur.
    SlurOff,
    /// `0xCE`: Enable legato (suppress key-on retrigger on consecutive notes).
    LegatoOn,
    /// `0xCF`: Disable legato.
    LegatoOff,
    /// `0xD8 <i8>`: Set fine-tune pitch offset (signed cents-like step).
    FineTune(i8),
    /// `0xD9 <i8>`: Add signed delta to fine-tune pitch offset.
    AddFineTune(i8),
    /// `0xDA <u8>`: Enable portamento glide over `duration` ticks.
    PortamentoOn(u8),
    /// `0xDB`: Disable portamento glide.
    PortamentoOff,
    /// `0xDC`: Enable right-channel stereo phase inversion (signed negative `V_VOLR`).
    PhaseInvertOn,
    /// `0xDD`: Disable right-channel stereo phase inversion.
    PhaseInvertOff,
    /// `0xDE <u8>`: Set channel expression / volume balance (`0..=127`).
    VolumeBalance(u8),
    /// `0xDF <i16 left> <i16 right>`: Set stereo SPU reverb return depth (`EVOLL`/`EVOLR`).
    SetReverbDepthStereo {
        /// Left reverb return volume (`EVOLL`).
        left: i16,
        /// Right reverb return volume (`EVOLR`).
        right: i16,
    },
    /// `0xE8 <u16>`: Set sequence tempo directly (shorthand alias of extended `0x00`).
    SetTempo(u16),
    /// `0xE9 <u8 duration> <u16 target>`: Slide sequence tempo directly.
    TempoSlide {
        /// Slide duration in musical ticks.
        duration: u8,
        /// Target 16-bit tempo value.
        target: u16,
    },
    /// `0xEA <u8>`: Set global sequence condition variable (tested by `ConditionalJump`).
    SetCondition(u8),
    /// Version-prefixed extended opcode (`0xFC` on `V1`/`V2`, `0xFE` on `V3`).
    Extended(ExtendedOpcode),
    /// Unrecognized opcode byte (consumed as 1 byte).
    Unknown(u8),
}

/// Decodes a single AKAO opcode at `pc` in `data` for `version`, returning `(opcode, next_pc)`.
pub fn decode_opcode(
    data: &[u8],
    pc: usize,
    version: AkaoVersion,
) -> Result<(Opcode, usize), Error> {
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

    // Version-specific high range (0xF0..=0xFF).
    if op >= 0xF0 {
        return decode_high_range(data, op, cur, version);
    }

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
        0xCE => Opcode::LegatoOn,
        0xCF => Opcode::LegatoOff,
        0xD8 => Opcode::FineTune(read_u8(data, &mut cur)? as i8),
        0xD9 => Opcode::AddFineTune(read_u8(data, &mut cur)? as i8),
        0xDA => Opcode::PortamentoOn(read_u8(data, &mut cur)?),
        0xDB => Opcode::PortamentoOff,
        0xDC => Opcode::PhaseInvertOn,
        0xDD => Opcode::PhaseInvertOff,
        0xDE => Opcode::VolumeBalance(read_u8(data, &mut cur)?),
        0xDF => {
            let left = read_i16_le(data, &mut cur)?;
            let right = read_i16_le(data, &mut cur)?;
            Opcode::SetReverbDepthStereo { left, right }
        }
        0xE8 => Opcode::SetTempo(read_u16_le(data, &mut cur)?),
        0xE9 => {
            let duration = read_u8(data, &mut cur)?;
            let target = read_u16_le(data, &mut cur)?;
            Opcode::TempoSlide { duration, target }
        }
        0xEA => Opcode::SetCondition(read_u8(data, &mut cur)?),
        other => Opcode::Unknown(other),
    };

    Ok((decoded, cur))
}

fn decode_high_range(
    data: &[u8],
    op: u8,
    mut cur: usize,
    version: AkaoVersion,
) -> Result<(Opcode, usize), Error> {
    if version.has_explicit_duration_opcodes() {
        // AKAO V3 high-range opcodes:
        // - 0xF0..=0xFB + u8 dur: explicit-duration notes
        // - 0xFC + u8 dur: explicit-duration tie
        // - 0xFD + u8 dur: explicit-duration rest
        // - 0xFE + sub: extended opcode prefix (including subroutines 0x0E / 0x0F)
        // - 0xFF: end of track alias
        match op {
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
                let ext = decode_extended(data, &mut cur, version)?;
                Ok((Opcode::Extended(ext), cur))
            }
            0xFF => Ok((Opcode::EndOfTrack, cur)),
            _ => Ok((Opcode::Unknown(op), cur)),
        }
    } else {
        // AKAO V1 / V2a / V2b high-range opcodes:
        // - 0xFC + sub: extended opcode prefix
        // - 0xFF: end of track
        match op {
            0xFC => {
                let ext = decode_extended(data, &mut cur, version)?;
                Ok((Opcode::Extended(ext), cur))
            }
            0xFF => Ok((Opcode::EndOfTrack, cur)),
            _ => Ok((Opcode::Unknown(op), cur)),
        }
    }
}

fn decode_extended(
    data: &[u8],
    cur: &mut usize,
    version: AkaoVersion,
) -> Result<ExtendedOpcode, Error> {
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
        0x0E if version.supports_subroutines() => {
            ExtendedOpcode::SubroutineCall(read_i16_le(data, cur)?)
        }
        0x0F if version.supports_subroutines() => ExtendedOpcode::SubroutineReturn,
        other => ExtendedOpcode::Unknown(other),
    };
    Ok(ext)
}

#[inline]
fn read_u8(data: &[u8], cur: &mut usize) -> Result<u8, Error> {
    let b = *data
        .get(*cur)
        .ok_or(Error::UnexpectedEof("AKAO opcode operand"))?;
    *cur += 1;
    Ok(b)
}

#[inline]
fn read_u16_le(data: &[u8], cur: &mut usize) -> Result<u16, Error> {
    let lo = read_u8(data, cur)?;
    let hi = read_u8(data, cur)?;
    Ok(u16::from_le_bytes([lo, hi]))
}

#[inline]
fn read_i16_le(data: &[u8], cur: &mut usize) -> Result<i16, Error> {
    Ok(read_u16_le(data, cur)? as i16)
}
