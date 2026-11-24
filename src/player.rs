// SPDX-License-Identifier: MIT OR Apache-2.0

//! AKAO sequencer playback engine driving `libspu2::Device::new_ps1()`.

use alloc::vec::Vec;
use core::fmt;

use libspu2::registers::{ps1, CONTROL_ENABLE, CONTROL_REVERB, CONTROL_UNMUTE};
use libspu2::{Device, StereoI16, VoiceState, PS1_RAM_SIZE, PS1_SAMPLE_RATE, VOICES_PER_CORE};

use crate::bank::AkaoWaveBank;
use crate::opcodes::{decode_opcode, ExtendedOpcode, Opcode};
use crate::reverb::{ReverbParams, ReverbPreset};
use crate::sequence::AkaoSequence;
use crate::{Error, Result};

/// Maximum logical sequencer channels supported (`V3` = 32, `V1`/`V2` = 24).
const MAX_CHANNELS: usize = 32;

/// Default 16.16 fixed-point tempo (~125 BPM at 240 Hz / 48 PPQN).
const DEFAULT_TEMPO_FP16: u32 = 0x6AAA_0000;

/// Full per-sequence volume in 16.16 fixed-point (`127.0` = `0x7F_0000`).
const FULL_SEQUENCE_VOLUME_FP16: u32 = 0x007F_0000;

/// Maximum opcode commands executed on a single channel in one tick (guards against zero-delay infinite loops).
const MAX_COMMANDS_PER_TICK: usize = 256;

/// 12-TET pitch table for octave 4 (`C4 = 0x1000` unity PS1 SPU pitch).
const PITCH_TABLE_OCT4: [u16; 12] = [
    0x1000, // C
    0x10F4, // C#
    0x11F6, // D
    0x1307, // D#
    0x1429, // E
    0x155C, // F
    0x16A1, // F#
    0x17FA, // G
    0x1966, // G#
    0x1AE9, // A
    0x1C82, // A#
    0x1E35, // B
];

/// 32-step signed sine LFO table (`-127..=127`).
const LFO_SINE_TABLE: [i8; 32] = [
    0, 25, 49, 71, 90, 106, 117, 125, 127, 125, 117, 106, 90, 71, 49, 25, 0, -25, -49, -71, -90,
    -106, -117, -125, -127, -125, -117, -106, -90, -71, -49, -25,
];

/// 128-entry constant-power quarter-sine pan LUT (`round(127 * sin(i * pi / 254))`).
///
/// Right gain = `PAN_SIN_LUT[pan]`, Left gain = `PAN_SIN_LUT[127 - pan]`.
/// Satisfies `L^2 + R^2 ≈ 127^2` across the stereo field (center `64` ≈ `90` = `-3 dB`).
const PAN_SIN_LUT: [u8; 128] = [
    0, 2, 3, 5, 6, 8, 9, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 27, 28, 30, 31, 33, 34, 36, 37,
    39, 40, 42, 43, 45, 46, 48, 49, 51, 52, 53, 55, 56, 58, 59, 60, 62, 63, 65, 66, 67, 69, 70, 71,
    72, 74, 75, 76, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 95, 96, 97, 98, 99,
    100, 101, 101, 102, 103, 104, 105, 106, 107, 108, 108, 109, 110, 111, 111, 112, 113, 114, 114,
    115, 116, 116, 117, 117, 118, 119, 119, 120, 120, 121, 121, 122, 122, 122, 123, 123, 124, 124,
    124, 125, 125, 125, 125, 126, 126, 126, 126, 126, 127, 127, 127, 127, 127, 127, 127,
];

/// Fade-out curve applied to the per-sequence volume scalar.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum FadeCurve {
    /// Hardware-faithful linear ramp (`1.0 -> 0.0`), matching PS1 AKAO `60 Hz` fixed-point fades.
    Linear,
    /// Smooth half-cosine (`0.5 * (1 + cos(pi * t))`) fade curve.
    Cosine,
}

/// Configuration for [`AkaoPlayer`] rendering and hardware-faithful BGM fade-out.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RenderConfig {
    /// Number of full sequence loops to play before initiating fade-out (`0` disables loop-triggered fade).
    pub target_loops: u32,
    /// Duration of the per-sequence voice-volume fade-out in seconds (default `8.0`).
    pub fade_seconds: f64,
    /// Post-fade / post-halt release and SPU reverb decay tail in seconds (default `2.0`).
    pub tail_seconds: f64,
    /// Hard safety ceiling on total rendered duration in seconds (default `300.0`).
    pub max_seconds: f64,
    /// Fade-out curve (default [`FadeCurve::Linear`]).
    pub fade_curve: FadeCurve,
    /// Output PCM sample rate in Hz (default `44100`, PS1 SPU native rate).
    pub sample_rate: u32,
    /// Whether SPU reverb master write is enabled (default `true`).
    pub reverb_enabled: bool,
    /// Fixed PS1 SPU master volume (`MVOLL`/`MVOLR`, default `0x3FFF`). Never modified by fade-out.
    pub master_volume: u16,
}

impl RenderConfig {
    /// Returns the standard BGM playback configuration (2 loops, 8 s linear fade, 2 s tail, 44.1 kHz).
    #[inline]
    #[must_use]
    pub const fn bgm_default() -> Self {
        Self {
            target_loops: 2,
            fade_seconds: 8.0,
            tail_seconds: 2.0,
            max_seconds: 300.0,
            fade_curve: FadeCurve::Linear,
            sample_rate: PS1_SAMPLE_RATE,
            reverb_enabled: true,
            master_volume: 0x3FFF,
        }
    }
}

impl Default for RenderConfig {
    fn default() -> Self {
        Self::bgm_default()
    }
}

/// Summary and interleaved stereo PCM returned by [`AkaoPlayer::render_all`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderResult {
    /// Rendered interleaved stereo 16-bit PCM (`[L0, R0, L1, R1, ...]`).
    pub samples: Vec<i16>,
    /// Total number of stereo frames rendered (`samples.len() / 2`).
    pub frames_rendered: u64,
    /// Number of completed sequence loops observed during playback.
    pub loops_completed: u32,
    /// `true` if playback completed a fade-out cycle.
    pub faded_out: bool,
    /// `true` if all channels halted naturally before hitting `max_seconds`.
    pub reached_end: bool,
}

/// Read-only runtime snapshot of an AKAO sequencer channel.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChannelSnapshot {
    /// Whether this channel is currently active.
    pub active: bool,
    /// Whether this channel is currently muted (via mute mask or solo mask).
    pub muted: bool,
    /// Current program counter offset into [`AkaoSequence::data`].
    pub pc: usize,
    /// Mapped PS1 SPU hardware voice index (`0..24`), or `usize::MAX` if unassigned.
    pub voice: usize,
    /// Active instrument / articulation index.
    pub instrument: u8,
    /// Current octave (`0..=10`).
    pub octave: u8,
    /// Current semitone transpose.
    pub transpose: i8,
    /// Current channel volume (`0..=127`).
    pub volume: u8,
    /// Current channel stereo pan (`0..=127`).
    pub pan: u8,
    /// Most recently computed PS1 SPU voice pitch register (`0..=0x3FFF`).
    pub pitch: u16,
    /// Current 4-deep loop stack depth (`0..=4`).
    pub loop_depth: usize,
    /// Whether drum/percussion mapping mode is enabled on this channel.
    pub drum_mode: bool,
    /// Current PS1 SPU `ADSR1` register value for this channel.
    pub adsr1: u16,
    /// Current PS1 SPU `ADSR2` register value for this channel.
    pub adsr2: u16,
    /// Whether SPU reverb send (`EON`) is enabled for this channel.
    pub reverb: bool,
    /// Whether SPU pitch modulation (`PMON`) is enabled for this channel.
    pub pmon: bool,
    /// Whether SPU hardware noise (`NON`) is enabled for this channel.
    pub noise: bool,
    /// Whether right-channel phase inversion is active.
    pub phase_invert: bool,
    /// Live hardware voice state from `libspu2`, if a hardware voice is currently assigned.
    pub voice_state: Option<VoiceState>,
}

#[derive(Debug, Clone, Copy, Default)]
struct LoopFrame {
    start_pc: usize,
    iterations: u16,
}

#[derive(Debug, Clone, Copy, Default)]
struct LfoState {
    enabled: bool,
    delay: u16,
    delay_remaining: u16,
    /// Full-cycle period in musical ticks (`1..=256`).
    period: u16,
    /// 16-bit phase accumulator (`phase_fp16 >> 11` selects 0..31 waveform step).
    phase_fp16: u16,
    /// Phase increment per musical tick (`0x10000 / period`).
    phase_step: u16,
    depth: u8,
    waveform: u8,
    current_value: i16,
}

impl LfoState {
    fn configure(&mut self, delay: u8, rate: u8, waveform: u8) {
        self.enabled = true;
        self.delay = u16::from(delay);
        self.period = if rate == 0 { 256 } else { u16::from(rate) };
        self.phase_step = (0x10000u32 / u32::from(self.period)).clamp(1, 0xFFFF) as u16;
        self.waveform = waveform;
        self.reset_note();
    }

    fn reset_note(&mut self) {
        self.delay_remaining = self.delay;
        self.phase_fp16 = 0;
        self.current_value = 0;
    }

    fn tick_vibrato(&mut self) -> bool {
        if !self.enabled || (self.depth & 0x7F) == 0 {
            if self.current_value != 0 {
                self.current_value = 0;
                return true;
            }
            return false;
        }
        if self.delay_remaining > 0 {
            self.delay_remaining -= 1;
            return false;
        }
        let step_idx = (self.phase_fp16 >> 11) as u8;
        self.phase_fp16 = self.phase_fp16.wrapping_add(self.phase_step);
        let sample = lfo_sample(self.waveform, step_idx);
        // Hardware AKAO vibrato depth bit 7 (`0x80`):
        // - bit 7 == 0: fine range (`depth & 0x7F`, subtle pitch vibrato)
        // - bit 7 == 1: coarse range (`(depth & 0x7F) * 4`, wide vibrato)
        let mag = i32::from(self.depth & 0x7F);
        let scaled = if (self.depth & 0x80) != 0 {
            ((i32::from(sample) * mag) >> 5) as i16
        } else {
            ((i32::from(sample) * mag) >> 7) as i16
        };
        if scaled != self.current_value {
            self.current_value = scaled;
            true
        } else {
            false
        }
    }

    fn tick_linear(&mut self) -> bool {
        if !self.enabled || self.depth == 0 {
            if self.current_value != 0 {
                self.current_value = 0;
                return true;
            }
            return false;
        }
        if self.delay_remaining > 0 {
            self.delay_remaining -= 1;
            return false;
        }
        let step_idx = (self.phase_fp16 >> 11) as u8;
        self.phase_fp16 = self.phase_fp16.wrapping_add(self.phase_step);
        let sample = lfo_sample(self.waveform, step_idx);
        let scaled = ((i32::from(sample) * i32::from(self.depth)) >> 7) as i16;
        if scaled != self.current_value {
            self.current_value = scaled;
            true
        } else {
            false
        }
    }
}

fn lfo_sample(waveform: u8, phase: u8) -> i8 {
    let idx = (phase & 31) as usize;
    match waveform & 3 {
        0 => LFO_SINE_TABLE[idx],
        1 => {
            // Triangle wave (-120..120)
            let v: i16 = if idx < 8 {
                (idx as i16) * 15
            } else if idx < 24 {
                120 - ((idx - 8) as i16) * 15
            } else {
                -120 + ((idx - 24) as i16) * 15
            };
            v.clamp(-127, 127) as i8
        }
        2 => {
            // Square wave
            if idx < 16 {
                127
            } else {
                -127
            }
        }
        _ => {
            // Sawtooth wave
            (((idx as i16) * 8) - 124).clamp(-127, 127) as i8
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct ChannelState {
    active: bool,
    pc: usize,
    /// Assigned PS1 SPU hardware voice (`0..24`), or `None` if unassigned / stolen.
    voice: Option<u8>,
    /// Channel's selected instrument index (`0xA1`).
    instrument: u8,
    /// Resolved articulation index for the currently sounding note (differs from `instrument` in drum/keysplit mode).
    resolved_instrument: u8,
    octave: u8,
    transpose: i8,
    fine_tune: i8,
    /// Channel volume in 8.8 fixed point (`0x7F00` = 127.0).
    volume_fp8: i32,
    vol_slide_step: i32,
    vol_slide_ticks: u16,
    /// Channel master volume (`0..=127`).
    master_volume: u8,
    /// Channel expression / balance (`0..=127`).
    volume_balance: u8,
    /// Per-note volume override from `V3` `DrumEntry::volume` (`None` for non-drum notes).
    note_volume_override: Option<u8>,
    /// Channel pan in 8.8 fixed point (`0x4000` = center 64.0).
    pan_fp8: i32,
    pan_slide_step: i32,
    pan_slide_ticks: u16,
    /// Per-note pan override from `V3` `DrumEntry::pan` (`None` for non-drum notes).
    note_pan_override: Option<u8>,
    /// Pitch slide offset in 8.8 fixed point semitones.
    pitch_slide_fp8: i32,
    pitch_slide_step: i32,
    pitch_slide_ticks: u16,
    /// Current note semitone (octave * 12 + semi).
    current_note: i16,
    last_pitch: u16,
    /// Remaining ticks of current note/tie/rest.
    duration_remaining: u16,
    /// Remaining gate ticks before key-off (`0` = already keyed off or tied).
    gate_remaining: u16,
    /// One-shot duration override (`0xA2`).
    duration_override: u8,
    /// Fixed note length gate delta (`0xDC`).
    fix_note_length: i8,
    /// 4-deep loop stack (`0xC8` / `0xC9` / `0xCA`).
    loop_stack: [LoopFrame; 4],
    loop_depth: usize,
    /// Single-level `V3` subroutine return address (`0xFE 0x0E` / `0x0F`).
    subroutine_return_pc: Option<usize>,
    /// Number of infinite loops completed by this channel.
    infinite_loops: u32,
    /// ADSR registers (copied from articulation and overrideable).
    adsr1: u16,
    adsr2: u16,
    /// Whether the active note applied a per-note drum/keysplit ADSR override.
    adsr_overridden_by_note: bool,
    /// Mode flags.
    slur: bool,
    legato: bool,
    portamento_ticks: u8,
    drum_mode: bool,
    keysplit_active: bool,
    secondary: bool,
    reverb: bool,
    pmon: bool,
    noise: bool,
    phase_invert: bool,
    /// LFOs.
    vibrato: LfoState,
    tremolo: LfoState,
    autopan: LfoState,
    /// Dirty flags for SPU register updates.
    dirty_volume: bool,
    dirty_pitch: bool,
    dirty_adsr: bool,
}

impl ChannelState {
    const fn empty() -> Self {
        Self {
            active: false,
            pc: 0,
            voice: None,
            instrument: 0,
            resolved_instrument: 0,
            octave: 4,
            transpose: 0,
            fine_tune: 0,
            volume_fp8: 0x7F00,
            vol_slide_step: 0,
            vol_slide_ticks: 0,
            master_volume: 127,
            volume_balance: 127,
            note_volume_override: None,
            pan_fp8: 0x4000,
            pan_slide_step: 0,
            pan_slide_ticks: 0,
            note_pan_override: None,
            pitch_slide_fp8: 0,
            pitch_slide_step: 0,
            pitch_slide_ticks: 0,
            current_note: 48,
            last_pitch: 0x1000,
            duration_remaining: 1,
            gate_remaining: 0,
            duration_override: 0,
            fix_note_length: 0,
            loop_stack: [LoopFrame {
                start_pc: 0,
                iterations: 0,
            }; 4],
            loop_depth: 0,
            subroutine_return_pc: None,
            infinite_loops: 0,
            adsr1: 0x00FF,
            adsr2: 0x0000,
            adsr_overridden_by_note: false,
            slur: false,
            legato: false,
            portamento_ticks: 0,
            drum_mode: false,
            keysplit_active: false,
            secondary: false,
            reverb: false,
            pmon: false,
            noise: false,
            phase_invert: false,
            vibrato: LfoState {
                enabled: false,
                delay: 0,
                delay_remaining: 0,
                period: 1,
                phase_fp16: 0,
                phase_step: 0,
                depth: 0,
                waveform: 0,
                current_value: 0,
            },
            tremolo: LfoState {
                enabled: false,
                delay: 0,
                delay_remaining: 0,
                period: 1,
                phase_fp16: 0,
                phase_step: 0,
                depth: 0,
                waveform: 0,
                current_value: 0,
            },
            autopan: LfoState {
                enabled: false,
                delay: 0,
                delay_remaining: 0,
                period: 1,
                phase_fp16: 0,
                phase_step: 0,
                depth: 0,
                waveform: 0,
                current_value: 0,
            },
            dirty_volume: true,
            dirty_pitch: false,
            dirty_adsr: true,
        }
    }

    fn init(
        voice: Option<u8>,
        start_pc: Option<usize>,
        rhythm: bool,
        secondary: bool,
        default_adsr: (u16, u16),
    ) -> Self {
        let mut s = Self::empty();
        s.active = start_pc.is_some();
        s.pc = start_pc.unwrap_or(0);
        s.voice = voice;
        s.drum_mode = rhythm;
        s.secondary = secondary;
        s.adsr1 = default_adsr.0;
        s.adsr2 = default_adsr.1;
        s
    }

    #[inline]
    fn volume_u8(&self) -> u8 {
        (self.volume_fp8 >> 8).clamp(0, 127) as u8
    }

    #[inline]
    fn pan_u8(&self) -> u8 {
        (self.pan_fp8 >> 8).clamp(0, 127) as u8
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PlayPhase {
    Playing,
    Fading,
    Tail,
    Finished,
}

/// PlayStation 1 AKAO music player driving [`libspu2::Device`] in PS1 SPU mode.
pub struct AkaoPlayer {
    sequence: AkaoSequence,
    bank: AkaoWaveBank,
    config: RenderConfig,
    device: Device,
    /// Fixed-size 32-channel state array (zero heap allocation).
    channels: [ChannelState; MAX_CHANNELS],
    /// Logical channel count for this sequence generation (`24` for `V1`/`V2`, `32` for `V3`).
    channel_count: usize,
    /// 24-voice hardware allocator: maps SPU voice `0..24` -> logical channel index `0..32`.
    voice_owner: [Option<u8>; VOICES_PER_CORE],
    /// Tick timestamp when each hardware voice was last triggered (for voice stealing).
    voice_age: [u64; VOICES_PER_CORE],
    /// Channel mute bitmask (bit `i` = channel `i` muted).
    mute_mask: u32,
    /// Channel solo bitmask (when non-zero, only channels with bit `i` set are audible).
    solo_mask: u32,
    /// Cached hardware voice-mode register masks for dirty-flagged updates.
    last_pmon: u32,
    last_noise: u32,
    last_reverb: u32,
    dirty_voice_modes: bool,
    /// Whether hardware SPU reverb is active (enabled in config, non-Off preset, and fits in SPU RAM without overlapping ADPCM).
    reverb_active: bool,
    /// 16.16 fixed-point tempo.
    tempo_fp16: u32,
    tempo_slide_step: i32,
    tempo_slide_ticks: u16,
    /// 16-bit fractional tempo accumulator.
    tempo_accum: u32,
    /// Global sequence condition byte.
    condition: u8,
    /// SPU reverb return depth (`EVOLL`/`EVOLR`) in 16.16 fixed point.
    reverb_depth_l_fp16: i32,
    reverb_depth_r_fp16: i32,
    reverb_slide_step: i32,
    reverb_slide_ticks: u16,
    /// PS1 SPU noise clock (`0..=63`, written to `SPUCNT` bits 8..=13).
    noise_clock: u8,
    /// Hardware timer tick counter (243.86 Hz on `V1`, 240.00 Hz on `V2`/`V3`).
    timer_ticks: u64,
    /// Exact rational sample accumulator numerator (denominator = `4_233_600` Hz RCnt2 clock).
    sample_accum_num: u64,
    /// Numerator added per Root Counter 2 tick (`sample_rate * timer_divider`).
    samples_per_tick_num: u64,
    /// Denominator of the Root Counter 2 sample clock (`4_233_600`).
    samples_per_tick_den: u64,
    /// Initial per-sequence master volume in 16.16 fixed point (from `sequence.initial_volume`).
    initial_sequence_volume_fp16: u32,
    /// Current per-sequence master volume in 16.16 fixed point (`0..=0x7F_0000`).
    /// Ramped linearly at 60 Hz during fade-out, scaling `V_VOLL`/`V_VOLR` (never `MVOLL`/`MVOLR`).
    sequence_volume_fp16: u32,
    /// Fixed-point 60 Hz linear fade step (`saturating_sub` per 60 Hz tick).
    fade_step_fp16: u32,
    fade_elapsed_60hz: u32,
    fade_total_60hz: u32,
    tail_frames_remaining: u64,
    max_frames: u64,
    frames_rendered: u64,
    loops_completed: u32,
    faded_out: bool,
    reached_end: bool,
    phase: PlayPhase,
}

impl fmt::Debug for AkaoPlayer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("AkaoPlayer")
            .field("version", &self.sequence.version)
            .field("song_id", &self.sequence.song_id)
            .field("channel_count", &self.channel_count)
            .field("phase", &self.phase)
            .field("tempo", &self.tempo())
            .field("sequence_volume", &self.sequence_volume())
            .field("frames_rendered", &self.frames_rendered)
            .field("loops_completed", &self.loops_completed)
            .field("mute_mask", &self.mute_mask)
            .field("solo_mask", &self.solo_mask)
            .field("reverb_active", &self.reverb_active)
            .finish()
    }
}

impl AkaoPlayer {
    /// Creates a new [`AkaoPlayer`], uploads `bank` to a fresh [`Device::new_ps1`],
    /// programs the PS1 SPU reverb preset registers, and initializes all active channels.
    pub fn new(sequence: AkaoSequence, bank: AkaoWaveBank, config: RenderConfig) -> Result<Self> {
        if config.sample_rate == 0 {
            return Err(Error::InvalidArgument("sample_rate must be non-zero"));
        }
        if config.fade_seconds < 0.0 || !config.fade_seconds.is_finite() {
            return Err(Error::InvalidArgument(
                "fade_seconds must be non-negative and finite",
            ));
        }
        if config.tail_seconds < 0.0 || !config.tail_seconds.is_finite() {
            return Err(Error::InvalidArgument(
                "tail_seconds must be non-negative and finite",
            ));
        }
        if config.max_seconds <= 0.0 || !config.max_seconds.is_finite() {
            return Err(Error::InvalidArgument(
                "max_seconds must be positive and finite",
            ));
        }

        let mut device = Device::new_ps1();
        if config.sample_rate != PS1_SAMPLE_RATE {
            device.set_output_sample_rate(config.sample_rate)?;
        }
        bank.upload_to_spu(&mut device)?;

        let channel_count = sequence.version.max_channels().min(MAX_CHANNELS);
        let samples_per_tick_num =
            u64::from(config.sample_rate) * u64::from(sequence.version.timer_divider());
        let samples_per_tick_den = 4_233_600u64;
        let fade_total_60hz = ((config.fade_seconds * 60.0) + 0.5) as u32;
        let max_frames = (config.max_seconds * f64::from(config.sample_rate)) as u64;

        let init_seq_vol_fp16 = if sequence.initial_volume > 0 {
            (sequence.initial_volume.min(127)) << 16
        } else {
            FULL_SEQUENCE_VOLUME_FP16
        };
        let fade_step_fp16 = init_seq_vol_fp16
            .checked_div(fade_total_60hz)
            .map_or(init_seq_vol_fp16, |step| step.max(1));

        let mut player = Self {
            sequence,
            bank,
            config,
            device,
            channels: [ChannelState::empty(); MAX_CHANNELS],
            channel_count,
            voice_owner: [None; VOICES_PER_CORE],
            voice_age: [0; VOICES_PER_CORE],
            mute_mask: 0,
            solo_mask: 0,
            last_pmon: u32::MAX,
            last_noise: u32::MAX,
            last_reverb: u32::MAX,
            dirty_voice_modes: true,
            reverb_active: false,
            tempo_fp16: DEFAULT_TEMPO_FP16,
            tempo_slide_step: 0,
            tempo_slide_ticks: 0,
            tempo_accum: 0,
            condition: 0,
            reverb_depth_l_fp16: 0,
            reverb_depth_r_fp16: 0,
            reverb_slide_step: 0,
            reverb_slide_ticks: 0,
            noise_clock: 0,
            timer_ticks: 0,
            sample_accum_num: 0,
            samples_per_tick_num,
            samples_per_tick_den,
            initial_sequence_volume_fp16: init_seq_vol_fp16,
            sequence_volume_fp16: init_seq_vol_fp16,
            fade_step_fp16,
            fade_elapsed_60hz: 0,
            fade_total_60hz,
            tail_frames_remaining: 0,
            max_frames,
            frames_rendered: 0,
            loops_completed: 0,
            faded_out: false,
            reached_end: false,
            phase: PlayPhase::Playing,
        };

        player.init_spu_and_channels()?;
        Ok(player)
    }

    /// Resets playback state, channels, voice allocations, and SPU registers back to
    /// the start of the sequence **without** re-uploading ADPCM wave-bank RAM.
    pub fn reset(&mut self) -> Result<()> {
        // Silence and reset all 24 hardware voices without wiping SPU RAM.
        self.device.write_ps1_register(ps1::KOFF_LO, 0xFFFF)?;
        self.device.write_ps1_register(ps1::KOFF_HI, 0x00FF)?;
        for v in 0..VOICES_PER_CORE {
            let vbase = (v as u32) * ps1::VOICE_STRIDE;
            self.device.write_ps1_register(vbase + ps1::V_VOLL, 0)?;
            self.device.write_ps1_register(vbase + ps1::V_VOLR, 0)?;
            self.device.write_ps1_register(vbase + ps1::V_PITCH, 0)?;
            self.device.write_ps1_register(vbase + ps1::V_ADSR1, 0)?;
            self.device.write_ps1_register(vbase + ps1::V_ADSR2, 0)?;
        }
        self.device.write_ps1_register(ps1::PMON_LO, 0)?;
        self.device.write_ps1_register(ps1::PMON_HI, 0)?;
        self.device.write_ps1_register(ps1::NON_LO, 0)?;
        self.device.write_ps1_register(ps1::NON_HI, 0)?;
        self.device.write_ps1_register(ps1::EON_LO, 0)?;
        self.device.write_ps1_register(ps1::EON_HI, 0)?;

        self.tempo_fp16 = DEFAULT_TEMPO_FP16;
        self.tempo_slide_step = 0;
        self.tempo_slide_ticks = 0;
        self.tempo_accum = 0;
        self.condition = 0;
        self.reverb_slide_step = 0;
        self.reverb_slide_ticks = 0;
        self.noise_clock = 0;
        self.timer_ticks = 0;
        self.sample_accum_num = 0;
        self.sequence_volume_fp16 = self.initial_sequence_volume_fp16;
        self.fade_elapsed_60hz = 0;
        self.tail_frames_remaining = 0;
        self.frames_rendered = 0;
        self.loops_completed = 0;
        self.faded_out = false;
        self.reached_end = false;
        self.phase = PlayPhase::Playing;
        self.last_pmon = u32::MAX;
        self.last_noise = u32::MAX;
        self.last_reverb = u32::MAX;
        self.dirty_voice_modes = true;

        self.init_spu_and_channels()
    }

    fn init_spu_and_channels(&mut self) -> Result<()> {
        // Program PS1 SPU reverb preset registers, guarding RVB_WORK_ADDR against ADPCM overlap.
        let preset = ReverbPreset::from_index((self.sequence.reverb_type & 0xFF) as u8)
            .unwrap_or(ReverbPreset::Off);
        let work_size = preset.work_area_size() as usize;
        let bank_end =
            (self.bank.spu_base_addr as usize).saturating_add(self.bank.adpcm_data.len());
        let work_start = PS1_RAM_SIZE.saturating_sub(work_size) & !7;

        self.reverb_active = self.config.reverb_enabled
            && preset != ReverbPreset::Off
            && work_size > 0
            && work_start >= bank_end;

        let params = if self.reverb_active {
            preset.params()
        } else {
            ReverbParams::ZERO
        };

        let regs = params.to_ps1_registers();
        for (idx, &word) in regs.iter().enumerate() {
            self.device
                .write_ps1_register(ps1::RVB_BASE + (idx as u32) * 2, word)?;
        }

        let work_addr_reg = if self.reverb_active {
            (work_start >> 3) as u16
        } else {
            // Safe top-of-RAM default when reverb write is disabled.
            ((PS1_RAM_SIZE - 8) >> 3) as u16
        };
        self.device
            .write_ps1_register(ps1::RVB_WORK_ADDR, work_addr_reg)?;

        self.update_spucnt()?;
        self.device
            .write_ps1_register(ps1::MVOLL, self.config.master_volume & 0x7FFF)?;
        self.device
            .write_ps1_register(ps1::MVOLR, self.config.master_volume & 0x7FFF)?;

        let init_rvb = if self.reverb_active { 0x2000i16 } else { 0 };
        self.reverb_depth_l_fp16 = i32::from(init_rvb) << 16;
        self.reverb_depth_r_fp16 = i32::from(init_rvb) << 16;
        self.device
            .write_ps1_register(ps1::EVOLL, init_rvb as u16)?;
        self.device
            .write_ps1_register(ps1::EVOLR, init_rvb as u16)?;

        // Initialize voice allocator and channel states.
        self.voice_owner = [None; VOICES_PER_CORE];
        self.voice_age = [0; VOICES_PER_CORE];
        self.channels = [ChannelState::empty(); MAX_CHANNELS];

        let default_adsr = self
            .bank
            .articulation(0)
            .map_or((0x00FF, 0x0000), |a| (a.adsr1, a.adsr2));

        // First pass: assign 1:1 hardware voices for active channels in 0..24.
        for ch in 0..self.channel_count {
            let start = self.sequence.channel_start(ch);
            let rhythm = (self.sequence.rhythm_mask & (1u32 << ch)) != 0;
            let secondary = (self.sequence.secondary_mask & (1u32 << ch)) != 0;
            let voice = if start.is_some() && ch < VOICES_PER_CORE {
                self.voice_owner[ch] = Some(ch as u8);
                Some(ch as u8)
            } else {
                None
            };
            self.channels[ch] = ChannelState::init(voice, start, rhythm, secondary, default_adsr);
        }

        // Second pass: assign any free hardware voices to active upper V3 channels (24..32).
        for ch in VOICES_PER_CORE..self.channel_count {
            if self.channels[ch].active {
                if let Some(free_v) = self.voice_owner.iter().position(|o| o.is_none()) {
                    self.voice_owner[free_v] = Some(ch as u8);
                    self.channels[ch].voice = Some(free_v as u8);
                }
            }
        }

        Ok(())
    }

    /// Returns a read-only reference to the underlying [`libspu2::Device`].
    #[inline]
    #[must_use]
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Returns a read-only reference to the loaded [`AkaoSequence`].
    #[inline]
    #[must_use]
    pub fn sequence(&self) -> &AkaoSequence {
        &self.sequence
    }

    /// Returns a read-only reference to the loaded [`AkaoWaveBank`].
    #[inline]
    #[must_use]
    pub fn bank(&self) -> &AkaoWaveBank {
        &self.bank
    }

    /// Returns the active [`RenderConfig`].
    #[inline]
    #[must_use]
    pub fn config(&self) -> RenderConfig {
        self.config
    }

    /// Returns the logical channel count for this sequence (`24` for `V1`/`V2`, `32` for `V3`).
    #[inline]
    #[must_use]
    pub const fn channel_count(&self) -> usize {
        self.channel_count
    }

    /// Returns `true` once playback (including fade-out and tail) has finished.
    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.phase == PlayPhase::Finished
    }

    /// Returns `true` while the player is actively ramping down during fade-out.
    #[inline]
    #[must_use]
    pub fn is_fading(&self) -> bool {
        self.phase == PlayPhase::Fading
    }

    /// Returns the elapsed playback time in seconds (`frames_rendered / sample_rate`).
    #[inline]
    #[must_use]
    pub fn elapsed_seconds(&self) -> f64 {
        (self.frames_rendered as f64) / f64::from(self.config.sample_rate)
    }

    /// Returns the current per-sequence volume scalar in `0..=127`.
    #[inline]
    #[must_use]
    pub fn sequence_volume(&self) -> u8 {
        (self.sequence_volume_fp16 >> 16).min(127) as u8
    }

    /// Returns the current 16-bit tempo value (`tempo_fp16 >> 16`).
    #[inline]
    #[must_use]
    pub fn tempo(&self) -> u16 {
        (self.tempo_fp16 >> 16) as u16
    }

    /// Returns the number of completed sequence loops observed so far.
    #[inline]
    #[must_use]
    pub fn loops_completed(&self) -> u32 {
        self.loops_completed
    }

    /// Returns the cumulative number of stereo frames rendered so far.
    #[inline]
    #[must_use]
    pub fn frames_rendered(&self) -> u64 {
        self.frames_rendered
    }

    /// Mutes or unmutes a single logical channel (`0..channel_count()`).
    pub fn set_channel_muted(&mut self, channel: usize, muted: bool) {
        if channel >= self.channel_count {
            return;
        }
        let bit = 1u32 << channel;
        if muted {
            self.mute_mask |= bit;
        } else {
            self.mute_mask &= !bit;
        }
        self.channels[channel].dirty_volume = true;
        let _ = self.flush_dirty_voices();
    }

    /// Replaces the channel mute bitmask (bit `i` = channel `i` muted).
    pub fn set_mute_mask(&mut self, mask: u32) {
        self.mute_mask = mask;
        for ch in &mut self.channels[..self.channel_count] {
            ch.dirty_volume = true;
        }
        let _ = self.flush_dirty_voices();
    }

    /// Returns the current channel mute bitmask.
    #[inline]
    #[must_use]
    pub const fn mute_mask(&self) -> u32 {
        self.mute_mask
    }

    /// Sets the channel solo bitmask (`0` disables solo; non-zero mutes all channels not in `mask`).
    pub fn set_solo_mask(&mut self, mask: u32) {
        self.solo_mask = mask;
        for ch in &mut self.channels[..self.channel_count] {
            ch.dirty_volume = true;
        }
        let _ = self.flush_dirty_voices();
    }

    /// Returns the current channel solo bitmask.
    #[inline]
    #[must_use]
    pub const fn solo_mask(&self) -> u32 {
        self.solo_mask
    }

    #[inline]
    const fn is_channel_effectively_muted(&self, channel: usize) -> bool {
        let bit = 1u32 << channel;
        if (self.mute_mask & bit) != 0 {
            return true;
        }
        self.solo_mask != 0 && (self.solo_mask & bit) == 0
    }

    /// Returns a snapshot of `channel`'s current state, or `None` if out of range.
    #[must_use]
    pub fn channel_snapshot(&self, channel: usize) -> Option<ChannelSnapshot> {
        if channel >= self.channel_count {
            return None;
        }
        let ch = &self.channels[channel];
        let voice_state = ch
            .voice
            .and_then(|v| self.device.inspect_voice(0, usize::from(v)).ok());
        Some(ChannelSnapshot {
            active: ch.active,
            muted: self.is_channel_effectively_muted(channel),
            pc: ch.pc,
            voice: ch.voice.map_or(usize::MAX, usize::from),
            instrument: ch.instrument,
            octave: ch.octave,
            transpose: ch.transpose,
            volume: ch.volume_u8(),
            pan: ch.pan_u8(),
            pitch: ch.last_pitch,
            loop_depth: ch.loop_depth,
            drum_mode: ch.drum_mode,
            adsr1: ch.adsr1,
            adsr2: ch.adsr2,
            reverb: ch.reverb,
            pmon: ch.pmon,
            noise: ch.noise,
            phase_invert: ch.phase_invert,
            voice_state,
        })
    }

    /// Manually triggers a per-sequence fade-out over [`RenderConfig::fade_seconds`].
    pub fn start_fade_out(&mut self) {
        if self.phase == PlayPhase::Playing {
            if self.fade_total_60hz == 0 {
                self.sequence_volume_fp16 = 0;
                self.faded_out = true;
                self.enter_tail();
            } else {
                self.phase = PlayPhase::Fading;
                self.fade_elapsed_60hz = 0;
            }
        }
    }

    /// Renders interleaved stereo `i16` PCM into `out`, returning the number of
    /// stereo frames rendered (up to `out.len() / 2`).
    ///
    /// Returns [`Error::InvalidArgument`] if `out.len()` is odd.
    pub fn render_samples(&mut self, out: &mut [i16]) -> Result<usize> {
        if out.len() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "interleaved stereo buffer length must be even",
            ));
        }
        let total_frames = out.len() / 2;
        let mut rendered = 0usize;

        while rendered < total_frames && self.phase != PlayPhase::Finished {
            let to_render = self.advance_until_renderable(total_frames - rendered)?;
            if to_render == 0 {
                break;
            }
            let start = rendered * 2;
            let end = (rendered + to_render) * 2;
            self.device.render_native(&mut out[start..end]);
            rendered += to_render;
            self.commit_rendered_frames(to_render);
        }

        Ok(rendered)
    }

    /// Renders stereo frames into `out`, returning the number of frames rendered.
    pub fn render_frames(&mut self, out: &mut [StereoI16]) -> Result<usize> {
        let mut chunk = [0i16; 512];
        let mut written = 0usize;
        while written < out.len() && !self.is_finished() {
            let n = (out.len() - written).min(256);
            let got = self.render_samples(&mut chunk[..n * 2])?;
            if got == 0 {
                break;
            }
            for i in 0..got {
                out[written + i] = StereoI16 {
                    left: chunk[i * 2],
                    right: chunk[i * 2 + 1],
                };
            }
            written += got;
        }
        Ok(written)
    }

    /// Renders planar stereo `f32` PCM (`[-1.0, 1.0]`) directly into `left` and `right`
    /// with zero heap allocation, returning the number of stereo frames rendered.
    ///
    /// Returns [`Error::InvalidArgument`] if `left.len() != right.len()`.
    pub fn render_f32_planar(&mut self, left: &mut [f32], right: &mut [f32]) -> Result<usize> {
        if left.len() != right.len() {
            return Err(Error::InvalidArgument(
                "planar left and right buffers must have equal length",
            ));
        }
        let total_frames = left.len();
        let mut rendered = 0usize;

        while rendered < total_frames && self.phase != PlayPhase::Finished {
            let to_render = self.advance_until_renderable(total_frames - rendered)?;
            if to_render == 0 {
                break;
            }
            let end = rendered + to_render;
            self.device
                .render_f32_planar(&mut left[rendered..end], &mut right[rendered..end])?;
            rendered += to_render;
            self.commit_rendered_frames(to_render);
        }

        Ok(rendered)
    }

    /// Renders interleaved stereo `f32` PCM (`[L0, R0, L1, R1, ...]`) into `out`
    /// with zero heap allocation, returning the number of stereo frames rendered.
    ///
    /// Returns [`Error::InvalidArgument`] if `out.len()` is odd.
    pub fn render_f32_interleaved(&mut self, out: &mut [f32]) -> Result<usize> {
        if out.len() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "interleaved stereo buffer length must be even",
            ));
        }
        let total_frames = out.len() / 2;
        let mut rendered = 0usize;
        let mut left_buf = [0.0f32; 256];
        let mut right_buf = [0.0f32; 256];

        while rendered < total_frames && self.phase != PlayPhase::Finished {
            let n = (total_frames - rendered).min(256);
            let got = self.render_f32_planar(&mut left_buf[..n], &mut right_buf[..n])?;
            if got == 0 {
                break;
            }
            for i in 0..got {
                let dst = (rendered + i) * 2;
                out[dst] = left_buf[i];
                out[dst + 1] = right_buf[i];
            }
            rendered += got;
        }

        Ok(rendered)
    }

    /// Renders the entire sequence (respecting `target_loops`, `fade_seconds`,
    /// `tail_seconds`, and `max_seconds`) into a [`RenderResult`].
    pub fn render_all(&mut self) -> Result<RenderResult> {
        let mut samples = Vec::new();
        let mut buf = [0i16; 2048];
        while !self.is_finished() {
            let frames = self.render_samples(&mut buf)?;
            if frames == 0 {
                break;
            }
            samples.extend_from_slice(&buf[..frames * 2]);
        }
        Ok(RenderResult {
            frames_rendered: self.frames_rendered,
            samples,
            loops_completed: self.loops_completed,
            faded_out: self.faded_out,
            reached_end: self.reached_end,
        })
    }

    /// Advances sequencer timers until at least 1 frame is ready to render, or playback ends.
    fn advance_until_renderable(&mut self, max_wanted: usize) -> Result<usize> {
        loop {
            if self.phase == PlayPhase::Finished || max_wanted == 0 {
                return Ok(0);
            }
            if self.frames_rendered >= self.max_frames {
                self.phase = PlayPhase::Finished;
                return Ok(0);
            }

            let remaining_cap = (self.max_frames - self.frames_rendered) as usize;
            if self.phase == PlayPhase::Tail {
                let to_render = max_wanted
                    .min(self.tail_frames_remaining as usize)
                    .min(remaining_cap);
                if to_render == 0 {
                    self.phase = PlayPhase::Finished;
                }
                return Ok(to_render);
            }

            // Step hardware Root Counter 2 timer as many times as needed until >= 1 full sample is due.
            // Using `while` prevents underflow panics at very low sample rates (<244 Hz).
            while self.sample_accum_num < self.samples_per_tick_den {
                self.step_hardware_timer()?;
                if self.phase == PlayPhase::Tail || self.phase == PlayPhase::Finished {
                    break;
                }
                self.sample_accum_num += self.samples_per_tick_num;
            }

            if self.phase == PlayPhase::Tail || self.phase == PlayPhase::Finished {
                continue;
            }

            let step_frames = (self.sample_accum_num / self.samples_per_tick_den) as usize;
            return Ok(step_frames.min(max_wanted).min(remaining_cap));
        }
    }

    fn commit_rendered_frames(&mut self, frames: usize) {
        self.frames_rendered += frames as u64;
        if self.phase == PlayPhase::Tail {
            self.tail_frames_remaining = self.tail_frames_remaining.saturating_sub(frames as u64);
            if self.tail_frames_remaining == 0 {
                self.phase = PlayPhase::Finished;
            }
        } else {
            self.sample_accum_num -= (frames as u64) * self.samples_per_tick_den;
        }
    }

    fn step_hardware_timer(&mut self) -> Result<()> {
        // 60 Hz fade-out tick (every 4th Root Counter 2 tick, matching AKAO hardware `(tick & 3) == 0`).
        if self.phase == PlayPhase::Fading && (self.timer_ticks & 3) == 0 {
            self.step_fade_60hz()?;
            if self.phase != PlayPhase::Fading {
                self.timer_ticks += 1;
                return Ok(());
            }
        }

        // 16.16 fixed-point tempo accumulator: `accum += tempo >> 16`.
        let step = self.tempo_fp16 >> 16;
        self.tempo_accum = self.tempo_accum.wrapping_add(step);
        if (self.tempo_accum & 0xFFFF_0000) != 0 {
            self.tempo_accum &= 0xFFFF;
            self.step_musical_tick()?;
        }

        self.timer_ticks += 1;
        Ok(())
    }

    fn step_fade_60hz(&mut self) -> Result<()> {
        self.fade_elapsed_60hz += 1;
        let old_int = self.sequence_volume_fp16 >> 16;

        if self.fade_elapsed_60hz >= self.fade_total_60hz {
            self.sequence_volume_fp16 = 0;
        } else {
            match self.config.fade_curve {
                FadeCurve::Linear => {
                    // Integer fixed-point 60 Hz linear fade (zero floating-point math).
                    self.sequence_volume_fp16 = self
                        .sequence_volume_fp16
                        .saturating_sub(self.fade_step_fp16);
                }
                FadeCurve::Cosine => {
                    let t = f64::from(self.fade_elapsed_60hz) / f64::from(self.fade_total_60hz);
                    let gain = half_cosine_gain(t);
                    let scaled =
                        (gain.clamp(0.0, 1.0) * (self.initial_sequence_volume_fp16 as f64)) as u32;
                    self.sequence_volume_fp16 = scaled.min(self.initial_sequence_volume_fp16);
                }
            }
        }

        let new_int = self.sequence_volume_fp16 >> 16;
        if new_int != old_int {
            for ch in &mut self.channels[..self.channel_count] {
                if ch.active {
                    ch.dirty_volume = true;
                }
            }
            self.flush_dirty_voices()?;
        }

        if self.sequence_volume_fp16 == 0 {
            self.faded_out = true;
            self.enter_tail();
        }
        Ok(())
    }

    fn enter_tail(&mut self) {
        // Key off all 24 PS1 SPU voices (leave MVOLL/MVOLR untouched so reverb/release decay naturally).
        let _ = self.device.write_ps1_register(ps1::KOFF_LO, 0xFFFF);
        let _ = self.device.write_ps1_register(ps1::KOFF_HI, 0x00FF);
        for ch in &mut self.channels[..self.channel_count] {
            ch.active = false;
        }
        let tail_frames = (self.config.tail_seconds * f64::from(self.config.sample_rate)) as u64;
        if tail_frames == 0 {
            self.phase = PlayPhase::Finished;
        } else {
            self.tail_frames_remaining = tail_frames;
            self.phase = PlayPhase::Tail;
        }
    }

    fn allocate_voice(&mut self, ch_idx: usize) -> usize {
        // 1. Reuse already-owned hardware voice.
        if let Some(v) = self.channels[ch_idx].voice {
            let vi = usize::from(v);
            self.voice_age[vi] = self.timer_ticks;
            return vi;
        }

        // 2. Prefer 1:1 hardware voice for channels 0..24 if currently free (preserves PMON adjacency).
        if ch_idx < VOICES_PER_CORE && self.voice_owner[ch_idx].is_none() {
            return self.assign_voice(ch_idx, ch_idx);
        }

        // 3. Any completely free hardware voice.
        if let Some(free_v) = self.voice_owner.iter().position(|o| o.is_none()) {
            return self.assign_voice(ch_idx, free_v);
        }

        // 4. Steal a voice owned by an inactive or resting/keyed-off channel.
        for (v, owner_opt) in self.voice_owner.iter().enumerate() {
            if let Some(owner) = *owner_opt {
                let och = &self.channels[usize::from(owner)];
                if !och.active || (och.gate_remaining == 0 && !och.slur && !och.legato) {
                    return self.assign_voice(ch_idx, v);
                }
            }
        }

        // 5. Steal the oldest voice (LRU by timer_ticks).
        let mut oldest_v = 0usize;
        let mut oldest_age = u64::MAX;
        for (v, &age) in self.voice_age.iter().enumerate() {
            if age < oldest_age {
                oldest_age = age;
                oldest_v = v;
            }
        }
        self.assign_voice(ch_idx, oldest_v)
    }

    fn assign_voice(&mut self, ch_idx: usize, voice: usize) -> usize {
        if let Some(prev_owner) = self.voice_owner[voice] {
            let prev_idx = usize::from(prev_owner);
            if prev_idx != ch_idx {
                self.channels[prev_idx].voice = None;
            }
        }
        self.voice_owner[voice] = Some(ch_idx as u8);
        self.voice_age[voice] = self.timer_ticks;
        let ch = &mut self.channels[ch_idx];
        ch.voice = Some(voice as u8);
        ch.dirty_adsr = true;
        ch.dirty_pitch = true;
        ch.dirty_volume = true;
        self.dirty_voice_modes = true;
        voice
    }

    fn release_voice(&mut self, ch_idx: usize) {
        if let Some(v) = self.channels[ch_idx].voice.take() {
            let vi = usize::from(v);
            if self.voice_owner[vi] == Some(ch_idx as u8) {
                self.voice_owner[vi] = None;
                self.dirty_voice_modes = true;
            }
        }
    }

    fn step_musical_tick(&mut self) -> Result<()> {
        let mut kon_mask = 0u32;
        let mut koff_mask = 0u32;
        let mut any_active = false;

        // Global tempo slide.
        if self.tempo_slide_ticks > 0 {
            self.tempo_fp16 = self.tempo_fp16.wrapping_add(self.tempo_slide_step as u32);
            self.tempo_slide_ticks -= 1;
        }

        // Global reverb depth slide.
        if self.reverb_slide_ticks > 0 {
            self.reverb_depth_l_fp16 = self
                .reverb_depth_l_fp16
                .wrapping_add(self.reverb_slide_step);
            self.reverb_depth_r_fp16 = self
                .reverb_depth_r_fp16
                .wrapping_add(self.reverb_slide_step);
            self.reverb_slide_ticks -= 1;
            if self.reverb_active {
                let vl = (self.reverb_depth_l_fp16 >> 16) as i16;
                let vr = (self.reverb_depth_r_fp16 >> 16) as i16;
                self.device.write_ps1_register(ps1::EVOLL, vl as u16)?;
                self.device.write_ps1_register(ps1::EVOLR, vr as u16)?;
            }
        }

        let num_channels = self.channel_count;
        for ch_idx in 0..num_channels {
            if !self.channels[ch_idx].active {
                continue;
            }
            any_active = true;

            // Tick duration & gate counters.
            let ch = &mut self.channels[ch_idx];
            if ch.duration_remaining > 0 {
                ch.duration_remaining -= 1;
            }
            if ch.gate_remaining > 0 {
                ch.gate_remaining -= 1;
                if ch.gate_remaining == 0 && !ch.slur && !ch.legato {
                    if let Some(v) = ch.voice {
                        koff_mask |= 1u32 << v;
                    }
                }
            }

            let ran_commands = if ch.duration_remaining == 0 {
                self.execute_channel_commands(ch_idx, &mut kon_mask, &mut koff_mask)?;
                true
            } else {
                false
            };

            // Update per-channel slides and LFOs on non-trigger ticks so newly started
            // slides/LFOs play at their initial value on tick 0 without premature advance.
            let ch = &mut self.channels[ch_idx];
            if !ch.active || ran_commands {
                continue;
            }

            if ch.vol_slide_ticks > 0 {
                let old = ch.volume_u8();
                ch.volume_fp8 = (ch.volume_fp8 + ch.vol_slide_step).clamp(0, 0x7F00);
                ch.vol_slide_ticks -= 1;
                if ch.volume_u8() != old {
                    ch.dirty_volume = true;
                }
            }
            if ch.pan_slide_ticks > 0 {
                let old = ch.pan_u8();
                ch.pan_fp8 = (ch.pan_fp8 + ch.pan_slide_step).clamp(0, 0x7F00);
                ch.pan_slide_ticks -= 1;
                if ch.pan_u8() != old {
                    ch.dirty_volume = true;
                }
            }
            if ch.pitch_slide_ticks > 0 {
                ch.pitch_slide_fp8 += ch.pitch_slide_step;
                ch.pitch_slide_ticks -= 1;
                ch.dirty_pitch = true;
            }

            if ch.vibrato.tick_vibrato() {
                ch.dirty_pitch = true;
            }
            if ch.tremolo.tick_linear() {
                ch.dirty_volume = true;
            }
            if ch.autopan.tick_linear() {
                ch.dirty_volume = true;
            }
        }

        // Write global voice-mode masks (dirty-flagged) and dirty per-voice registers.
        if self.dirty_voice_modes {
            self.flush_voice_mode_masks()?;
        }
        self.flush_dirty_voices()?;

        if koff_mask != 0 {
            self.device
                .write_ps1_register(ps1::KOFF_LO, (koff_mask & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::KOFF_HI, ((koff_mask >> 16) & 0xFF) as u16)?;
        }
        if kon_mask != 0 {
            self.device
                .write_ps1_register(ps1::KON_LO, (kon_mask & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::KON_HI, ((kon_mask >> 16) & 0xFF) as u16)?;
        }

        // Check natural end-of-track across all channels.
        let still_active = self.channels[..self.channel_count].iter().any(|c| c.active);
        if !any_active || !still_active {
            self.reached_end = true;
            self.enter_tail();
            return Ok(());
        }

        // Check target loop completion -> start hardware-faithful per-sequence voice-volume fade-out.
        if self.phase == PlayPhase::Playing && self.config.target_loops > 0 {
            let max_loops = self.channels[..self.channel_count]
                .iter()
                .filter(|c| c.active)
                .map(|c| c.infinite_loops)
                .max()
                .unwrap_or(0);
            if max_loops > self.loops_completed {
                self.loops_completed = max_loops;
            }
            if self.loops_completed >= self.config.target_loops {
                self.start_fade_out();
            }
        }

        Ok(())
    }

    fn execute_channel_commands(
        &mut self,
        ch_idx: usize,
        kon_mask: &mut u32,
        koff_mask: &mut u32,
    ) -> Result<()> {
        let version = self.sequence.version;
        let mut commands = 0usize;

        while commands < MAX_COMMANDS_PER_TICK {
            commands += 1;
            let pc = self.channels[ch_idx].pc;
            let (opcode, next_pc) = decode_opcode(&self.sequence.data, pc, version)?;
            self.channels[ch_idx].pc = next_pc;

            match opcode {
                Opcode::Note { semitone, duration } => {
                    let dur = self.resolve_duration(ch_idx, duration);
                    self.trigger_note(ch_idx, semitone, dur, kon_mask, koff_mask)?;
                    break;
                }
                Opcode::Tie { duration } => {
                    let dur = self.resolve_duration(ch_idx, duration);
                    let ch = &mut self.channels[ch_idx];
                    ch.duration_remaining = u16::from(dur).max(1);
                    ch.gate_remaining = 0; // Suppress key-off during tie
                    break;
                }
                Opcode::Rest { duration } => {
                    let dur = self.resolve_duration(ch_idx, duration);
                    let ch = &mut self.channels[ch_idx];
                    ch.duration_remaining = u16::from(dur).max(1);
                    ch.gate_remaining = 0;
                    ch.slur = false;
                    ch.portamento_ticks = 0;
                    if let Some(v) = ch.voice {
                        *koff_mask |= 1u32 << v;
                    }
                    break;
                }
                Opcode::Nop
                | Opcode::Unknown(_)
                | Opcode::NoiseTimerDelay(_)
                | Opcode::PitchModTimerDelay(_)
                | Opcode::StereoModeOn
                | Opcode::StereoModeOff
                | Opcode::VibratoDepthSlide { .. }
                | Opcode::TremoloDepthSlide { .. }
                | Opcode::AutoPanDepthSlide { .. }
                | Opcode::VibratoRateSlide { .. }
                | Opcode::TremoloRateSlide { .. }
                | Opcode::AutoPanRateSlide { .. }
                | Opcode::VolumeBalanceSlide { .. }
                | Opcode::Reserved1(_, _)
                | Opcode::Reserved2(_, _, _) => {}
                Opcode::EndOfTrack => {
                    if let Some(v) = self.channels[ch_idx].voice {
                        *koff_mask |= 1u32 << v;
                    }
                    self.channels[ch_idx].active = false;
                    self.release_voice(ch_idx);
                    break;
                }
                Opcode::SetInstrument(inst) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.instrument = inst;
                    ch.resolved_instrument = inst;
                    ch.adsr_overridden_by_note = false;
                    if inst < 0x80 {
                        ch.keysplit_active = false;
                    }
                    if let Some(art) = self.bank.articulation(usize::from(inst)) {
                        ch.adsr1 = art.adsr1;
                        ch.adsr2 = art.adsr2;
                        ch.dirty_adsr = true;
                        ch.dirty_pitch = true;
                    }
                }
                Opcode::SetInstrumentAlt(inst) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.instrument = inst;
                    ch.resolved_instrument = inst;
                    ch.dirty_pitch = true;
                }
                Opcode::OverrideDuration(dur) => {
                    self.channels[ch_idx].duration_override = dur;
                }
                Opcode::SetMasterVolume(vol) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.master_volume = vol.min(127);
                    ch.dirty_volume = true;
                }
                Opcode::PitchBendSlide {
                    duration,
                    semitones,
                } => {
                    let ch = &mut self.channels[ch_idx];
                    let ticks = u16::from(duration).max(1);
                    let delta_fp8 = i32::from(semitones) << 8;
                    ch.pitch_slide_step = delta_fp8 / i32::from(ticks);
                    ch.pitch_slide_ticks = ticks;
                }
                Opcode::SetOctave(oct) => {
                    self.channels[ch_idx].octave = oct.min(10);
                }
                Opcode::OctaveUp => {
                    let ch = &mut self.channels[ch_idx];
                    ch.octave = ch.octave.saturating_add(1).min(10);
                }
                Opcode::OctaveDown => {
                    let ch = &mut self.channels[ch_idx];
                    ch.octave = ch.octave.saturating_sub(1);
                }
                Opcode::SetVolume(vol) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.volume_fp8 = i32::from(vol.min(127)) << 8;
                    ch.vol_slide_ticks = 0;
                    ch.dirty_volume = true;
                }
                Opcode::VolumeSlide { duration, target } => {
                    let ch = &mut self.channels[ch_idx];
                    let ticks = u16::from(duration).max(1);
                    let target_fp8 = i32::from(target.min(127)) << 8;
                    ch.vol_slide_step = (target_fp8 - ch.volume_fp8) / i32::from(ticks);
                    ch.vol_slide_ticks = ticks;
                }
                Opcode::SetPan(pan) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.pan_fp8 = i32::from(pan.min(127)) << 8;
                    ch.pan_slide_ticks = 0;
                    ch.dirty_volume = true;
                }
                Opcode::PanSlide { duration, target } => {
                    let ch = &mut self.channels[ch_idx];
                    let ticks = u16::from(duration).max(1);
                    let target_fp8 = i32::from(target.min(127)) << 8;
                    ch.pan_slide_step = (target_fp8 - ch.pan_fp8) / i32::from(ticks);
                    ch.pan_slide_ticks = ticks;
                }
                Opcode::NoiseClock(clock) => {
                    self.noise_clock = clock & 0x3F;
                    self.update_spucnt()?;
                }
                Opcode::SetAdsrAttackRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x7F00) | ((u16::from(rate) & 0x7F) << 8);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrDecayRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x00F0) | ((u16::from(rate) & 0x0F) << 4);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrSustainLevel(level) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x000F) | (u16::from(level) & 0x0F);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrDecayAndSustain {
                    decay,
                    sustain_level,
                } => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x00FF)
                        | ((u16::from(decay) & 0x0F) << 4)
                        | (u16::from(sustain_level) & 0x0F);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrSustainRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr2 = (ch.adsr2 & !0x1FC0) | ((u16::from(rate) & 0x7F) << 6);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrReleaseRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr2 = (ch.adsr2 & !0x001F) | (u16::from(rate) & 0x1F);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::ResetAdsr => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr_overridden_by_note = false;
                    if let Some(art) = self.bank.articulation(usize::from(ch.instrument)) {
                        ch.adsr1 = art.adsr1;
                        ch.adsr2 = art.adsr2;
                        ch.dirty_adsr = true;
                    }
                }
                Opcode::Vibrato {
                    delay,
                    rate,
                    waveform,
                } => {
                    self.channels[ch_idx]
                        .vibrato
                        .configure(delay, rate, waveform);
                }
                Opcode::VibratoDepth(depth) => {
                    self.channels[ch_idx].vibrato.depth = depth;
                }
                Opcode::VibratoOff => {
                    let ch = &mut self.channels[ch_idx];
                    ch.vibrato.enabled = false;
                    ch.vibrato.current_value = 0;
                    ch.dirty_pitch = true;
                }
                Opcode::SetAdsrAttackMode(mode) => {
                    let ch = &mut self.channels[ch_idx];
                    if mode != 0 {
                        ch.adsr1 |= 0x8000;
                    } else {
                        ch.adsr1 &= !0x8000;
                    }
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::Tremolo {
                    delay,
                    rate,
                    waveform,
                } => {
                    self.channels[ch_idx]
                        .tremolo
                        .configure(delay, rate, waveform);
                }
                Opcode::TremoloDepth(depth) => {
                    self.channels[ch_idx].tremolo.depth = depth;
                }
                Opcode::TremoloOff => {
                    let ch = &mut self.channels[ch_idx];
                    ch.tremolo.enabled = false;
                    ch.tremolo.current_value = 0;
                    ch.dirty_volume = true;
                }
                Opcode::SetAdsrSustainMode(mode) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr2 = (ch.adsr2 & !0xC000) | ((u16::from(mode) & 0x03) << 14);
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::AutoPan { rate, waveform } => {
                    self.channels[ch_idx].autopan.configure(0, rate, waveform);
                }
                Opcode::AutoPanDepth(depth) => {
                    self.channels[ch_idx].autopan.depth = depth;
                }
                Opcode::AutoPanOff => {
                    let ch = &mut self.channels[ch_idx];
                    ch.autopan.enabled = false;
                    ch.autopan.current_value = 0;
                    ch.dirty_volume = true;
                }
                Opcode::SetAdsrReleaseMode(mode) => {
                    let ch = &mut self.channels[ch_idx];
                    if mode != 0 {
                        ch.adsr2 |= 0x0020;
                    } else {
                        ch.adsr2 &= !0x0020;
                    }
                    ch.adsr_overridden_by_note = false;
                    ch.dirty_adsr = true;
                }
                Opcode::SetTranspose(t) => {
                    self.channels[ch_idx].transpose = t;
                }
                Opcode::AddTranspose(dt) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.transpose = ch.transpose.saturating_add(dt);
                }
                Opcode::ReverbOn => {
                    self.channels[ch_idx].reverb = true;
                    self.dirty_voice_modes = true;
                }
                Opcode::ReverbOff => {
                    self.channels[ch_idx].reverb = false;
                    self.dirty_voice_modes = true;
                }
                Opcode::NoiseOn | Opcode::NoiseTimerOn(_) => {
                    self.channels[ch_idx].noise = true;
                    self.dirty_voice_modes = true;
                }
                Opcode::NoiseOff => {
                    self.channels[ch_idx].noise = false;
                    self.dirty_voice_modes = true;
                }
                Opcode::PitchModOn | Opcode::PitchModTimerOn(_) => {
                    self.channels[ch_idx].pmon = true;
                    self.dirty_voice_modes = true;
                }
                Opcode::PitchModOff => {
                    self.channels[ch_idx].pmon = false;
                    self.dirty_voice_modes = true;
                }
                Opcode::LoopStart => {
                    let ch = &mut self.channels[ch_idx];
                    // Guard against 5-deep nested LoopStart overflowing and corrupting slot 0.
                    if ch.loop_depth < 4 {
                        let slot = ch.loop_depth;
                        ch.loop_stack[slot] = LoopFrame {
                            start_pc: next_pc,
                            iterations: 0,
                        };
                        ch.loop_depth += 1;
                    }
                }
                Opcode::LoopEnd(count) => {
                    let ch = &mut self.channels[ch_idx];
                    if ch.loop_depth > 0 {
                        let slot = ch.loop_depth - 1;
                        // Hardware AKAO (V1/V2/V3): count == 0 means 256 iterations (0xCA is infinite).
                        let target = if count == 0 { 256u16 } else { u16::from(count) };
                        ch.loop_stack[slot].iterations =
                            ch.loop_stack[slot].iterations.saturating_add(1);
                        if ch.loop_stack[slot].iterations < target {
                            ch.pc = ch.loop_stack[slot].start_pc;
                        } else {
                            ch.loop_depth -= 1;
                        }
                    }
                }
                Opcode::LoopInfinite => {
                    let ch = &mut self.channels[ch_idx];
                    if ch.loop_depth > 0 {
                        let slot = ch.loop_depth - 1;
                        ch.pc = ch.loop_stack[slot].start_pc;
                        ch.infinite_loops += 1;
                    } else {
                        if let Some(v) = ch.voice {
                            *koff_mask |= 1u32 << v;
                        }
                        ch.active = false;
                        self.release_voice(ch_idx);
                        break;
                    }
                }
                Opcode::ResetVoiceModes => {
                    let ch = &mut self.channels[ch_idx];
                    ch.reverb = false;
                    ch.pmon = false;
                    ch.noise = false;
                    ch.slur = false;
                    ch.legato = false;
                    ch.portamento_ticks = 0;
                    ch.phase_invert = false;
                    ch.dirty_volume = true;
                    self.dirty_voice_modes = true;
                }
                Opcode::SlurOn => self.channels[ch_idx].slur = true,
                Opcode::SlurOff => self.channels[ch_idx].slur = false,
                Opcode::LegatoOn => self.channels[ch_idx].legato = true,
                Opcode::LegatoOff => self.channels[ch_idx].legato = false,
                Opcode::FineTune(ft) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.fine_tune = ft;
                    ch.dirty_pitch = true;
                }
                Opcode::AddFineTune(dt) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.fine_tune = ch.fine_tune.saturating_add(dt);
                    ch.dirty_pitch = true;
                }
                Opcode::PortamentoOn(ticks) => {
                    self.channels[ch_idx].portamento_ticks = ticks;
                }
                Opcode::PortamentoOff => {
                    self.channels[ch_idx].portamento_ticks = 0;
                }
                Opcode::PhaseInvertOn => {
                    let ch = &mut self.channels[ch_idx];
                    ch.phase_invert = true;
                    ch.dirty_volume = true;
                }
                Opcode::PhaseInvertOff => {
                    let ch = &mut self.channels[ch_idx];
                    ch.phase_invert = false;
                    ch.dirty_volume = true;
                }
                Opcode::FixNoteLength(delta) => {
                    self.channels[ch_idx].fix_note_length = delta;
                }
                Opcode::VolumeBalance(bal) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.volume_balance = bal.min(127);
                    ch.dirty_volume = true;
                }
                Opcode::SetTempo(tempo) => {
                    self.tempo_fp16 = u32::from(tempo) << 16;
                    self.tempo_slide_ticks = 0;
                }
                Opcode::TempoSlide { duration, target } => {
                    let ticks = u16::from(duration).max(1);
                    let target_fp16 = (u32::from(target) << 16) as i64;
                    let cur_fp16 = i64::from(self.tempo_fp16);
                    self.tempo_slide_step = ((target_fp16 - cur_fp16) / i64::from(ticks)) as i32;
                    self.tempo_slide_ticks = ticks;
                }
                Opcode::SetReverbDepth(depth) => {
                    self.reverb_depth_l_fp16 = i32::from(depth) << 16;
                    self.reverb_depth_r_fp16 = i32::from(depth) << 16;
                    self.reverb_slide_ticks = 0;
                    if self.reverb_active {
                        self.device.write_ps1_register(ps1::EVOLL, depth as u16)?;
                        self.device.write_ps1_register(ps1::EVOLR, depth as u16)?;
                    }
                }
                Opcode::ReverbDepthSlide { duration, target } => {
                    let ticks = u16::from(duration).max(1);
                    let target_fp16 = i64::from(target) << 16;
                    let cur_fp16 = i64::from(self.reverb_depth_l_fp16);
                    self.reverb_slide_step = ((target_fp16 - cur_fp16) / i64::from(ticks)) as i32;
                    self.reverb_slide_ticks = ticks;
                }
                Opcode::DrumModeOnOffset(_) => {
                    self.channels[ch_idx].drum_mode = true;
                }
                Opcode::DrumModeOff => {
                    self.channels[ch_idx].drum_mode = false;
                }
                Opcode::Jump(offset) => {
                    let target = resolve_rel_jump(pc + 1, offset, self.sequence.data.len())?;
                    let ch = &mut self.channels[ch_idx];
                    if target <= pc {
                        ch.infinite_loops += 1;
                    }
                    ch.pc = target;
                }
                Opcode::ConditionalJump { condition, offset } => {
                    if self.condition >= condition {
                        let target = resolve_rel_jump(pc + 2, offset, self.sequence.data.len())?;
                        self.channels[ch_idx].pc = target;
                    }
                }
                Opcode::LoopBreak { iteration, offset }
                | Opcode::LoopBreakAlt { iteration, offset } => {
                    let ch = &mut self.channels[ch_idx];
                    if ch.loop_depth > 0 {
                        let slot = ch.loop_depth - 1;
                        if ch.loop_stack[slot].iterations + 1 == u16::from(iteration) {
                            ch.loop_depth -= 1;
                            let target =
                                resolve_rel_jump(pc + 2, offset, self.sequence.data.len())?;
                            ch.pc = target;
                        }
                    }
                }
                Opcode::SetKeysplit(_) => {
                    self.channels[ch_idx].keysplit_active = true;
                }
                Opcode::ResetKeysplit => {
                    self.channels[ch_idx].keysplit_active = false;
                }
                Opcode::Extended(ext) => {
                    self.execute_extended(ch_idx, pc, next_pc, ext)?;
                }
            }
        }

        Ok(())
    }

    fn execute_extended(
        &mut self,
        ch_idx: usize,
        op_pc: usize,
        next_pc: usize,
        ext: ExtendedOpcode,
    ) -> Result<()> {
        match ext {
            ExtendedOpcode::SetTempo(tempo) => {
                self.tempo_fp16 = u32::from(tempo) << 16;
                self.tempo_slide_ticks = 0;
            }
            ExtendedOpcode::TempoSlide { duration, target } => {
                let ticks = u16::from(duration).max(1);
                let target_fp16 = (u32::from(target) << 16) as i64;
                let cur_fp16 = i64::from(self.tempo_fp16);
                self.tempo_slide_step = ((target_fp16 - cur_fp16) / i64::from(ticks)) as i32;
                self.tempo_slide_ticks = ticks;
            }
            ExtendedOpcode::SetReverbDepth(depth) => {
                self.reverb_depth_l_fp16 = i32::from(depth) << 16;
                self.reverb_depth_r_fp16 = i32::from(depth) << 16;
                self.reverb_slide_ticks = 0;
                if self.reverb_active {
                    self.device.write_ps1_register(ps1::EVOLL, depth as u16)?;
                    self.device.write_ps1_register(ps1::EVOLR, depth as u16)?;
                }
            }
            ExtendedOpcode::ReverbDepthSlide { duration, target } => {
                let ticks = u16::from(duration).max(1);
                let target_fp16 = i64::from(target) << 16;
                let cur_fp16 = i64::from(self.reverb_depth_l_fp16);
                self.reverb_slide_step = ((target_fp16 - cur_fp16) / i64::from(ticks)) as i32;
                self.reverb_slide_ticks = ticks;
            }
            ExtendedOpcode::DrumModeOn | ExtendedOpcode::DrumModeOnOffset(_) => {
                self.channels[ch_idx].drum_mode = true;
            }
            ExtendedOpcode::DrumModeOff => {
                self.channels[ch_idx].drum_mode = false;
            }
            ExtendedOpcode::Jump(offset) => {
                let target = resolve_rel_jump(op_pc + 2, offset, self.sequence.data.len())?;
                let ch = &mut self.channels[ch_idx];
                if target <= op_pc {
                    ch.infinite_loops += 1;
                }
                ch.pc = target;
            }
            ExtendedOpcode::ConditionalJump { condition, offset } => {
                if self.condition >= condition {
                    let target = resolve_rel_jump(op_pc + 3, offset, self.sequence.data.len())?;
                    self.channels[ch_idx].pc = target;
                }
            }
            ExtendedOpcode::LoopBreak { iteration, offset }
            | ExtendedOpcode::LoopBreakAlt { iteration, offset } => {
                let ch = &mut self.channels[ch_idx];
                if ch.loop_depth > 0 {
                    let slot = ch.loop_depth - 1;
                    if ch.loop_stack[slot].iterations + 1 == u16::from(iteration) {
                        ch.loop_depth -= 1;
                        let target = resolve_rel_jump(op_pc + 3, offset, self.sequence.data.len())?;
                        ch.pc = target;
                    }
                }
            }
            ExtendedOpcode::SetInstrumentAlt(inst) => {
                let ch = &mut self.channels[ch_idx];
                ch.instrument = inst;
                ch.resolved_instrument = inst;
                ch.dirty_pitch = true;
            }
            ExtendedOpcode::VolumeBalance(bal) => {
                let ch = &mut self.channels[ch_idx];
                ch.volume_balance = bal.min(127);
                ch.dirty_volume = true;
            }
            ExtendedOpcode::SubroutineCall(offset) => {
                let target = resolve_rel_jump(op_pc + 2, offset, self.sequence.data.len())?;
                let ch = &mut self.channels[ch_idx];
                ch.subroutine_return_pc = Some(next_pc);
                ch.pc = target;
            }
            ExtendedOpcode::SubroutineReturn => {
                let ch = &mut self.channels[ch_idx];
                if let Some(ret_pc) = ch.subroutine_return_pc.take() {
                    ch.pc = ret_pc;
                }
            }
            ExtendedOpcode::SetKeysplit(_) => {
                self.channels[ch_idx].keysplit_active = true;
            }
            ExtendedOpcode::SelectKeysplitInstrument(inst) => {
                let ch = &mut self.channels[ch_idx];
                ch.instrument = inst;
                ch.keysplit_active = true;
            }
            ExtendedOpcode::ResetKeysplit => {
                self.channels[ch_idx].keysplit_active = false;
            }
            ExtendedOpcode::VolumeBalanceSlide { .. }
            | ExtendedOpcode::MasterVolumeSlide { .. }
            | ExtendedOpcode::TimeSignature(_, _)
            | ExtendedOpcode::MeasureNumber(_)
            | ExtendedOpcode::Reserved1(_, _)
            | ExtendedOpcode::Reserved2(_, _, _)
            | ExtendedOpcode::Reserved3(_, _)
            | ExtendedOpcode::Reserved4(_, _)
            | ExtendedOpcode::Unknown(_) => {}
        }
        Ok(())
    }

    fn resolve_duration(&mut self, ch_idx: usize, default_dur: u8) -> u8 {
        let ch = &mut self.channels[ch_idx];
        if ch.duration_override != 0 {
            let d = ch.duration_override;
            ch.duration_override = 0;
            d
        } else {
            default_dur
        }
    }

    fn trigger_note(
        &mut self,
        ch_idx: usize,
        semitone: u8,
        duration: u8,
        kon_mask: &mut u32,
        _koff_mask: &mut u32,
    ) -> Result<()> {
        let dur_u16 = u16::from(duration).max(1);
        let prev_note = self.channels[ch_idx].current_note;
        let prev_resolved_inst = self.channels[ch_idx].resolved_instrument;
        let drum_mode = self.channels[ch_idx].drum_mode;
        let keysplit_enabled =
            self.channels[ch_idx].keysplit_active || self.channels[ch_idx].instrument >= 0x80;

        let mut note = i16::from(self.channels[ch_idx].octave) * 12
            + i16::from(semitone)
            + i16::from(self.channels[ch_idx].transpose);
        let mut inst_idx = usize::from(self.channels[ch_idx].instrument);
        let mut override_adsr: Option<(u16, u16)> = None;
        let mut note_vol_override: Option<u8> = None;
        let mut note_pan_override: Option<u8> = None;

        // V3 embedded drum / keysplit resolution.
        if drum_mode && !self.sequence.drum_table.is_empty() {
            // Bounds-check drum index instead of wrapping with `%`.
            let drum_idx = note.max(0) as usize;
            if let Some(&entry) = self.sequence.drum_table.get(drum_idx) {
                inst_idx = usize::from(entry.articulation_index);
                note = i16::from(entry.note);
                if entry.adsr1 != 0 || entry.adsr2 != 0 {
                    override_adsr = Some((entry.adsr1, entry.adsr2));
                }
                note_vol_override = Some(entry.volume.min(127));
                note_pan_override = Some(entry.pan.min(127));
            }
        } else if keysplit_enabled && !self.sequence.keysplit_table.is_empty() {
            // Only apply keysplit_table when the channel selected a keysplit instrument.
            let note_u8 = note.clamp(0, 127) as u8;
            if let Some(split) = self
                .sequence
                .keysplit_table
                .iter()
                .find(|k| note_u8 >= k.low_note && note_u8 <= k.high_note)
            {
                inst_idx = usize::from(split.articulation_index);
                note = note - i16::from(split.base_note) + 48;
                if split.adsr1 != 0 || split.adsr2 != 0 {
                    override_adsr = Some((split.adsr1, split.adsr2));
                }
            }
        }

        // Allocate or reuse a hardware SPU voice (0..24) cleanly without `ch % 24` aliasing.
        let voice = self.allocate_voice(ch_idx);

        let art_opt = self.bank.articulation(inst_idx).copied();

        let ch = &mut self.channels[ch_idx];
        ch.resolved_instrument = inst_idx.min(255) as u8;
        ch.note_volume_override = note_vol_override;
        ch.note_pan_override = note_pan_override;
        ch.duration_remaining = dur_u16;

        // Hardware AKAO default gate-off is `dur - 2 + fix_note_length` unless slur/legato is active.
        ch.gate_remaining = if ch.slur || ch.legato {
            0
        } else {
            let base_gate = i32::from(dur_u16.saturating_sub(2)) + i32::from(ch.fix_note_length);
            base_gate.clamp(1, i32::from(dur_u16)) as u16
        };

        if ch.portamento_ticks > 0 {
            let ticks = u16::from(ch.portamento_ticks);
            let delta_fp8 = i32::from(note - prev_note) << 8;
            ch.current_note = prev_note;
            ch.pitch_slide_fp8 = 0;
            ch.pitch_slide_step = delta_fp8 / i32::from(ticks);
            ch.pitch_slide_ticks = ticks;
        } else {
            ch.current_note = note;
            ch.pitch_slide_fp8 = 0;
            ch.pitch_slide_ticks = 0;
        }

        // Don't leak sticky per-note ADSR overrides across notes: restore articulation
        // default ADSR whenever a note has no override and either the previous note
        // overrode ADSR or the resolved instrument changed.
        if let Some((a1, a2)) = override_adsr {
            ch.adsr1 = a1;
            ch.adsr2 = a2;
            ch.adsr_overridden_by_note = true;
            ch.dirty_adsr = true;
        } else if ch.adsr_overridden_by_note || ch.resolved_instrument != prev_resolved_inst {
            if let Some(art) = art_opt {
                ch.adsr1 = art.adsr1;
                ch.adsr2 = art.adsr2;
                ch.dirty_adsr = true;
            }
            ch.adsr_overridden_by_note = false;
        }

        ch.vibrato.reset_note();
        ch.tremolo.reset_note();
        ch.dirty_pitch = true;
        ch.dirty_volume = true;

        // Program SPU start/loop addresses from resolved articulation.
        if let Some(art) = art_opt {
            let vbase = (voice as u32) * ps1::VOICE_STRIDE;
            self.device
                .write_ps1_register(vbase + ps1::V_START_ADDR, (art.start_addr >> 3) as u16)?;
            self.device
                .write_ps1_register(vbase + ps1::V_REPEAT_ADDR, (art.loop_addr >> 3) as u16)?;
        }

        if !self.channels[ch_idx].legato {
            *kon_mask |= 1u32 << voice;
        }
        Ok(())
    }

    fn flush_dirty_voices(&mut self) -> Result<()> {
        let seq_vol = i32::from(self.sequence_volume());
        let mute_mask = self.mute_mask;
        let solo_mask = self.solo_mask;

        for ch_idx in 0..self.channel_count {
            let ch = &mut self.channels[ch_idx];
            if !ch.active {
                continue;
            }
            let Some(voice) = ch.voice else {
                continue;
            };
            let vbase = u32::from(voice) * ps1::VOICE_STRIDE;

            if ch.dirty_adsr {
                self.device
                    .write_ps1_register(vbase + ps1::V_ADSR1, ch.adsr1)?;
                self.device
                    .write_ps1_register(vbase + ps1::V_ADSR2, ch.adsr2)?;
                ch.dirty_adsr = false;
            }

            if ch.dirty_pitch {
                // Read fine_tune from the resolved articulation (handles drum/keysplit instruments).
                let fine_tune_art = self
                    .bank
                    .articulation(usize::from(ch.resolved_instrument))
                    .map_or(0x1000, |a| {
                        if a.fine_tune == 0 {
                            0x1000
                        } else {
                            a.fine_tune
                        }
                    });
                let pitch = compute_spu_pitch(
                    ch.current_note,
                    ch.pitch_slide_fp8,
                    ch.fine_tune,
                    ch.vibrato.current_value,
                    fine_tune_art,
                );
                ch.last_pitch = pitch;
                self.device
                    .write_ps1_register(vbase + ps1::V_PITCH, pitch)?;
                ch.dirty_pitch = false;
            }

            if ch.dirty_volume {
                let bit = 1u32 << ch_idx;
                let muted = (mute_mask & bit) != 0 || (solo_mask != 0 && (solo_mask & bit) == 0);
                let (vol_l, vol_r) = if muted {
                    (0u16, 0u16)
                } else {
                    let base_vol = i32::from(ch.volume_u8()) + i32::from(ch.tremolo.current_value);
                    let clamped_vol = base_vol.clamp(0, 127);
                    let note_vol = i32::from(ch.note_volume_override.unwrap_or(127));
                    let combined = ((i64::from(clamped_vol)
                        * i64::from(note_vol)
                        * i64::from(ch.master_volume)
                        * i64::from(ch.volume_balance)
                        * i64::from(seq_vol))
                        / (127 * 127 * 127 * 127)) as i32;

                    let base_pan = ch.note_pan_override.unwrap_or_else(|| ch.pan_u8());
                    let pan = (i32::from(base_pan) + i32::from(ch.autopan.current_value))
                        .clamp(0, 127) as u8;
                    let (pan_l, pan_r) = stereo_pan_gains(pan);

                    let vl = ((combined * i32::from(pan_l) * 0x3FFF) / (127 * 127)).clamp(0, 0x3FFF)
                        as u16;
                    let mut vr = ((combined * i32::from(pan_r) * 0x3FFF) / (127 * 127))
                        .clamp(0, 0x3FFF) as u16;
                    if ch.phase_invert && vr != 0 {
                        // PS1 SPU 15-bit two's-complement negative fixed volume.
                        vr = (!vr).wrapping_add(1) & 0x7FFF;
                    }
                    (vl, vr)
                };

                self.device.write_ps1_register(vbase + ps1::V_VOLL, vol_l)?;
                self.device.write_ps1_register(vbase + ps1::V_VOLR, vol_r)?;
                ch.dirty_volume = false;
            }
        }
        Ok(())
    }

    fn flush_voice_mode_masks(&mut self) -> Result<()> {
        let mut pmon = 0u32;
        let mut noise = 0u32;
        let mut reverb = 0u32;
        for ch in &self.channels[..self.channel_count] {
            if !ch.active {
                continue;
            }
            let Some(v) = ch.voice else {
                continue;
            };
            let bit = 1u32 << v;
            if ch.pmon && v > 0 {
                pmon |= bit;
            }
            if ch.noise {
                noise |= bit;
            }
            if ch.reverb && self.reverb_active {
                reverb |= bit;
            }
        }

        if pmon != self.last_pmon {
            self.device
                .write_ps1_register(ps1::PMON_LO, (pmon & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::PMON_HI, ((pmon >> 16) & 0xFF) as u16)?;
            self.last_pmon = pmon;
        }
        if noise != self.last_noise {
            self.device
                .write_ps1_register(ps1::NON_LO, (noise & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::NON_HI, ((noise >> 16) & 0xFF) as u16)?;
            self.last_noise = noise;
        }
        if reverb != self.last_reverb {
            self.device
                .write_ps1_register(ps1::EON_LO, (reverb & 0xFFFF) as u16)?;
            self.device
                .write_ps1_register(ps1::EON_HI, ((reverb >> 16) & 0xFF) as u16)?;
            self.last_reverb = reverb;
        }
        self.dirty_voice_modes = false;
        Ok(())
    }

    fn update_spucnt(&mut self) -> Result<()> {
        let mut spucnt =
            CONTROL_ENABLE | CONTROL_UNMUTE | (u16::from(self.noise_clock & 0x3F) << 8);
        if self.reverb_active {
            spucnt |= CONTROL_REVERB;
        }
        self.device.write_ps1_register(ps1::SPUCNT, spucnt)?;
        Ok(())
    }
}

fn resolve_rel_jump(operand_addr: usize, offset: i16, len: usize) -> Result<usize> {
    let target = (operand_addr as i64) + i64::from(offset);
    if target < 0 || target as usize >= len {
        return Err(Error::OutOfRange("AKAO relative jump target out of bounds"));
    }
    Ok(target as usize)
}

#[inline]
fn stereo_pan_gains(pan: u8) -> (u8, u8) {
    let p = usize::from(pan.min(127));
    (PAN_SIN_LUT[127 - p], PAN_SIN_LUT[p])
}

fn compute_spu_pitch(
    note: i16,
    pitch_slide_fp8: i32,
    fine_tune: i8,
    vibrato: i16,
    art_fine_tune: u16,
) -> u16 {
    let total_fp8 = (i32::from(note) << 8) + pitch_slide_fp8 + i32::from(fine_tune) * 2;
    let clamped_fp8 = total_fp8.clamp(0, 127 << 8);
    let whole_note = (clamped_fp8 >> 8) as usize;
    let frac = (clamped_fp8 & 0xFF) as u32;

    let octave = whole_note / 12;
    let semi = whole_note % 12;
    let next_semi = (semi + 1) % 12;
    let next_oct = if semi == 11 { octave + 1 } else { octave };

    let p0 = octave_shifted_pitch(PITCH_TABLE_OCT4[semi], octave);
    let p1 = octave_shifted_pitch(PITCH_TABLE_OCT4[next_semi], next_oct);
    let mut pitch = p0 + (((p1.saturating_sub(p0)) * frac) >> 8);

    // Apply articulation fine-tune (`0x1000` = 1.0x).
    pitch = (pitch * u32::from(art_fine_tune)) >> 12;

    // Apply vibrato offset cleanly.
    let vib_delta = (i64::from(pitch) * i64::from(vibrato)) >> 10;
    let final_pitch = (i64::from(pitch) + vib_delta).clamp(1, 0x3FFF);
    final_pitch as u16
}

fn octave_shifted_pitch(base_oct4: u16, octave: usize) -> u32 {
    let base = u32::from(base_oct4);
    if octave >= 4 {
        base.checked_shl((octave - 4) as u32).unwrap_or(0x3FFF)
    } else {
        base >> (4 - octave)
    }
}

/// Pure `#![no_std]` polynomial approximation of `0.5 * (1.0 + cos(pi * t))` for `t in [0, 1]`.
fn half_cosine_gain(t: f64) -> f64 {
    let t = t.clamp(0.0, 1.0);
    let t2 = t * t;
    let t3 = t2 * t;
    1.0 - (3.0 * t2 - 2.0 * t3)
}
