// SPDX-License-Identifier: MIT

//! AKAO sequencer playback engine driving `libspu2::Device::new_ps1()`.

use alloc::vec::Vec;

use libspu2::registers::{ps1, CONTROL_ENABLE, CONTROL_REVERB, CONTROL_UNMUTE};
use libspu2::{Device, StereoI16, PS1_SAMPLE_RATE, VOICES_PER_CORE};

use crate::bank::AkaoWaveBank;
use crate::opcodes::{decode_opcode, ExtendedOpcode, Opcode};
use crate::sequence::AkaoSequence;
use crate::Error;

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

impl Default for RenderConfig {
    fn default() -> Self {
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
    /// Current program counter offset into [`AkaoSequence::data`].
    pub pc: usize,
    /// Mapped PS1 SPU hardware voice index (`0..24`).
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
    /// Whether SPU reverb send (`EON`) is enabled for this channel.
    pub reverb: bool,
    /// Whether SPU pitch modulation (`PMON`) is enabled for this channel.
    pub pmon: bool,
    /// Whether SPU hardware noise (`NON`) is enabled for this channel.
    pub noise: bool,
    /// Whether right-channel phase inversion is active.
    pub phase_invert: bool,
}

#[derive(Debug, Clone, Copy, Default)]
struct LoopFrame {
    start_pc: usize,
    iterations: u8,
}

#[derive(Debug, Clone, Copy, Default)]
struct LfoState {
    enabled: bool,
    delay: u16,
    delay_remaining: u16,
    rate: u16,
    tick_counter: u16,
    phase: u8,
    depth: u8,
    waveform: u8,
    current_value: i16,
}

impl LfoState {
    fn reset_note(&mut self) {
        self.delay_remaining = self.delay;
        self.tick_counter = self.rate.max(1);
        self.phase = 0;
        self.current_value = 0;
    }

    fn tick(&mut self) -> bool {
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
        if self.tick_counter > 1 {
            self.tick_counter -= 1;
            return false;
        }
        self.tick_counter = self.rate.max(1);
        let sample = lfo_sample(self.waveform, self.phase);
        self.phase = (self.phase + 1) & 31;
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
            // Triangle wave (-127..127)
            if idx < 8 {
                (idx as i8) * 15
            } else if idx < 24 {
                120 - ((idx - 8) as i8) * 15
            } else {
                -120 + ((idx - 24) as i8) * 15
            }
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

#[derive(Debug, Clone)]
struct ChannelState {
    active: bool,
    pc: usize,
    voice: usize,
    instrument: u8,
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
    /// Channel pan in 8.8 fixed point (`0x4000` = center 64.0).
    pan_fp8: i32,
    pan_slide_step: i32,
    pan_slide_ticks: u16,
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
    /// Mode flags.
    slur: bool,
    legato: bool,
    portamento_ticks: u8,
    drum_mode: bool,
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
    fn new(voice: usize, start_pc: Option<usize>, rhythm: bool) -> Self {
        Self {
            active: start_pc.is_some(),
            pc: start_pc.unwrap_or(0),
            voice,
            instrument: 0,
            octave: 4,
            transpose: 0,
            fine_tune: 0,
            volume_fp8: 0x7F00,
            vol_slide_step: 0,
            vol_slide_ticks: 0,
            master_volume: 127,
            volume_balance: 127,
            pan_fp8: 0x4000,
            pan_slide_step: 0,
            pan_slide_ticks: 0,
            pitch_slide_fp8: 0,
            pitch_slide_step: 0,
            pitch_slide_ticks: 0,
            current_note: 48,
            last_pitch: 0x1000,
            duration_remaining: 1,
            gate_remaining: 0,
            duration_override: 0,
            loop_stack: [LoopFrame::default(); 4],
            loop_depth: 0,
            subroutine_return_pc: None,
            infinite_loops: 0,
            adsr1: 0x00FF,
            adsr2: 0x0000,
            slur: false,
            legato: false,
            portamento_ticks: 0,
            drum_mode: rhythm,
            reverb: false,
            pmon: false,
            noise: false,
            phase_invert: false,
            vibrato: LfoState::default(),
            tremolo: LfoState::default(),
            autopan: LfoState::default(),
            dirty_volume: true,
            dirty_pitch: false,
            dirty_adsr: true,
        }
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
    channels: Vec<ChannelState>,
    /// 16.16 fixed-point tempo.
    tempo_fp16: u32,
    tempo_slide_step: i32,
    tempo_slide_ticks: u16,
    /// 16-bit fractional tempo accumulator.
    tempo_accum: u32,
    /// Global sequence condition byte (`0xEA` / extended `0x07`).
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
    /// Per-sequence master volume in 16.16 fixed point (`0x7F_0000` = 127.0).
    /// Ramped linearly at 60 Hz during fade-out, scaling `V_VOLL`/`V_VOLR` (never `MVOLL`/`MVOLR`).
    sequence_volume_fp16: u32,
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

impl AkaoPlayer {
    /// Creates a new [`AkaoPlayer`], uploads `bank` to a fresh [`Device::new_ps1`],
    /// and initializes all active sequence channels.
    pub fn new(
        sequence: AkaoSequence,
        bank: AkaoWaveBank,
        config: RenderConfig,
    ) -> Result<Self, Error> {
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

        // Configure PS1 SPU control, master volume, and reverb work area.
        let mut spucnt = CONTROL_ENABLE | CONTROL_UNMUTE;
        if config.reverb_enabled {
            spucnt |= CONTROL_REVERB;
        }
        device.write_ps1_register(ps1::SPUCNT, spucnt)?;
        device.write_ps1_register(ps1::MVOLL, config.master_volume & 0x7FFF)?;
        device.write_ps1_register(ps1::MVOLR, config.master_volume & 0x7FFF)?;

        let init_rvb = if config.reverb_enabled { 0x2000i16 } else { 0 };
        device.write_ps1_register(ps1::EVOLL, init_rvb as u16)?;
        device.write_ps1_register(ps1::EVOLR, init_rvb as u16)?;
        // Place reverb work area in the upper 64 KiB of 512 KiB PS1 RAM (`0x70000 >> 3 = 0xE000`).
        device.write_ps1_register(ps1::RVB_WORK_ADDR, 0xE000)?;

        let max_ch = sequence.version.max_channels();
        let mut channels = Vec::with_capacity(max_ch);
        for ch in 0..max_ch {
            let start = sequence.channel_start(ch);
            let rhythm = (sequence.rhythm_mask & (1u32 << ch)) != 0;
            let voice = ch % VOICES_PER_CORE;
            let mut state = ChannelState::new(voice, start, rhythm);
            if let Some(art) = bank.articulation(0) {
                state.adsr1 = art.adsr1;
                state.adsr2 = art.adsr2;
            }
            channels.push(state);
        }

        let samples_per_tick_num =
            u64::from(config.sample_rate) * u64::from(sequence.version.timer_divider());
        let samples_per_tick_den = 4_233_600u64;
        let fade_total_60hz = ((config.fade_seconds * 60.0) + 0.5) as u32;
        let max_frames = (config.max_seconds * f64::from(config.sample_rate)) as u64;

        Ok(Self {
            sequence,
            bank,
            config,
            device,
            channels,
            tempo_fp16: DEFAULT_TEMPO_FP16,
            tempo_slide_step: 0,
            tempo_slide_ticks: 0,
            tempo_accum: 0,
            condition: 0,
            reverb_depth_l_fp16: i32::from(init_rvb) << 16,
            reverb_depth_r_fp16: i32::from(init_rvb) << 16,
            reverb_slide_step: 0,
            reverb_slide_ticks: 0,
            noise_clock: 0,
            timer_ticks: 0,
            sample_accum_num: 0,
            samples_per_tick_num,
            samples_per_tick_den,
            sequence_volume_fp16: FULL_SEQUENCE_VOLUME_FP16,
            fade_elapsed_60hz: 0,
            fade_total_60hz,
            tail_frames_remaining: 0,
            max_frames,
            frames_rendered: 0,
            loops_completed: 0,
            faded_out: false,
            reached_end: false,
            phase: PlayPhase::Playing,
        })
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

    /// Returns `true` once playback (including fade-out and tail) has finished.
    #[inline]
    #[must_use]
    pub fn is_finished(&self) -> bool {
        self.phase == PlayPhase::Finished
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

    /// Returns a snapshot of `channel`'s current state, or `None` if out of range.
    #[must_use]
    pub fn channel_snapshot(&self, channel: usize) -> Option<ChannelSnapshot> {
        let ch = self.channels.get(channel)?;
        Some(ChannelSnapshot {
            active: ch.active,
            pc: ch.pc,
            voice: ch.voice,
            instrument: ch.instrument,
            octave: ch.octave,
            transpose: ch.transpose,
            volume: ch.volume_u8(),
            pan: ch.pan_u8(),
            pitch: ch.last_pitch,
            loop_depth: ch.loop_depth,
            reverb: ch.reverb,
            pmon: ch.pmon,
            noise: ch.noise,
            phase_invert: ch.phase_invert,
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
    pub fn render_samples(&mut self, out: &mut [i16]) -> Result<usize, Error> {
        if out.len() % 2 != 0 {
            return Err(Error::InvalidArgument(
                "interleaved stereo buffer length must be even",
            ));
        }
        let total_frames = out.len() / 2;
        let mut rendered = 0usize;

        while rendered < total_frames && self.phase != PlayPhase::Finished {
            if self.frames_rendered >= self.max_frames {
                self.phase = PlayPhase::Finished;
                break;
            }

            if self.phase == PlayPhase::Tail {
                let remaining_cap = (self.max_frames - self.frames_rendered) as usize;
                let to_render = (total_frames - rendered)
                    .min(self.tail_frames_remaining as usize)
                    .min(remaining_cap);
                if to_render == 0 {
                    self.phase = PlayPhase::Finished;
                    break;
                }
                let start = rendered * 2;
                let end = (rendered + to_render) * 2;
                self.device.render_native(&mut out[start..end]);
                rendered += to_render;
                self.frames_rendered += to_render as u64;
                self.tail_frames_remaining -= to_render as u64;
                if self.tail_frames_remaining == 0 {
                    self.phase = PlayPhase::Finished;
                }
                continue;
            }

            // Step hardware Root Counter 2 timer whenever fewer than 1 full sample remains.
            if self.sample_accum_num < self.samples_per_tick_den {
                self.step_hardware_timer()?;
                if self.phase == PlayPhase::Tail || self.phase == PlayPhase::Finished {
                    continue;
                }
                self.sample_accum_num += self.samples_per_tick_num;
            }

            let step_frames = ((self.sample_accum_num / self.samples_per_tick_den) as usize).max(1);
            let remaining_cap = (self.max_frames - self.frames_rendered) as usize;
            let to_render = step_frames.min(total_frames - rendered).min(remaining_cap);

            let start = rendered * 2;
            let end = (rendered + to_render) * 2;
            self.device.render_native(&mut out[start..end]);
            rendered += to_render;
            self.frames_rendered += to_render as u64;
            self.sample_accum_num -= (to_render as u64) * self.samples_per_tick_den;
        }

        Ok(rendered)
    }

    /// Renders stereo frames into `out`, returning the number of frames rendered.
    pub fn render_frames(&mut self, out: &mut [StereoI16]) -> Result<usize, Error> {
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

    /// Renders the entire sequence (respecting `target_loops`, `fade_seconds`,
    /// `tail_seconds`, and `max_seconds`) into a [`RenderResult`].
    pub fn render_all(&mut self) -> Result<RenderResult, Error> {
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

    fn step_hardware_timer(&mut self) -> Result<(), Error> {
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

    fn step_fade_60hz(&mut self) -> Result<(), Error> {
        self.fade_elapsed_60hz += 1;
        let old_int = self.sequence_volume_fp16 >> 16;

        if self.fade_elapsed_60hz >= self.fade_total_60hz {
            self.sequence_volume_fp16 = 0;
        } else {
            let t = f64::from(self.fade_elapsed_60hz) / f64::from(self.fade_total_60hz);
            let gain = match self.config.fade_curve {
                FadeCurve::Linear => 1.0 - t,
                FadeCurve::Cosine => half_cosine_gain(t),
            };
            let scaled = (gain.clamp(0.0, 1.0) * (FULL_SEQUENCE_VOLUME_FP16 as f64)) as u32;
            self.sequence_volume_fp16 = scaled.min(FULL_SEQUENCE_VOLUME_FP16);
        }

        let new_int = self.sequence_volume_fp16 >> 16;
        if new_int != old_int {
            for ch in &mut self.channels {
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
        for ch in &mut self.channels {
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

    fn step_musical_tick(&mut self) -> Result<(), Error> {
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
            if self.config.reverb_enabled {
                let vl = (self.reverb_depth_l_fp16 >> 16) as i16;
                let vr = (self.reverb_depth_r_fp16 >> 16) as i16;
                self.device.write_ps1_register(ps1::EVOLL, vl as u16)?;
                self.device.write_ps1_register(ps1::EVOLR, vr as u16)?;
            }
        }

        let num_channels = self.channels.len();
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
                    koff_mask |= 1u32 << ch.voice;
                }
            }

            if ch.duration_remaining == 0 {
                self.execute_channel_commands(ch_idx, &mut kon_mask, &mut koff_mask)?;
            }

            // Update per-channel slides and LFOs.
            let ch = &mut self.channels[ch_idx];
            if !ch.active {
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

            if ch.vibrato.tick() {
                ch.dirty_pitch = true;
            }
            if ch.tremolo.tick() {
                ch.dirty_volume = true;
            }
            if ch.autopan.tick() {
                ch.dirty_volume = true;
            }
        }

        // Write global voice-mode masks (PMON, NON, EON) and dirty per-voice registers.
        self.flush_voice_mode_masks()?;
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
        let still_active = self.channels.iter().any(|c| c.active);
        if !any_active || !still_active {
            self.reached_end = true;
            self.enter_tail();
            return Ok(());
        }

        // Check target loop completion -> start hardware-faithful per-sequence voice-volume fade-out.
        if self.phase == PlayPhase::Playing && self.config.target_loops > 0 {
            let max_loops = self
                .channels
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
    ) -> Result<(), Error> {
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
                    *koff_mask |= 1u32 << ch.voice;
                    break;
                }
                Opcode::Nop | Opcode::Unknown(_) => {}
                Opcode::EndOfTrack => {
                    let ch = &mut self.channels[ch_idx];
                    ch.active = false;
                    *koff_mask |= 1u32 << ch.voice;
                    break;
                }
                Opcode::SetInstrument(inst) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.instrument = inst;
                    if let Some(art) = self.bank.articulation(usize::from(inst)) {
                        ch.adsr1 = art.adsr1;
                        ch.adsr2 = art.adsr2;
                        ch.dirty_adsr = true;
                        ch.dirty_pitch = true;
                    }
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
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrDecayRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x00F0) | ((u16::from(rate) & 0x0F) << 4);
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrSustainLevel(level) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr1 = (ch.adsr1 & !0x000F) | (u16::from(level) & 0x0F);
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
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrSustainRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr2 = (ch.adsr2 & !0x1FC0) | ((u16::from(rate) & 0x7F) << 6);
                    ch.dirty_adsr = true;
                }
                Opcode::SetAdsrReleaseRate(rate) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.adsr2 = (ch.adsr2 & !0x001F) | (u16::from(rate) & 0x1F);
                    ch.dirty_adsr = true;
                }
                Opcode::ResetAdsr => {
                    let ch = &mut self.channels[ch_idx];
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
                    let v = &mut self.channels[ch_idx].vibrato;
                    v.enabled = true;
                    v.delay = u16::from(delay);
                    v.rate = if rate == 0 { 256 } else { u16::from(rate) };
                    v.waveform = waveform;
                    v.reset_note();
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
                    ch.dirty_adsr = true;
                }
                Opcode::Tremolo {
                    delay,
                    rate,
                    waveform,
                } => {
                    let t = &mut self.channels[ch_idx].tremolo;
                    t.enabled = true;
                    t.delay = u16::from(delay);
                    t.rate = if rate == 0 { 256 } else { u16::from(rate) };
                    t.waveform = waveform;
                    t.reset_note();
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
                    ch.dirty_adsr = true;
                }
                Opcode::AutoPan { rate, waveform } => {
                    let p = &mut self.channels[ch_idx].autopan;
                    p.enabled = true;
                    p.delay = 0;
                    p.rate = if rate == 0 { 256 } else { u16::from(rate) };
                    p.waveform = waveform;
                    p.reset_note();
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
                    ch.dirty_adsr = true;
                }
                Opcode::SetTranspose(t) => {
                    self.channels[ch_idx].transpose = t;
                }
                Opcode::AddTranspose(dt) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.transpose = ch.transpose.saturating_add(dt);
                }
                Opcode::ReverbOn => self.channels[ch_idx].reverb = true,
                Opcode::ReverbOff => self.channels[ch_idx].reverb = false,
                Opcode::NoiseOn => self.channels[ch_idx].noise = true,
                Opcode::NoiseOff => self.channels[ch_idx].noise = false,
                Opcode::PitchModOn => self.channels[ch_idx].pmon = true,
                Opcode::PitchModOff => self.channels[ch_idx].pmon = false,
                Opcode::LoopStart => {
                    let ch = &mut self.channels[ch_idx];
                    let slot = ch.loop_depth & 3;
                    ch.loop_stack[slot] = LoopFrame {
                        start_pc: next_pc,
                        iterations: 0,
                    };
                    if ch.loop_depth < 4 {
                        ch.loop_depth += 1;
                    }
                }
                Opcode::LoopEnd(count) => {
                    let ch = &mut self.channels[ch_idx];
                    if ch.loop_depth > 0 {
                        let slot = (ch.loop_depth - 1) & 3;
                        ch.loop_stack[slot].iterations =
                            ch.loop_stack[slot].iterations.wrapping_add(1);
                        if count == 0 {
                            ch.pc = ch.loop_stack[slot].start_pc;
                            ch.infinite_loops += 1;
                        } else if ch.loop_stack[slot].iterations < count {
                            ch.pc = ch.loop_stack[slot].start_pc;
                        } else {
                            ch.loop_depth -= 1;
                        }
                    }
                }
                Opcode::LoopInfinite => {
                    let ch = &mut self.channels[ch_idx];
                    if ch.loop_depth > 0 {
                        let slot = (ch.loop_depth - 1) & 3;
                        ch.pc = ch.loop_stack[slot].start_pc;
                        ch.infinite_loops += 1;
                    } else {
                        ch.active = false;
                        *koff_mask |= 1u32 << ch.voice;
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
                Opcode::VolumeBalance(bal) => {
                    let ch = &mut self.channels[ch_idx];
                    ch.volume_balance = bal.min(127);
                    ch.dirty_volume = true;
                }
                Opcode::SetReverbDepthStereo { left, right } => {
                    self.reverb_depth_l_fp16 = i32::from(left) << 16;
                    self.reverb_depth_r_fp16 = i32::from(right) << 16;
                    self.reverb_slide_ticks = 0;
                    if self.config.reverb_enabled {
                        self.device.write_ps1_register(ps1::EVOLL, left as u16)?;
                        self.device.write_ps1_register(ps1::EVOLR, right as u16)?;
                    }
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
                Opcode::SetCondition(cond) => {
                    self.condition = cond;
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
    ) -> Result<(), Error> {
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
                if self.config.reverb_enabled {
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
            ExtendedOpcode::DrumModeOn => {
                self.channels[ch_idx].drum_mode = true;
            }
            ExtendedOpcode::DrumModeOff => {
                self.channels[ch_idx].drum_mode = false;
            }
            ExtendedOpcode::Jump(offset) => {
                // Operand `i16` starts at `op_pc + 2`.
                let target = resolve_rel_jump(op_pc + 2, offset, self.sequence.data.len())?;
                let ch = &mut self.channels[ch_idx];
                if target <= op_pc {
                    ch.infinite_loops += 1;
                }
                ch.pc = target;
            }
            ExtendedOpcode::ConditionalJump { condition, offset } => {
                if self.condition >= condition {
                    // Operand `i16` starts at `op_pc + 3`.
                    let target = resolve_rel_jump(op_pc + 3, offset, self.sequence.data.len())?;
                    self.channels[ch_idx].pc = target;
                }
            }
            ExtendedOpcode::LoopBreak { iteration, offset }
            | ExtendedOpcode::LoopBreakAlt { iteration, offset } => {
                let ch = &mut self.channels[ch_idx];
                if ch.loop_depth > 0 {
                    let slot = (ch.loop_depth - 1) & 3;
                    if ch.loop_stack[slot].iterations + 1 == iteration {
                        ch.loop_depth -= 1;
                        let target = resolve_rel_jump(op_pc + 3, offset, self.sequence.data.len())?;
                        ch.pc = target;
                    }
                }
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
            ExtendedOpcode::Unknown(_) => {}
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
    ) -> Result<(), Error> {
        let dur_u16 = u16::from(duration).max(1);
        let prev_note = self.channels[ch_idx].current_note;
        let drum_mode = self.channels[ch_idx].drum_mode;

        let mut note = i16::from(self.channels[ch_idx].octave) * 12
            + i16::from(semitone)
            + i16::from(self.channels[ch_idx].transpose);
        let mut inst_idx = usize::from(self.channels[ch_idx].instrument);
        let mut override_adsr: Option<(u16, u16)> = None;

        // V3 embedded drum / keysplit resolution.
        if drum_mode && !self.sequence.drum_table.is_empty() {
            let drum_idx = (note.max(0) as usize) % self.sequence.drum_table.len();
            let entry = self.sequence.drum_table[drum_idx];
            inst_idx = usize::from(entry.articulation_index);
            note = i16::from(entry.note);
            if entry.adsr1 != 0 || entry.adsr2 != 0 {
                override_adsr = Some((entry.adsr1, entry.adsr2));
            }
        } else if !self.sequence.keysplit_table.is_empty() {
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

        let voice = self.channels[ch_idx].voice;
        let ch = &mut self.channels[ch_idx];
        ch.duration_remaining = dur_u16;
        // Hardware AKAO default gate-off is `dur - 2` unless slur/legato is active.
        ch.gate_remaining = if ch.slur || ch.legato {
            0
        } else {
            dur_u16.saturating_sub(2).max(1)
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

        if let Some((a1, a2)) = override_adsr {
            ch.adsr1 = a1;
            ch.adsr2 = a2;
            ch.dirty_adsr = true;
        }

        ch.vibrato.reset_note();
        ch.tremolo.reset_note();
        ch.dirty_pitch = true;
        ch.dirty_volume = true;

        // Program SPU start/loop addresses from articulation.
        if let Some(art) = self.bank.articulation(inst_idx) {
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

    fn flush_dirty_voices(&mut self) -> Result<(), Error> {
        let seq_vol = i32::from(self.sequence_volume());

        for ch in &mut self.channels {
            if !ch.active {
                continue;
            }
            let vbase = (ch.voice as u32) * ps1::VOICE_STRIDE;

            if ch.dirty_adsr {
                self.device
                    .write_ps1_register(vbase + ps1::V_ADSR1, ch.adsr1)?;
                self.device
                    .write_ps1_register(vbase + ps1::V_ADSR2, ch.adsr2)?;
                ch.dirty_adsr = false;
            }

            if ch.dirty_pitch {
                let fine_tune_art =
                    self.bank
                        .articulation(usize::from(ch.instrument))
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
                let base_vol = i32::from(ch.volume_u8()) + i32::from(ch.tremolo.current_value);
                let clamped_vol = base_vol.clamp(0, 127);
                let combined = (clamped_vol
                    * i32::from(ch.master_volume)
                    * i32::from(ch.volume_balance)
                    * seq_vol)
                    / (127 * 127 * 127);

                let pan =
                    (i32::from(ch.pan_u8()) + i32::from(ch.autopan.current_value)).clamp(0, 127);
                let (pan_l, pan_r) = stereo_pan_gains(pan as u8);

                let vol_l =
                    ((combined * i32::from(pan_l) * 0x3FFF) / (127 * 127)).clamp(0, 0x3FFF) as u16;
                let mut vol_r =
                    ((combined * i32::from(pan_r) * 0x3FFF) / (127 * 127)).clamp(0, 0x3FFF) as u16;
                if ch.phase_invert && vol_r != 0 {
                    // PS1 SPU 15-bit two's-complement negative fixed volume.
                    vol_r = (!vol_r).wrapping_add(1) & 0x7FFF;
                }

                self.device.write_ps1_register(vbase + ps1::V_VOLL, vol_l)?;
                self.device.write_ps1_register(vbase + ps1::V_VOLR, vol_r)?;
                ch.dirty_volume = false;
            }
        }
        Ok(())
    }

    fn flush_voice_mode_masks(&mut self) -> Result<(), Error> {
        let mut pmon = 0u32;
        let mut noise = 0u32;
        let mut reverb = 0u32;
        for ch in &self.channels {
            if !ch.active {
                continue;
            }
            let bit = 1u32 << ch.voice;
            if ch.pmon && ch.voice > 0 {
                pmon |= bit;
            }
            if ch.noise {
                noise |= bit;
            }
            if ch.reverb && self.config.reverb_enabled {
                reverb |= bit;
            }
        }

        self.device
            .write_ps1_register(ps1::PMON_LO, (pmon & 0xFFFF) as u16)?;
        self.device
            .write_ps1_register(ps1::PMON_HI, ((pmon >> 16) & 0xFF) as u16)?;
        self.device
            .write_ps1_register(ps1::NON_LO, (noise & 0xFFFF) as u16)?;
        self.device
            .write_ps1_register(ps1::NON_HI, ((noise >> 16) & 0xFF) as u16)?;
        self.device
            .write_ps1_register(ps1::EON_LO, (reverb & 0xFFFF) as u16)?;
        self.device
            .write_ps1_register(ps1::EON_HI, ((reverb >> 16) & 0xFF) as u16)?;
        Ok(())
    }

    fn update_spucnt(&mut self) -> Result<(), Error> {
        let mut spucnt =
            CONTROL_ENABLE | CONTROL_UNMUTE | (u16::from(self.noise_clock & 0x3F) << 8);
        if self.config.reverb_enabled {
            spucnt |= CONTROL_REVERB;
        }
        self.device.write_ps1_register(ps1::SPUCNT, spucnt)?;
        Ok(())
    }
}

fn resolve_rel_jump(operand_addr: usize, offset: i16, len: usize) -> Result<usize, Error> {
    let target = (operand_addr as i64) + i64::from(offset);
    if target < 0 || target as usize >= len {
        return Err(Error::OutOfRange("AKAO relative jump target out of bounds"));
    }
    Ok(target as usize)
}

fn stereo_pan_gains(pan: u8) -> (u8, u8) {
    if pan <= 64 {
        (127, (pan * 2).min(127))
    } else {
        (((127 - pan) * 2).min(127), 127)
    }
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
    // Smooth cubic Hermite / cosine-like curve matching `0.5 * (1 + cos(pi * t))` closely.
    let t2 = t * t;
    let t3 = t2 * t;
    1.0 - (3.0 * t2 - 2.0 * t3)
}
