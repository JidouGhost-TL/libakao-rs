// SPDX-License-Identifier: MIT

use libakao::{
    decode_opcode, detect_version, AkaoPlayer, AkaoSequence, AkaoVersion, AkaoWaveBank,
    Articulation, Error, ExtendedOpcode, FadeCurve, Opcode, RenderConfig, DURATION_TABLE,
};
use libspu2::registers::ps1;
use libspu2::{Device, StereoI16};

/// Builds a 1-block looping PS1 ADPCM waveform (non-zero audible tone).
fn synthetic_adpcm_block() -> Vec<u8> {
    let mut block = vec![0u8; 16];
    block[0] = 0x00; // shift 0, filter 0
    block[1] = 0x03; // loop + loop-start (1-block continuous loop)
    for (i, b) in block[2..].iter_mut().enumerate() {
        *b = if i % 2 == 0 { 0x74 } else { 0x9B };
    }
    block
}

fn synthetic_bank(version: AkaoVersion, spu_base: u32) -> AkaoWaveBank {
    let adpcm = synthetic_adpcm_block();
    let art0 = Articulation {
        start_addr: spu_base,
        loop_addr: spu_base,
        fine_tune: 0x1000,
        adsr1: 0x00FF,
        adsr2: 0x0000,
    };
    let art1 = Articulation {
        start_addr: spu_base,
        loop_addr: spu_base,
        fine_tune: 0x1200,
        adsr1: 0x00FF,
        adsr2: 0x000F,
    };
    AkaoWaveBank::from_parts(version, spu_base, 0, vec![art0, art1], adpcm).unwrap()
}

/// Builds a synthetic sequence buffer for any of the 4 AKAO generations.
fn build_sequence(
    version: AkaoVersion,
    channel_mask: u32,
    tracks: &[&[u8]],
    rhythm_mask: u32,
    secondary_mask: u32,
) -> Vec<u8> {
    let hdr = version.header_size();
    let count = channel_mask.count_ones() as usize;
    assert_eq!(count, tracks.len());
    let table_end = hdr + count * 2;

    let mut track_offsets = Vec::with_capacity(count);
    let mut cursor = table_end;
    for t in tracks {
        track_offsets.push(cursor);
        cursor += t.len();
    }

    let mut buf = vec![0u8; cursor];
    buf[0..4].copy_from_slice(b"AKAO");
    buf[4..6].copy_from_slice(&0x0042u16.to_le_bytes()); // song_id
    buf[6..8].copy_from_slice(&(cursor as u16).to_le_bytes()); // declared_length
    buf[8..10].copy_from_slice(&0x0003u16.to_le_bytes()); // reverb_type

    match version {
        AkaoVersion::V1 => {
            buf[0x10..0x14].copy_from_slice(&(channel_mask & 0x00FF_FFFF).to_le_bytes());
        }
        AkaoVersion::V2a | AkaoVersion::V2b => {
            buf[0x10..0x14].copy_from_slice(&(channel_mask & 0x00FF_FFFF).to_le_bytes());
            buf[0x14..0x18].copy_from_slice(&(rhythm_mask & 0x00FF_FFFF).to_le_bytes());
            buf[0x18..0x1C].copy_from_slice(&(secondary_mask & 0x00FF_FFFF).to_le_bytes());
        }
        AkaoVersion::V3 => {
            buf[0x10..0x14].copy_from_slice(&0x7Fu32.to_le_bytes());
            buf[0x14..0x18].copy_from_slice(&0x01u32.to_le_bytes());
            buf[0x20..0x24].copy_from_slice(&channel_mask.to_le_bytes());
            buf[0x24..0x28].copy_from_slice(&rhythm_mask.to_le_bytes());
            buf[0x28..0x2C].copy_from_slice(&secondary_mask.to_le_bytes());
        }
    }

    for (idx, &abs) in track_offsets.iter().enumerate() {
        let entry_addr = hdr + idx * 2;
        let rel = if version.uses_self_relative_offsets() {
            (abs - entry_addr) as u16
        } else {
            (abs - (entry_addr + 2)) as u16
        };
        buf[entry_addr..entry_addr + 2].copy_from_slice(&rel.to_le_bytes());
        buf[abs..abs + tracks[idx].len()].copy_from_slice(tracks[idx]);
    }

    buf
}

#[test]
fn version_geometry_and_timers_match_hardware() {
    assert!((AkaoVersion::V1.timer_hz() - 243.85692068429237).abs() < 1e-6);
    assert_eq!(AkaoVersion::V2a.timer_hz(), 240.0);
    assert_eq!(AkaoVersion::V2b.timer_hz(), 240.0);
    assert_eq!(AkaoVersion::V3.timer_hz(), 240.0);

    assert_eq!(AkaoVersion::V1.header_size(), 0x14);
    assert_eq!(AkaoVersion::V2a.header_size(), 0x20);
    assert_eq!(AkaoVersion::V2b.header_size(), 0x20);
    assert_eq!(AkaoVersion::V3.header_size(), 0x40);

    assert_eq!(AkaoVersion::V1.max_channels(), 24);
    assert_eq!(AkaoVersion::V2a.max_channels(), 24);
    assert_eq!(AkaoVersion::V2b.max_channels(), 24);
    assert_eq!(AkaoVersion::V3.max_channels(), 32);

    assert_eq!(AkaoVersion::V1.articulation_stride(), 64);
    assert_eq!(AkaoVersion::V2a.articulation_stride(), 64);
    assert_eq!(AkaoVersion::V2b.articulation_stride(), 64);
    assert_eq!(AkaoVersion::V3.articulation_stride(), 16);

    assert_eq!(AkaoVersion::V1.extended_prefix(), 0xFC);
    assert_eq!(AkaoVersion::V2a.extended_prefix(), 0xFC);
    assert_eq!(AkaoVersion::V2b.extended_prefix(), 0xFC);
    assert_eq!(AkaoVersion::V3.extended_prefix(), 0xFE);

    assert!(!AkaoVersion::V1.uses_self_relative_offsets());
    assert!(!AkaoVersion::V2a.uses_self_relative_offsets());
    assert!(!AkaoVersion::V2b.uses_self_relative_offsets());
    assert!(AkaoVersion::V3.uses_self_relative_offsets());
}

#[test]
fn sequence_parsing_offsets_and_autodetect_all_versions() {
    let t0: &[u8] = &[0xA1, 0x00, 0x02, 0xA0];
    let t1: &[u8] = &[0xA1, 0x01, 0x0D, 0xA0];

    // V1: 0x14 header, post-entry relative
    let v1_bytes = build_sequence(AkaoVersion::V1, 0b101, &[t0, t1], 0, 0);
    assert_eq!(detect_version(&v1_bytes).unwrap(), AkaoVersion::V1);
    let v1 = AkaoSequence::parse(&v1_bytes, None).unwrap();
    assert_eq!(v1.version, AkaoVersion::V1);
    assert_eq!(v1.song_id, 0x0042);
    assert_eq!(v1.channel_start(0), Some(0x18));
    assert_eq!(v1.channel_start(1), None);
    assert_eq!(v1.channel_start(2), Some(0x18 + t0.len()));

    // V2a / V2b: 0x20 header, post-entry relative
    let v2_bytes = build_sequence(AkaoVersion::V2a, 0b11, &[t0, t1], 0b10, 0b01);
    assert_eq!(detect_version(&v2_bytes).unwrap(), AkaoVersion::V2a);
    let v2b = AkaoSequence::parse(&v2_bytes, Some(AkaoVersion::V2b)).unwrap();
    assert_eq!(v2b.version, AkaoVersion::V2b);
    assert_eq!(v2b.rhythm_mask, 0b10);
    assert_eq!(v2b.secondary_mask, 0b01);
    assert_eq!(v2b.channel_start(0), Some(0x24));
    assert_eq!(v2b.channel_start(1), Some(0x24 + t0.len()));

    // V3: 0x40 header, self-relative offsets, 32-channel mask (channel 0 & channel 28!)
    let v3_mask = (1u32 << 0) | (1u32 << 28);
    let v3_bytes = build_sequence(AkaoVersion::V3, v3_mask, &[t0, t1], 0, 0);
    assert_eq!(detect_version(&v3_bytes).unwrap(), AkaoVersion::V3);
    let v3 = AkaoSequence::parse(&v3_bytes, None).unwrap();
    assert_eq!(v3.version, AkaoVersion::V3);
    assert_eq!(v3.channel_starts.len(), 32);
    assert_eq!(v3.channel_start(0), Some(0x44));
    assert_eq!(v3.channel_start(28), Some(0x44 + t0.len()));
}

#[test]
fn v3_embedded_drum_and_keysplit_tables() {
    // Layout: 0x40 header + 2B channel table + 8B drum + 8B keysplit + track
    let track: &[u8] = &[0xA1, 0x00, 0xF0, 0x10, 0xFF];
    let drum_pos = 0x42usize;
    let keysplit_pos = 0x4Ausize;
    let track_pos = 0x52usize;
    let mut buf = vec![0u8; track_pos + track.len()];
    buf[0..4].copy_from_slice(b"AKAO");
    buf[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
    buf[0x30..0x34].copy_from_slice(&((drum_pos as i32) - 0x30).to_le_bytes());
    buf[0x34..0x38].copy_from_slice(&((keysplit_pos as i32) - 0x34).to_le_bytes());
    // Self-relative start at 0x40: 0x52 - 0x40 = 0x12
    buf[0x40..0x42].copy_from_slice(&((track_pos - 0x40) as u16).to_le_bytes());
    // 8B drum entry
    buf[drum_pos..drum_pos + 8].copy_from_slice(&[1, 60, 0xFF, 0x00, 0x0F, 0x00, 100, 64]);
    // 8B keysplit entry
    buf[keysplit_pos..keysplit_pos + 8].copy_from_slice(&[0, 36, 72, 48, 0xFE, 0x00, 0x05, 0x00]);
    buf[track_pos..].copy_from_slice(track);

    let seq = AkaoSequence::parse(&buf, None).unwrap();
    assert_eq!(seq.version, AkaoVersion::V3);
    assert_eq!(seq.drum_table.len(), 1);
    assert_eq!(seq.drum_table[0].articulation_index, 1);
    assert_eq!(seq.drum_table[0].note, 60);
    assert_eq!(seq.drum_table[0].adsr1, 0x00FF);
    assert_eq!(seq.drum_table[0].adsr2, 0x000F);
    assert_eq!(seq.keysplit_table.len(), 1);
    assert_eq!(seq.keysplit_table[0].low_note, 36);
    assert_eq!(seq.keysplit_table[0].high_note, 72);
    assert_eq!(seq.keysplit_table[0].base_note, 48);
}

#[test]
fn wave_bank_parsing_64b_vs_16b_relocation_and_spu_upload() {
    let adpcm = synthetic_adpcm_block();

    // 1. V1 separate (.DAT 64B + .ALL 16B header)
    let mut dat = vec![0u8; 64];
    dat[0..4].copy_from_slice(&0x2000u32.to_le_bytes());
    dat[4..8].copy_from_slice(&0x2000u32.to_le_bytes());
    dat[8..10].copy_from_slice(&0x1000u16.to_le_bytes());
    dat[12..14].copy_from_slice(&0x00FFu16.to_le_bytes());
    dat[14..16].copy_from_slice(&0x0005u16.to_le_bytes());
    let mut all = vec![0u8; 0x10 + adpcm.len()];
    all[0..4].copy_from_slice(&0x2000u32.to_le_bytes());
    all[4..8].copy_from_slice(&(adpcm.len() as u32).to_le_bytes());
    all[0x10..].copy_from_slice(&adpcm);

    let bank_v1 = AkaoWaveBank::parse_v1_separate(&dat, &all).unwrap();
    assert_eq!(bank_v1.articulations.len(), 1);
    assert_eq!(bank_v1.articulations[0].start_addr, 0x2000);
    assert_eq!(bank_v1.articulations[0].adsr2, 0x0005);

    // 2. V2b unified 64B header + 64B entries + dynamic relocation
    let mut v2b_raw = vec![0u8; 0x40 + 64 * 2 + adpcm.len()];
    v2b_raw[0..4].copy_from_slice(b"AKAO");
    v2b_raw[0x10..0x14].copy_from_slice(&0x4000u32.to_le_bytes());
    v2b_raw[0x14..0x18].copy_from_slice(&(adpcm.len() as u32).to_le_bytes());
    v2b_raw[0x18..0x1C].copy_from_slice(&10u32.to_le_bytes()); // start_idx = 10
    v2b_raw[0x1C..0x20].copy_from_slice(&12u32.to_le_bytes()); // end_idx = 12 (2 entries)
    v2b_raw[0x40..0x44].copy_from_slice(&0x4000u32.to_le_bytes());
    v2b_raw[0x44..0x48].copy_from_slice(&0x4000u32.to_le_bytes());
    v2b_raw[0x40 + 64 * 2..].copy_from_slice(&adpcm);

    let mut bank_v2b = AkaoWaveBank::parse(&v2b_raw, AkaoVersion::V2b).unwrap();
    assert_eq!(bank_v2b.start_index, 10);
    assert_eq!(bank_v2b.articulations.len(), 2);
    assert!(bank_v2b.articulation(9).is_none());
    assert_eq!(bank_v2b.articulation(10).unwrap().start_addr, 0x4000);
    bank_v2b.relocate(0x8000).unwrap();
    assert_eq!(bank_v2b.spu_base_addr, 0x8000);
    assert_eq!(bank_v2b.articulation(10).unwrap().start_addr, 0x8000);

    // 3. V3 unified 64B header + compacted 16B entries (auto-relocated by spu_base_addr)
    let mut v3_raw = vec![0u8; 0x40 + 16 + adpcm.len()];
    v3_raw[0..4].copy_from_slice(b"AKAO");
    v3_raw[0x10..0x14].copy_from_slice(&0x6000u32.to_le_bytes());
    v3_raw[0x14..0x18].copy_from_slice(&(adpcm.len() as u32).to_le_bytes());
    v3_raw[0x18..0x1C].copy_from_slice(&0u32.to_le_bytes());
    v3_raw[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes()); // count = 1
                                                             // Relative start_addr = 0x00, loop_addr = 0x00 -> relocated to 0x6000
    v3_raw[0x48..0x4A].copy_from_slice(&0x1000u16.to_le_bytes());
    v3_raw[0x4C..0x4E].copy_from_slice(&0x00FFu16.to_le_bytes());
    v3_raw[0x4E..0x50].copy_from_slice(&0x0001u16.to_le_bytes());
    v3_raw[0x50..].copy_from_slice(&adpcm);

    let bank_v3 = AkaoWaveBank::parse(&v3_raw, AkaoVersion::V3).unwrap();
    assert_eq!(bank_v3.articulations[0].start_addr, 0x6000);
    assert_eq!(bank_v3.articulations[0].loop_addr, 0x6000);

    let mut dev = Device::new_ps1();
    bank_v3.upload_to_spu(&mut dev).unwrap();
    let mut readback = vec![0u8; adpcm.len()];
    dev.read_ram(0x6000, &mut readback).unwrap();
    assert_eq!(readback, adpcm);
}

#[test]
fn opcode_decoder_handles_v1_v2_vs_v3_high_range() {
    // V1/V2: 0xFC is extended prefix
    let v1_stream = [0x02, 0x84, 0x8F, 0xFC, 0x06, 0x04, 0x00, 0xA0];
    let (op, next) = decode_opcode(&v1_stream, 0, AkaoVersion::V1).unwrap();
    assert_eq!(
        op,
        Opcode::Note {
            semitone: 0,
            duration: DURATION_TABLE[2]
        }
    );
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(
        op,
        Opcode::Tie {
            duration: DURATION_TABLE[0]
        }
    );
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(
        op,
        Opcode::Rest {
            duration: DURATION_TABLE[0]
        }
    );
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::Jump(4)));
    let (op, _) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::EndOfTrack);

    // V3: 0xF0..0xFD explicit durations + 0xFE extended prefix (with subroutines) + 0xFF halt
    let v3_stream = [
        0xF5, 19, // F note, explicit duration 19
        0xFC, 7, // explicit tie, duration 7
        0xFD, 5, // explicit rest, duration 5
        0xFE, 0x0E, 0x10, 0x00, // subroutine call +16
        0xFE, 0x0F, // subroutine return
        0xFF, // halt alias
    ];
    let (op, next) = decode_opcode(&v3_stream, 0, AkaoVersion::V3).unwrap();
    assert_eq!(
        op,
        Opcode::Note {
            semitone: 5,
            duration: 19
        }
    );
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Tie { duration: 7 });
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Rest { duration: 5 });
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::SubroutineCall(16)));
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::SubroutineReturn));
    let (op, _) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::EndOfTrack);
}

#[test]
fn four_deep_nested_loop_stack_and_v3_subroutine_call_return() {
    // 4-deep nested loop: 2 * 2 * 2 * 2 = 16 notes of duration 3 ticks each, then halt.
    let track_nested: &[u8] = &[
        0xE8, 0x00, 0x80, // fast tempo (0x8000)
        0xA1, 0x00, // inst 0
        0xC8, // loop 1
        0xC8, // loop 2
        0xC8, // loop 3
        0xC8, // loop 4
        0x06, // C, 3 ticks
        0xC9, 0x02, // end loop 4 (x2)
        0xC9, 0x02, // end loop 3 (x2)
        0xC9, 0x02, // end loop 2 (x2)
        0xC9, 0x02, // end loop 1 (x2)
        0xA0, // halt
    ];
    let seq_bytes = build_sequence(AkaoVersion::V2a, 1, &[track_nested], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V2a)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V2a, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 2.0,
        ..RenderConfig::default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.reached_end);
    assert!(res.samples.iter().any(|&s| s != 0));
    // 16 notes * 3 ticks = 48 musical ticks at 0x8000 tempo -> ~96 timer ticks @ 240 Hz = 0.4 s (~17640 frames)
    assert!(
        res.frames_rendered > 10_000 && res.frames_rendered < 30_000,
        "expected ~17640 frames for 16x3-tick 4-deep loop, got {}",
        res.frames_rendered
    );

    // V3 subroutine call (`0xFE 0x0E`) & return (`0xFE 0x0F`):
    // Main: inst 0, call sub (+5 from operand addr -> skips over 0x00, 0xA0), note E (dur 6), halt.
    // Sub:  note C (dur 6), octave up, return.
    // Byte offsets inside track:
    // 0..2: A1 00
    // 2..6: FE 0E <rel: i16> (operand at 4, next_pc = 6)
    // 6..8: F4 06 (note E, dur 6)
    // 8:    FF    (halt)
    // 9..13: F0 06 A6 FE 0F (subroutine! rel = 9 - 4 = +5)
    let track_sub: &[u8] = &[
        0xA1, 0x00, 0xFE, 0x0E, 0x05, 0x00, 0xF4, 0x06, 0xFF, 0xF0, 0x06, 0xA6, 0xFE, 0x0F,
    ];
    let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track_sub], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.reached_end);
    // Subroutine executed `OctaveUp` (0xA6), so channel 0 octave moved from 4 -> 5!
    assert_eq!(player.channel_snapshot(0).unwrap().octave, 5);
    assert!(res.samples.iter().any(|&s| s != 0));
}

#[test]
fn slides_lfos_pmon_non_phase_invert_and_reverb_registers() {
    // Channel 0: carrier + reverb + vibrato + tremolo + autopan + volume/pan/pitch slides + phase invert
    let ch0: &[u8] = &[
        0xA1, 0x00, // inst 0
        0xA8, 0x40, // vol 64
        0xA9, 0x08, 0x7F, // vol slide -> 127 over 8 ticks
        0xAA, 0x20, // pan 32
        0xAB, 0x08, 0x60, // pan slide -> 96 over 8 ticks
        0xA4, 0x08, 0x02, // pitch bend +2 semitones over 8 ticks
        0xB4, 0x00, 0x01, 0x00, // vibrato: delay 0, rate 1, sine
        0xB5, 0x40, // vibrato depth
        0xB8, 0x00, 0x01, 0x01, // tremolo: delay 0, rate 1, triangle
        0xB9, 0x30, // tremolo depth
        0xBC, 0x01, 0x02, // autopan: rate 1, square
        0xBD, 0x20, // autopan depth
        0xC2, // reverb ON
        0xDC, // phase invert ON
        0x03, // note C, 24 ticks
        0xA0, // halt
    ];
    // Channel 1: PMON + NON + noise clock
    let ch1: &[u8] = &[
        0xA1, 0x00, // inst 0
        0xAC, 0x1A, // noise clock = 0x1A
        0xC4, // noise ON
        0xC6, // PMON ON (voice 1 modulated by voice 0)
        0x03, // note C, 24 ticks
        0xA0, // halt
    ];

    let seq_bytes = build_sequence(AkaoVersion::V2b, 0b11, &[ch0, ch1], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V2b)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V2b, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 1.0,
        ..RenderConfig::default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();

    // Render partway through the 24-tick note so all registers are live.
    let mut pcm = vec![0i16; 4096 * 2];
    let frames = player.render_samples(&mut pcm).unwrap();
    assert!(frames > 0);
    assert!(pcm.iter().any(|&s| s != 0));

    let dev = player.device();
    // EON bit 0 set (reverb on ch0), NON bit 1 set (noise on ch1), PMON bit 1 set (pmon on ch1)
    assert_eq!(dev.read_ps1_register(ps1::EON_LO).unwrap() & 0x01, 0x01);
    assert_eq!(dev.read_ps1_register(ps1::NON_LO).unwrap() & 0x02, 0x02);
    assert_eq!(dev.read_ps1_register(ps1::PMON_LO).unwrap() & 0x02, 0x02);
    // Noise clock bits 8..13 in SPUCNT == 0x1A
    let spucnt = dev.read_ps1_register(ps1::SPUCNT).unwrap();
    assert_eq!((spucnt >> 8) & 0x3F, 0x1A);
    // Phase invert on ch0 -> V_VOLR bit 14 is set (15-bit two's-complement negative fixed volume)
    let vol_r = dev.read_ps1_register(ps1::V_VOLR).unwrap();
    assert_ne!(
        vol_r & 0x4000,
        0,
        "expected negative two's-complement V_VOLR for phase invert, got 0x{vol_r:04X}"
    );
    // Slides advanced volume & pan toward targets
    let snap0 = player.channel_snapshot(0).unwrap();
    assert!(snap0.volume > 0x40);
    assert!(snap0.pan > 0x20);
    assert!(snap0.phase_invert);
    assert!(snap0.reverb);
}

#[test]
fn linear_per_sequence_voice_volume_fade_out_preserves_mvoll_mvolr() {
    // Infinite looping track: C note (6 ticks) -> infinite loop `0xCA`
    let track: &[u8] = &[
        0xE8, 0x00, 0x80, // fast tempo
        0xA1, 0x00, // inst 0
        0xA8, 0x7F, // vol 127
        0xC8, // loop start
        0x05, // C, 6 ticks
        0xCA, // infinite loop
    ];
    let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let cfg = RenderConfig {
        target_loops: 2,
        fade_seconds: 0.3,
        tail_seconds: 0.05,
        max_seconds: 5.0,
        fade_curve: FadeCurve::Linear,
        master_volume: 0x3ABC,
        ..RenderConfig::default()
    };

    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let mut seq_vols = Vec::new();
    let mut buf = [0i16; 1024];
    let mut total_pcm_nonzero = false;

    while !player.is_finished() {
        let got = player.render_samples(&mut buf).unwrap();
        if buf[..got * 2].iter().any(|&s| s != 0) {
            total_pcm_nonzero = true;
        }
        seq_vols.push(player.sequence_volume());
        // Hardware-faithful rule: MVOLL/MVOLR must NEVER be touched by sequence fade-out!
        assert_eq!(
            player.device().read_ps1_register(ps1::MVOLL).unwrap(),
            0x3ABC
        );
        assert_eq!(
            player.device().read_ps1_register(ps1::MVOLR).unwrap(),
            0x3ABC
        );
    }

    assert!(total_pcm_nonzero);
    assert!(player.loops_completed() >= 2);
    // Sequence volume started at 127, monotonically decreased during fade, and reached 0.
    assert_eq!(*seq_vols.first().unwrap(), 127);
    assert_eq!(*seq_vols.last().unwrap(), 0);
    assert!(seq_vols.windows(2).all(|w| w[0] >= w[1]));
    // Voice volume reached 0 at end of fade while MVOLL/MVOLR stayed 0x3ABC.
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLL).unwrap(), 0);
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLR).unwrap(), 0);
}

#[test]
fn end_to_end_pcm_render_all_four_versions_and_streaming_parity() {
    let track_v1_v2: &[u8] = &[0xA1, 0x00, 0xA8, 0x7F, 0x04, 0x0F, 0x1A, 0xA0];
    let track_v3: &[u8] = &[0xA1, 0x00, 0xA8, 0x7F, 0xF0, 12, 0xF4, 12, 0xF7, 12, 0xFF];

    for version in [
        AkaoVersion::V1,
        AkaoVersion::V2a,
        AkaoVersion::V2b,
        AkaoVersion::V3,
    ] {
        let trk = if version == AkaoVersion::V3 {
            track_v3
        } else {
            track_v1_v2
        };
        let seq_bytes = build_sequence(version, 1, &[trk], 0, 0);
        let seq_a = AkaoSequence::parse(&seq_bytes, Some(version)).unwrap();
        let seq_b = seq_a.clone();
        let bank_a = synthetic_bank(version, 0x1000);
        let bank_b = bank_a.clone();

        let cfg = RenderConfig {
            target_loops: 0,
            tail_seconds: 0.02,
            max_seconds: 2.0,
            ..RenderConfig::default()
        };

        let mut p_all = AkaoPlayer::new(seq_a, bank_a, cfg).unwrap();
        let res = p_all.render_all().unwrap();
        assert!(res.reached_end, "{version:?} should halt cleanly");
        assert!(
            res.samples.iter().any(|&s| s != 0),
            "{version:?} should render non-zero stereo PCM"
        );

        // Verify streaming `render_frames` produces bit-identical output.
        let mut p_stream = AkaoPlayer::new(seq_b, bank_b, cfg).unwrap();
        let mut streamed = Vec::new();
        let mut frame_buf = [StereoI16::default(); 128];
        while !p_stream.is_finished() {
            let n = p_stream.render_frames(&mut frame_buf).unwrap();
            for f in &frame_buf[..n] {
                streamed.push(f.left);
                streamed.push(f.right);
            }
        }
        assert_eq!(
            res.samples, streamed,
            "{version:?} streaming parity mismatch"
        );
    }
}

#[test]
fn error_validation_rejects_malformed_inputs() {
    assert_eq!(
        AkaoSequence::parse(b"NOPE0000000000000000", Some(AkaoVersion::V1)),
        Err(Error::InvalidMagic)
    );
    assert_eq!(
        AkaoSequence::parse(b"AKAO", Some(AkaoVersion::V1)),
        Err(Error::UnexpectedEof("AKAO sequence header"))
    );
    let mut empty_mask = vec![0u8; 0x40];
    empty_mask[0..4].copy_from_slice(b"AKAO");
    assert_eq!(
        AkaoSequence::parse(&empty_mask, Some(AkaoVersion::V3)),
        Err(Error::InvalidHeader("channel mask has no active channels"))
    );
    assert_eq!(detect_version(&empty_mask), Err(Error::UnknownVersion));
}
