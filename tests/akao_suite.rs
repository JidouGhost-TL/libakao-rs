// SPDX-License-Identifier: MIT OR Apache-2.0

use libakao::{
    decode_opcode, detect_version, AkaoPlayer, AkaoSequence, AkaoSequenceView, AkaoVersion,
    AkaoWaveBank, AkaoWaveBankView, Articulation, Error, ExtendedOpcode, FadeCurve, Opcode,
    RenderConfig, ReverbParams, ReverbPreset, DURATION_TABLE,
};
use libspu2::registers::{ps1, CONTROL_REVERB};
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
        fine_tune: 0x1400, // distinct fine_tune for drum/keysplit resolution tests
        adsr1: 0x00EF,
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
    build_sequence_with_reverb(
        version,
        channel_mask,
        tracks,
        rhythm_mask,
        secondary_mask,
        3,
    )
}

fn build_sequence_with_reverb(
    version: AkaoVersion,
    channel_mask: u32,
    tracks: &[&[u8]],
    rhythm_mask: u32,
    secondary_mask: u32,
    reverb_type: u16,
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
    buf[8..10].copy_from_slice(&reverb_type.to_le_bytes()); // reverb_type

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

    let v1_bytes = build_sequence(AkaoVersion::V1, 0b101, &[t0, t1], 0, 0);
    assert_eq!(detect_version(&v1_bytes).unwrap(), AkaoVersion::V1);
    let v1 = AkaoSequence::parse(&v1_bytes, None).unwrap();
    assert_eq!(v1.version, AkaoVersion::V1);
    assert_eq!(v1.song_id, 0x0042);
    assert_eq!(v1.channel_start(0), Some(0x18));
    assert_eq!(v1.channel_start(1), None);
    assert_eq!(v1.channel_start(2), Some(0x18 + t0.len()));

    let v2_bytes = build_sequence(AkaoVersion::V2a, 0b11, &[t0, t1], 0b10, 0b01);
    assert_eq!(detect_version(&v2_bytes).unwrap(), AkaoVersion::V2a);
    let v2b = AkaoSequence::parse(&v2_bytes, Some(AkaoVersion::V2b)).unwrap();
    assert_eq!(v2b.version, AkaoVersion::V2b);
    assert_eq!(v2b.rhythm_mask, 0b10);
    assert_eq!(v2b.secondary_mask, 0b01);
    assert_eq!(v2b.channel_start(0), Some(0x24));
    assert_eq!(v2b.channel_start(1), Some(0x24 + t0.len()));

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
    let track: &[u8] = &[0xA1, 0x00, 0xF0, 0x10, 0xFF];
    let drum_pos = 0x42usize;
    let keysplit_pos = 0x4Ausize;
    let track_pos = 0x52usize;
    let mut buf = vec![0u8; track_pos + track.len()];
    buf[0..4].copy_from_slice(b"AKAO");
    buf[0x20..0x24].copy_from_slice(&1u32.to_le_bytes());
    buf[0x30..0x34].copy_from_slice(&((drum_pos as i32) - 0x30).to_le_bytes());
    buf[0x34..0x38].copy_from_slice(&((keysplit_pos as i32) - 0x34).to_le_bytes());
    buf[0x40..0x42].copy_from_slice(&((track_pos - 0x40) as u16).to_le_bytes());
    buf[drum_pos..drum_pos + 8].copy_from_slice(&[1, 60, 0xFF, 0x00, 0x0F, 0x00, 100, 64]);
    buf[keysplit_pos..keysplit_pos + 8].copy_from_slice(&[0, 36, 72, 48, 0xFE, 0x00, 0x05, 0x00]);
    buf[track_pos..].copy_from_slice(track);

    let view = AkaoSequenceView::parse(&buf, None).unwrap();
    assert_eq!(view.drum_count(), 1);
    assert_eq!(view.keysplit_count(), 1);
    assert_eq!(view.drum_entry(0).unwrap().note, 60);
    assert_eq!(view.keysplit_entry(0).unwrap().base_note, 48);

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

    let view_v1 = AkaoWaveBankView::parse_v1_separate(&dat, &all).unwrap();
    assert_eq!(view_v1.articulation_count(), 1);
    assert_eq!(view_v1.articulation(0).unwrap().start_addr, 0x2000);
    assert_eq!(view_v1.adpcm_data(), adpcm.as_slice());

    let bank_v1 = AkaoWaveBank::parse_v1_separate(&dat, &all).unwrap();
    assert_eq!(bank_v1.articulations.len(), 1);
    assert_eq!(bank_v1.articulations[0].start_addr, 0x2000);
    assert_eq!(bank_v1.articulations[0].adsr2, 0x0005);

    // 2. V2b unified 64B header + 64B entries + dynamic relocation
    let mut v2b_raw = vec![0u8; 0x40 + 64 * 2 + adpcm.len()];
    v2b_raw[0..4].copy_from_slice(b"AKAO");
    v2b_raw[0x10..0x14].copy_from_slice(&0x4000u32.to_le_bytes());
    v2b_raw[0x14..0x18].copy_from_slice(&(adpcm.len() as u32).to_le_bytes());
    v2b_raw[0x18..0x1C].copy_from_slice(&10u32.to_le_bytes());
    v2b_raw[0x1C..0x20].copy_from_slice(&12u32.to_le_bytes());
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
    v3_raw[0x1C..0x20].copy_from_slice(&1u32.to_le_bytes());
    v3_raw[0x48..0x4A].copy_from_slice(&0x1000u16.to_le_bytes());
    v3_raw[0x4C..0x4E].copy_from_slice(&0x00FFu16.to_le_bytes());
    v3_raw[0x4E..0x50].copy_from_slice(&0x0001u16.to_le_bytes());
    v3_raw[0x50..].copy_from_slice(&adpcm);

    let view_v3 = AkaoWaveBankView::parse(&v3_raw, AkaoVersion::V3).unwrap();
    assert_eq!(view_v3.articulation(0).unwrap().start_addr, 0x6000);

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
fn opcode_decoder_handles_v1_v2_vs_v3_operand_lengths() {
    // Common 0xCE..0xDF exact operand lengths:
    // 0xCE (1B), 0xCF (1B), 0xD0 (0B), 0xD2 (1B), 0xD3 (1B), 0xD4 (0B),
    // 0xDC (1B), 0xDD (2B), 0xDE (2B), 0xDF (2B)
    let common_stream = [
        0xCE, 0x05, // NoiseTimerOn(5) -> 2B total
        0xCF, 0x02, // NoiseTimerDelay(2) -> 2B
        0xD0, // LegatoOn -> 1B
        0xD2, 0x03, // PitchModTimerOn(3) -> 2B
        0xD3, 0x04, // PitchModTimerDelay(4) -> 2B
        0xD4, // PhaseInvertOn -> 1B
        0xDC, 0xFE, // FixNoteLength(-2) -> 2B
        0xDD, 0x08, 0x40, // VibratoDepthSlide -> 3B
        0xDE, 0x08, 0x30, // TremoloDepthSlide -> 3B
        0xDF, 0x08, 0x20, // AutoPanDepthSlide -> 3B
        0xA0,
    ];
    for ver in [
        AkaoVersion::V1,
        AkaoVersion::V2a,
        AkaoVersion::V2b,
        AkaoVersion::V3,
    ] {
        let mut pc = 0;
        let mut ops = Vec::new();
        while pc < common_stream.len() {
            let (op, next) = decode_opcode(&common_stream, pc, ver).unwrap();
            ops.push(op);
            pc = next;
        }
        assert_eq!(pc, common_stream.len());
        assert_eq!(ops.last(), Some(&Opcode::EndOfTrack));
        assert!(matches!(ops[0], Opcode::NoiseTimerOn(5)));
        assert!(matches!(ops[2], Opcode::LegatoOn));
        assert!(matches!(ops[5], Opcode::PhaseInvertOn));
        assert!(matches!(ops[6], Opcode::FixNoteLength(-2)));
        assert!(matches!(
            ops[7],
            Opcode::VibratoDepthSlide {
                duration: 8,
                target: 0x40
            }
        ));
    }

    // V1: direct opcodes 0xE8..0xFE (NO 0xFC prefix!)
    let v1_stream = [
        0x02, 0x84, 0x8F, 0xE8, 0x00, 0x6A, // SetTempo (3B)
        0xEE, 0x04, 0x00, // Jump(+4) (3B)
        0xEC, 0x10, 0x00, // DrumModeOnOffset (3B)
        0xF6, 0x40, // VolumeBalance (2B)
        0xA0,
    ];
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
    assert_eq!(op, Opcode::SetTempo(0x6A00));
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::Jump(4));
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::DrumModeOnOffset(16));
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::VolumeBalance(0x40));
    let (op, next) = decode_opcode(&v1_stream, next, AkaoVersion::V1).unwrap();
    assert_eq!(op, Opcode::EndOfTrack);
    assert_eq!(next, v1_stream.len());

    // V2b: 0xFC extended prefix where 0xFC 0x04 takes 2-byte offset and 0xFC 0x0B takes 4 bytes
    let v2_stream = [
        0xFC, 0x04, 0x10, 0x00, // DrumModeOnOffset(16) -> 4B total
        0xFC, 0x0B, 1, 2, 3, 4, // Reserved4 -> 6B total
        0xFC, 0x06, 0x04, 0x00, // Jump(4) -> 4B total
        0xA0,
    ];
    let (op, next) = decode_opcode(&v2_stream, 0, AkaoVersion::V2b).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::DrumModeOnOffset(16)));
    let (op, next) = decode_opcode(&v2_stream, next, AkaoVersion::V2b).unwrap();
    assert_eq!(
        op,
        Opcode::Extended(ExtendedOpcode::Reserved4(0x0B, [1, 2, 3, 4]))
    );
    let (op, next) = decode_opcode(&v2_stream, next, AkaoVersion::V2b).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::Jump(4)));
    let (op, next) = decode_opcode(&v2_stream, next, AkaoVersion::V2b).unwrap();
    assert_eq!(op, Opcode::EndOfTrack);
    assert_eq!(next, v2_stream.len());

    // V3: 0xE4..0xE6 rate slides, 0xF0..0xFD explicit durations, 0xFE extended prefix (0x04 = 0B, 0x0E/0x0F sub, 0x19 = 3B)
    let v3_stream = [
        0xE4, 0x08, 0x20, // VibratoRateSlide (3B)
        0xF5, 19, // F note, explicit duration 19
        0xFC, 7, // explicit tie, duration 7
        0xFD, 5, // explicit rest, duration 5
        0xFE, 0x04, // DrumModeOn (0 operand bytes on V3!)
        0xFE, 0x0E, 0x10, 0x00, // subroutine call +16
        0xFE, 0x0F, // subroutine return
        0xFE, 0x19, 1, 2, 3,    // Reserved3 (3 operand bytes on V3)
        0xFF, // halt alias
    ];
    let (op, next) = decode_opcode(&v3_stream, 0, AkaoVersion::V3).unwrap();
    assert_eq!(
        op,
        Opcode::VibratoRateSlide {
            duration: 8,
            target: 0x20
        }
    );
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
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
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::DrumModeOn));
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::SubroutineCall(16)));
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::Extended(ExtendedOpcode::SubroutineReturn));
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(
        op,
        Opcode::Extended(ExtendedOpcode::Reserved3(0x19, [1, 2, 3]))
    );
    let (op, next) = decode_opcode(&v3_stream, next, AkaoVersion::V3).unwrap();
    assert_eq!(op, Opcode::EndOfTrack);
    assert_eq!(next, v3_stream.len());
}

#[test]
fn four_deep_nested_loop_stack_loop_end_256_and_v3_subroutine_call_return() {
    // 4-deep nested loop: 2 * 2 * 2 * 2 = 16 notes of duration 3 ticks each, then halt.
    let track_nested: &[u8] = &[
        0xFC, 0x00, 0x00, 0x80, // V2a SetTempo(0x8000)
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
        ..RenderConfig::bgm_default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.reached_end);
    assert!(res.samples.iter().any(|&s| s != 0));
    assert!(
        res.frames_rendered > 10_000 && res.frames_rendered < 30_000,
        "expected ~17640 frames for 16x3-tick 4-deep loop, got {}",
        res.frames_rendered
    );

    // Hardware AKAO LoopEnd(0) == 256 iterations (NOT infinite!): halts cleanly after 256 iterations!
    let track_256: &[u8] = &[
        0xFE, 0x00, 0xFF, 0xFF, // max tempo
        0xA1, 0x00, 0xC8, 0xF0, 1, // 1-tick note
        0xC9, 0x00, // 256 iterations!
        0xFF,
    ];
    let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track_256], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(
        res.reached_end,
        "LoopEnd(0) must complete 256 iterations and halt (not loop infinitely)"
    );
    assert_eq!(res.loops_completed, 0);

    // V3 subroutine call (`0xFE 0x0E`) & return (`0xFE 0x0F`):
    let track_sub: &[u8] = &[
        0xA1, 0x00, 0xFE, 0x0E, 0x05, 0x00, 0xF4, 0x06, 0xFF, 0xF0, 0x06, 0xA6, 0xFE, 0x0F,
    ];
    let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track_sub], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.reached_end);
    assert_eq!(player.channel_snapshot(0).unwrap().octave, 5);
    assert!(res.samples.iter().any(|&s| s != 0));
}

#[test]
fn loop_start_five_deep_overflow_does_not_corrupt_slot_0() {
    // Push 5 nested LoopStarts (max depth = 4). The 5th must be ignored so slot 0 is NOT overwritten!
    // Outer loop (slot 0) starts before `0xA6` (OctaveUp). Inner loops don't touch octave.
    let track: &[u8] = &[
        0xFE, 0x00, 0xFF, 0xFF, // fast tempo
        0xA1, 0x00, 0xC8, // depth 1 (slot 0) -> points to OctaveUp
        0xA6, // OctaveUp (runs twice if slot 0 is intact!)
        0xC8, // depth 2 (slot 1)
        0xC8, // depth 3 (slot 2)
        0xC8, // depth 4 (slot 3)
        0xC8, // 5th LoopStart -> MUST BE IGNORED (must not overwrite slot 0!)
        0xF0, 1, 0xC9, 0x01, // pop slot 3
        0xC9, 0x01, // pop slot 2
        0xC9, 0x01, // pop slot 1
        0xC9, 0x02, // repeat slot 0 once -> runs OctaveUp a second time!
        0xFF,
    ];
    let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 1.0,
        ..RenderConfig::bgm_default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let res = player.render_all().unwrap();
    assert!(res.reached_end);
    // Initial octave 4 + 2 OctaveUps = 6 (if slot 0 had been overwritten, OctaveUp would run only once -> 5).
    assert_eq!(player.channel_snapshot(0).unwrap().octave, 6);
}

#[test]
fn reverb_preset_registers_programmed_and_ram_overlap_guarded() {
    let track: &[u8] = &[0xA1, 0x00, 0xC2, 0xF0, 12, 0xFF];
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 0.5,
        reverb_enabled: true,
        ..RenderConfig::bgm_default()
    };

    // 1. Hall preset (index 5): all 32 RVB registers & RVB_WORK_ADDR must match hardware preset.
    let seq_bytes = build_sequence_with_reverb(AkaoVersion::V3, 1, &[track], 0, 0, 5);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let player = AkaoPlayer::new(seq, bank, cfg).unwrap();

    let expected = ReverbPreset::Hall.params().to_ps1_registers();
    for (idx, &exp_word) in expected.iter().enumerate() {
        let got = player
            .device()
            .read_ps1_register(ps1::RVB_BASE + (idx as u32) * 2)
            .unwrap();
        assert_eq!(got, exp_word, "RVB register {idx} mismatch for Hall preset");
    }
    let expected_work_start =
        (libspu2::PS1_RAM_SIZE - ReverbPreset::Hall.work_area_size() as usize) & !7;
    assert_eq!(
        player
            .device()
            .read_ps1_register(ps1::RVB_WORK_ADDR)
            .unwrap(),
        (expected_work_start >> 3) as u16
    );
    assert_ne!(
        player.device().read_ps1_register(ps1::SPUCNT).unwrap() & CONTROL_REVERB,
        0
    );

    // 2. Overlap guard: place wave bank near top of 512 KiB RAM where Echo (0x18040) would collide!
    let seq_echo_bytes = build_sequence_with_reverb(AkaoVersion::V3, 1, &[track], 0, 0, 7);
    let seq_echo = AkaoSequence::parse(&seq_echo_bytes, Some(AkaoVersion::V3)).unwrap();
    let high_bank = synthetic_bank(AkaoVersion::V3, 0x70000); // ends at 0x70010 > 0x80000 - 0x18040 (0x67FC0)
    let player_guarded = AkaoPlayer::new(seq_echo, high_bank, cfg).unwrap();
    assert_eq!(
        player_guarded
            .device()
            .read_ps1_register(ps1::SPUCNT)
            .unwrap()
            & CONTROL_REVERB,
        0,
        "reverb write must be disabled when work area would overlap ADPCM bank"
    );
    assert_eq!(
        player_guarded
            .device()
            .read_ps1_register(ps1::RVB_BASE)
            .unwrap(),
        ReverbParams::ZERO.d_apf1
    );
}

#[test]
fn v3_drum_map_and_keysplit_all_five_fixes() {
    // Channel 0 (drum mode):
    //   - Note 0 (oct 0, C -> drum index 0): uses art1 (fine_tune 0x1400), note 60, custom ADSR1=0x1234, ADSR2=0x5678, vol=64, pan=0 (hard left).
    //   - Note 1 (oct 0, C# -> drum index 1): uses art0 (fine_tune 0x1000), note 48, adsr1=0/adsr2=0 -> MUST restore art0 default ADSR (0x00FF, 0x0000)!
    //   - Note 2 (oct 0, D -> drum index 2, OUT OF BOUNDS since table has 2 entries!): must NOT wrap to index 0!
    // Channel 1 (normal melodic, inst 0, keysplit NOT enabled):
    //   - Plays note 60 -> MUST NOT be hijacked by keysplit_table (keeps inst 0 / art0 ADSR 0x00FF)!
    // Channel 2 (keysplit enabled via `0xFE 0x14 0x80`):
    //   - Plays note 60 -> MUST match keysplit_table entry (art1, custom ADSR 0x0ABC / 0x0DEF)!
    let ch0: &[u8] = &[
        0xFE, 0x00, 0x00, 0x80, // tempo
        0xA1, 0x00, // inst 0
        0xA5, 0x00, // octave 0
        0xFE, 0x04, // DrumModeOn
        0xF0, 12, // drum idx 0
        0xF1, 12, // drum idx 1 (adsr 0/0 -> restore art0 ADSR)
        0xF2, 12, // drum idx 2 (out of bounds -> no modulo wrap!)
        0xFF,
    ];
    let ch1: &[u8] = &[
        0xA1, 0x00, // normal inst 0 (NO keysplit!)
        0xA5, 0x05, // octave 5 -> note 60
        0xF0, 36, // must NOT be hijacked by keysplit_table
        0xFF,
    ];
    let ch2: &[u8] = &[
        0xFE, 0x14, 0x80, // SelectKeysplitInstrument(0x80)
        0xA5, 0x05, // octave 5 -> note 60
        0xF0, 36, // matches keysplit [36..=72]
        0xFF,
    ];

    let drum_pos = 0x46usize; // 0x40 + 3*2
    let keysplit_pos = drum_pos + 16; // 2 drum entries
    let ch0_pos = keysplit_pos + 8; // 1 keysplit entry
    let ch1_pos = ch0_pos + ch0.len();
    let ch2_pos = ch1_pos + ch1.len();
    let mut buf = vec![0u8; ch2_pos + ch2.len()];
    buf[0..4].copy_from_slice(b"AKAO");
    buf[0x10..0x14].copy_from_slice(&0x7Fu32.to_le_bytes());
    buf[0x20..0x24].copy_from_slice(&0b111u32.to_le_bytes());
    buf[0x30..0x34].copy_from_slice(&((drum_pos as i32) - 0x30).to_le_bytes());
    buf[0x34..0x38].copy_from_slice(&((keysplit_pos as i32) - 0x34).to_le_bytes());
    buf[0x40..0x42].copy_from_slice(&((ch0_pos - 0x40) as u16).to_le_bytes());
    buf[0x42..0x44].copy_from_slice(&((ch1_pos - 0x42) as u16).to_le_bytes());
    buf[0x44..0x46].copy_from_slice(&((ch2_pos - 0x44) as u16).to_le_bytes());

    // Drum entry 0: art=1, note=48, adsr1=0x1234, adsr2=0x5678, vol=64, pan=0 (hard left)
    buf[drum_pos..drum_pos + 8].copy_from_slice(&[1, 48, 0x34, 0x12, 0x78, 0x56, 64, 0]);
    // Drum entry 1: art=0, note=48, adsr1=0, adsr2=0, vol=127, pan=127 (hard right)
    buf[drum_pos + 8..drum_pos + 16].copy_from_slice(&[0, 48, 0, 0, 0, 0, 127, 127]);
    // Keysplit entry 0: art=1, low=36, high=72, base=48, adsr1=0x0ABC, adsr2=0x0DEF
    buf[keysplit_pos..keysplit_pos + 8].copy_from_slice(&[1, 36, 72, 48, 0xBC, 0x0A, 0xEF, 0x0D]);

    buf[ch0_pos..ch1_pos].copy_from_slice(ch0);
    buf[ch1_pos..ch2_pos].copy_from_slice(ch1);
    buf[ch2_pos..].copy_from_slice(ch2);

    let seq = AkaoSequence::parse(&buf, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 2.0,
        ..RenderConfig::bgm_default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();

    // Render 1500 frames (~2 ticks into first note)
    let mut pcm = vec![0i16; 1500 * 2];
    player.render_samples(&mut pcm).unwrap();

    let snap0 = player.channel_snapshot(0).unwrap();
    let snap1 = player.channel_snapshot(1).unwrap();
    let snap2 = player.channel_snapshot(2).unwrap();

    // Fix 1: ch0 resolved art1 (fine_tune=0x1400) -> pitch for note 48 is 0x1400 (NOT 0x1000 from art0!)
    assert_eq!(snap0.pitch, 0x1400);
    // Fix 2: DrumEntry::pan = 0 (hard left) -> V_VOLR == 0 and V_VOLL > 0!
    assert!(player.device().read_ps1_register(ps1::V_VOLL).unwrap() > 0);
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLR).unwrap(), 0);
    // Drum entry 0 ADSR override applied
    assert_eq!(snap0.adsr1, 0x1234);
    assert_eq!(snap0.adsr2, 0x5678);

    // Fix 4: ch1 (normal melodic) NOT hijacked by keysplit_table, while ch2 IS!
    assert_eq!(snap1.adsr1, 0x00FF);
    assert_eq!(snap2.adsr1, 0x0ABC);
    assert_eq!(snap2.adsr2, 0x0DEF);

    // Advance into ch0's 2nd note (drum entry 1: adsr1=0, adsr2=0, art0, pan=127 hard right)
    let mut pcm2 = vec![0i16; 5000 * 2];
    player.render_samples(&mut pcm2).unwrap();
    let snap0_note2 = player.channel_snapshot(0).unwrap();
    // Fix 3: Sticky ADSR was cleared and restored to art0 default (0x00FF, 0x0000)!
    assert_eq!(snap0_note2.adsr1, 0x00FF);
    assert_eq!(snap0_note2.adsr2, 0x0000);
    assert_eq!(snap0_note2.pitch, 0x1000);
    // Pan=127 (hard right) -> V_VOLL == 0 and V_VOLR > 0!
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLL).unwrap(), 0);
    assert!(player.device().read_ps1_register(ps1::V_VOLR).unwrap() > 0);

    // Advance into ch0's 3rd note (drum idx 2, out of bounds -> must NOT wrap to drum entry 0's ADSR 0x1234!)
    let mut pcm3 = vec![0i16; 5000 * 2];
    player.render_samples(&mut pcm3).unwrap();
    let snap0_note3 = player.channel_snapshot(0).unwrap();
    assert_ne!(
        snap0_note3.adsr1, 0x1234,
        "out-of-bounds drum index must not wrap to entry 0"
    );
}

#[test]
fn v3_32_channel_voice_allocator_no_aliasing() {
    // Channel 0 and Channel 24 active simultaneously: with `ch % 24`, both would alias voice 0!
    let ch0: &[u8] = &[0xA1, 0x00, 0xA8, 0x7F, 0xF0, 24, 0xFF];
    let ch24: &[u8] = &[0xA1, 0x00, 0xA8, 0x40, 0xF7, 24, 0xFF];
    let mask = (1u32 << 0) | (1u32 << 24);
    let seq_bytes = build_sequence(AkaoVersion::V3, mask, &[ch0, ch24], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 1.0,
        ..RenderConfig::bgm_default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let mut buf = vec![0i16; 1024 * 2];
    player.render_samples(&mut buf).unwrap();

    let s0 = player.channel_snapshot(0).unwrap();
    let s24 = player.channel_snapshot(24).unwrap();
    assert_eq!(s0.voice, 0);
    assert_ne!(
        s24.voice, 0,
        "channel 24 must NOT alias onto hardware voice 0"
    );
    assert!(s24.voice < 24);
    assert!(s0.voice_state.is_some());
    assert!(s24.voice_state.is_some());

    // Also test 26 simultaneously active channels (>24 hardware voices) without panic or aliasing!
    let track: &[u8] = &[0xA1, 0x00, 0xF0, 12, 0xFF];
    let mask26 = (1u32 << 26) - 1;
    let tracks26: Vec<&[u8]> = vec![track; 26];
    let seq26_bytes = build_sequence(AkaoVersion::V3, mask26, &tracks26, 0, 0);
    let seq26 = AkaoSequence::parse(&seq26_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank26 = synthetic_bank(AkaoVersion::V3, 0x1000);
    let mut p26 = AkaoPlayer::new(seq26, bank26, cfg).unwrap();
    let res = p26.render_all().unwrap();
    assert!(res.reached_end);
    assert!(res.samples.iter().any(|&s| s != 0));
}

#[test]
fn lfo_rate_and_vibrato_depth_bit_0x80() {
    // Compare fine vibrato (depth = 0x20) vs coarse vibrato (depth = 0x80 | 0x20 = 0xA0)
    // with period = 16 ticks (should swing clearly within 4 ticks!).
    let mk_track = |depth: u8| -> Vec<u8> {
        vec![
            0xFE, 0x00, 0x00, 0x80, // tempo
            0xA1, 0x00, // inst 0
            0xB4, 0x00, 16, 0x00, // vibrato: delay 0, full-cycle period 16 ticks, sine
            0xB5, depth, // vibrato depth
            0xF0, 24, // C4 for 24 ticks
            0xFF,
        ]
    };

    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 1.0,
        ..RenderConfig::bgm_default()
    };

    let measure_excursion = |depth: u8| -> i32 {
        let trk = mk_track(depth);
        let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[&trk], 0, 0);
        let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
        let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
        let mut p = AkaoPlayer::new(seq, bank, cfg).unwrap();
        let mut max_dev = 0i32;
        let mut buf = [0i16; 256 * 2];
        for _ in 0..16 {
            p.render_samples(&mut buf).unwrap();
            let pitch = i32::from(p.channel_snapshot(0).unwrap().pitch);
            max_dev = max_dev.max((pitch - 0x1000).abs());
        }
        max_dev
    };

    let fine_dev = measure_excursion(0x20);
    let coarse_dev = measure_excursion(0xA0); // bit 0x80 set!
    assert!(
        fine_dev > 10,
        "LFO should swing visibly within 16-tick period, got fine_dev={fine_dev}"
    );
    assert!(
        coarse_dev >= fine_dev * 3,
        "vibrato depth bit 0x80 (coarse) should be ~4x wider than fine: fine={fine_dev}, coarse={coarse_dev}"
    );
}

#[test]
fn low_sample_rate_below_244hz_no_underflow() {
    let track: &[u8] = &[0xA1, 0x00, 0xF0, 12, 0xFF];
    for rate in [100u32, 60u32, 1u32] {
        let seq_bytes = build_sequence(AkaoVersion::V3, 1, &[track], 0, 0);
        let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
        let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
        let cfg = RenderConfig {
            target_loops: 0,
            tail_seconds: 0.0,
            max_seconds: 1.0,
            sample_rate: rate,
            ..RenderConfig::bgm_default()
        };
        let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
        let res = player.render_all().unwrap();
        assert!(
            res.reached_end,
            "sample_rate={rate} should complete cleanly"
        );
    }
}

#[test]
fn slides_lfos_pmon_non_phase_invert_and_reverb_registers() {
    let ch0: &[u8] = &[
        0xA1, 0x00, // inst 0
        0xA8, 0x40, // vol 64
        0xA9, 0x08, 0x7F, // vol slide -> 127 over 8 ticks
        0xAA, 0x20, // pan 32
        0xAB, 0x08, 0x60, // pan slide -> 96 over 8 ticks
        0xA4, 0x08, 0x02, // pitch bend +2 semitones over 8 ticks
        0xB4, 0x00, 0x08, 0x00, // vibrato: delay 0, period 8, sine
        0xB5, 0x40, // vibrato depth
        0xB8, 0x00, 0x08, 0x01, // tremolo: delay 0, period 8, triangle
        0xB9, 0x30, // tremolo depth
        0xBC, 0x08, 0x02, // autopan: period 8, square
        0xBD, 0x20, // autopan depth
        0xC2, // reverb ON
        0xD4, // phase invert ON (hardware 0xD4!)
        0x03, // note C, 24 ticks
        0xA0, // halt
    ];
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
        ..RenderConfig::bgm_default()
    };
    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();

    // First-tick check: render past the first musical tick (~368 frames @ 44.1kHz, 0x6AAA tempo)
    // and before the second musical tick (~919 frames); volume slide must NOT have advanced prematurely!
    let mut first_tick = vec![0i16; 500 * 2];
    player.render_samples(&mut first_tick).unwrap();
    assert_eq!(
        player.channel_snapshot(0).unwrap().volume,
        0x40,
        "volume slide must not advance prematurely on trigger tick 0"
    );

    let mut pcm = vec![0i16; 4096 * 2];
    let frames = player.render_samples(&mut pcm).unwrap();
    assert!(frames > 0);
    assert!(pcm.iter().any(|&s| s != 0));

    let dev = player.device();
    assert_eq!(dev.read_ps1_register(ps1::EON_LO).unwrap() & 0x01, 0x01);
    assert_eq!(dev.read_ps1_register(ps1::NON_LO).unwrap() & 0x02, 0x02);
    assert_eq!(dev.read_ps1_register(ps1::PMON_LO).unwrap() & 0x02, 0x02);
    let spucnt = dev.read_ps1_register(ps1::SPUCNT).unwrap();
    assert_eq!((spucnt >> 8) & 0x3F, 0x1A);
    let vol_r = dev.read_ps1_register(ps1::V_VOLR).unwrap();
    assert_ne!(
        vol_r & 0x4000,
        0,
        "expected negative two's-complement V_VOLR for phase invert, got 0x{vol_r:04X}"
    );
    let snap0 = player.channel_snapshot(0).unwrap();
    assert!(snap0.volume > 0x40);
    assert!(snap0.pan > 0x20);
    assert!(snap0.phase_invert);
    assert!(snap0.reverb);
}

#[test]
fn linear_per_sequence_voice_volume_fade_out_preserves_mvoll_mvolr() {
    let track: &[u8] = &[
        0xFE, 0x00, 0x00, 0x80, // V3 fast tempo
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
        ..RenderConfig::bgm_default()
    };

    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    let mut seq_vols = Vec::new();
    let mut saw_fading = false;
    let mut buf = [0i16; 1024];
    let mut total_pcm_nonzero = false;

    while !player.is_finished() {
        let got = player.render_samples(&mut buf).unwrap();
        if buf[..got * 2].iter().any(|&s| s != 0) {
            total_pcm_nonzero = true;
        }
        if player.is_fading() {
            saw_fading = true;
        }
        seq_vols.push(player.sequence_volume());
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
    assert!(saw_fading);
    assert!(player.loops_completed() >= 2);
    assert_eq!(*seq_vols.first().unwrap(), 127);
    assert_eq!(*seq_vols.last().unwrap(), 0);
    assert!(seq_vols.windows(2).all(|w| w[0] >= w[1]));
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLL).unwrap(), 0);
    assert_eq!(player.device().read_ps1_register(ps1::V_VOLR).unwrap(), 0);
}

#[test]
fn f32_planar_interleaved_mute_solo_and_reset() {
    let ch0: &[u8] = &[0xA1, 0x00, 0xA8, 0x7F, 0xF0, 18, 0xFF];
    let ch1: &[u8] = &[0xA1, 0x00, 0xA8, 0x7F, 0xF7, 18, 0xFF];
    let seq_bytes = build_sequence(AkaoVersion::V3, 0b11, &[ch0, ch1], 0, 0);
    let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();
    let bank = synthetic_bank(AkaoVersion::V3, 0x1000);
    let cfg = RenderConfig {
        target_loops: 0,
        tail_seconds: 0.0,
        max_seconds: 2.0,
        ..RenderConfig::bgm_default()
    };

    let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
    assert_eq!(player.channel_count(), 32);
    assert!(format!("{player:?}").contains("AkaoPlayer"));

    // 1. Render full i16 baseline
    let first_run = player.render_all().unwrap();
    assert!(first_run.reached_end);
    assert!(player.elapsed_seconds() > 0.0);

    // 2. Reset without re-uploading SPU RAM -> must produce non-zero planar f32 & interleaved f32
    player.reset().unwrap();
    assert_eq!(player.frames_rendered(), 0);
    assert!(!player.is_finished());

    let mut left = vec![0.0f32; 512];
    let mut right = vec![0.0f32; 512];
    let mut planar_frames = 0usize;
    let mut planar_nonzero = false;
    while !player.is_finished() {
        let got = player.render_f32_planar(&mut left, &mut right).unwrap();
        planar_frames += got;
        if left[..got].iter().any(|&x| x != 0.0) || right[..got].iter().any(|&x| x != 0.0) {
            planar_nonzero = true;
        }
        assert!(left[..got].iter().all(|x| x.is_finite()));
        assert!(right[..got].iter().all(|x| x.is_finite()));
    }
    assert_eq!(planar_frames as u64, first_run.frames_rendered);
    assert!(planar_nonzero);

    // 3. Reset again -> interleaved f32
    player.reset().unwrap();
    let mut inter = vec![0.0f32; 512 * 2];
    let mut inter_frames = 0usize;
    let mut inter_nonzero = false;
    while !player.is_finished() {
        let got = player.render_f32_interleaved(&mut inter).unwrap();
        inter_frames += got;
        if inter[..got * 2].iter().any(|&x| x != 0.0) {
            inter_nonzero = true;
        }
    }
    assert_eq!(inter_frames as u64, first_run.frames_rendered);
    assert!(inter_nonzero);

    // 4. Mute & Solo controls
    player.reset().unwrap();
    player.set_channel_muted(0, true);
    assert_eq!(player.mute_mask(), 1);
    assert!(player.channel_snapshot(0).unwrap().muted);
    assert!(!player.channel_snapshot(1).unwrap().muted);

    player.set_mute_mask(0b11);
    assert_eq!(player.mute_mask(), 0b11);
    let muted_run = player.render_all().unwrap();
    assert!(
        muted_run.samples.iter().all(|&s| s == 0),
        "muting all active channels must produce silence"
    );

    player.reset().unwrap();
    player.set_mute_mask(0);
    player.set_solo_mask(0b01);
    assert_eq!(player.solo_mask(), 0b01);
    assert!(!player.channel_snapshot(0).unwrap().muted);
    assert!(player.channel_snapshot(1).unwrap().muted);
    let solo_run = player.render_all().unwrap();
    assert!(solo_run.samples.iter().any(|&s| s != 0));
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
            ..RenderConfig::bgm_default()
        };

        let mut p_all = AkaoPlayer::new(seq_a, bank_a, cfg).unwrap();
        let res = p_all.render_all().unwrap();
        assert!(res.reached_end, "{version:?} should halt cleanly");
        assert!(
            res.samples.iter().any(|&s| s != 0),
            "{version:?} should render non-zero stereo PCM"
        );

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
