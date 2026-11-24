// SPDX-License-Identifier: MIT OR Apache-2.0

//! Audio rendering throughput benchmark for `libakao`.

use std::hint::black_box;
use std::time::Instant;

use libakao::{AkaoPlayer, AkaoSequence, AkaoVersion, AkaoWaveBank, Articulation, RenderConfig};

fn build_bench_assets() -> (AkaoSequence, AkaoWaveBank) {
    let mut block = vec![0u8; 16];
    block[0] = 0x00;
    block[1] = 0x03;
    for (i, b) in block[2..].iter_mut().enumerate() {
        *b = if i % 2 == 0 { 0x74 } else { 0x9B };
    }

    let art = Articulation {
        start_addr: 0x1000,
        loop_addr: 0x1000,
        fine_tune: 0x1000,
        adsr1: 0x00FF,
        adsr2: 0x000F,
    };
    let bank = AkaoWaveBank::from_parts(AkaoVersion::V3, 0x1000, 0, vec![art], block).unwrap();

    // 8 active channels with vibrato, reverb, and looping arpeggios.
    let track: &[u8] = &[
        0xFE, 0x00, 0x00, 0x6A, // SetTempo
        0xA1, 0x00, // inst 0
        0xA8, 0x70, // vol
        0xB4, 0x00, 0x18, 0x00, // vibrato
        0xB5, 0x20, // vibrato depth
        0xC2, // reverb ON
        0xC8, // loop start
        0xF0, 12, 0xF4, 12, 0xF7, 12, 0xFB, 12,   // arpeggio
        0xCA, // infinite loop
    ];

    let channel_mask = 0xFFu32; // 8 channels
    let hdr = 0x40usize;
    let table_end = hdr + 8 * 2;
    let mut buf = vec![0u8; table_end + track.len() * 8];
    buf[0..4].copy_from_slice(b"AKAO");
    buf[8..10].copy_from_slice(&5u16.to_le_bytes()); // Hall reverb preset
    buf[0x10..0x14].copy_from_slice(&0x7Fu32.to_le_bytes());
    buf[0x20..0x24].copy_from_slice(&channel_mask.to_le_bytes());
    for ch in 0..8 {
        let entry_addr = hdr + ch * 2;
        let abs = table_end + ch * track.len();
        let rel = (abs - entry_addr) as u16;
        buf[entry_addr..entry_addr + 2].copy_from_slice(&rel.to_le_bytes());
        buf[abs..abs + track.len()].copy_from_slice(track);
    }

    let seq = AkaoSequence::parse(&buf, Some(AkaoVersion::V3)).unwrap();
    (seq, bank)
}

fn main() {
    const TOTAL_FRAMES: usize = 44_100 * 10; // 10 seconds of audio @ 44.1 kHz
    const CHUNK: usize = 1024;

    let (seq, bank) = build_bench_assets();
    let cfg = RenderConfig {
        target_loops: 0,
        max_seconds: 60.0,
        ..RenderConfig::bgm_default()
    };

    // 1. Interleaved i16
    let mut player = AkaoPlayer::new(seq.clone(), bank.clone(), cfg).unwrap();
    let mut i16_buf = vec![0i16; CHUNK * 2];
    let t0 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player.render_samples(&mut i16_buf[..n * 2]).unwrap();
        black_box(&i16_buf[..got * 2]);
        done += got;
    }
    let dt_i16 = t0.elapsed().as_secs_f64();
    let rt_i16 = (TOTAL_FRAMES as f64 / 44_100.0) / dt_i16;

    // 2. Planar f32 (WebAudio)
    player.reset().unwrap();
    let mut left = vec![0.0f32; CHUNK];
    let mut right = vec![0.0f32; CHUNK];
    let t1 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player
            .render_f32_planar(&mut left[..n], &mut right[..n])
            .unwrap();
        black_box(&left[..got]);
        black_box(&right[..got]);
        done += got;
    }
    let dt_planar = t1.elapsed().as_secs_f64();
    let rt_planar = (TOTAL_FRAMES as f64 / 44_100.0) / dt_planar;

    // 3. Interleaved f32
    player.reset().unwrap();
    let mut f32_buf = vec![0.0f32; CHUNK * 2];
    let t2 = Instant::now();
    let mut done = 0usize;
    while done < TOTAL_FRAMES {
        let n = (TOTAL_FRAMES - done).min(CHUNK);
        let got = player
            .render_f32_interleaved(&mut f32_buf[..n * 2])
            .unwrap();
        black_box(&f32_buf[..got * 2]);
        done += got;
    }
    let dt_inter = t2.elapsed().as_secs_f64();
    let rt_inter = (TOTAL_FRAMES as f64 / 44_100.0) / dt_inter;

    println!(
        "bench_audio (8ch V3 + Hall reverb, 10s @ 44.1kHz = {TOTAL_FRAMES} frames):\n\
         - render_samples (i16):         {:.2} ms ({:.1}x realtime, {:.2} Mframes/s)\n\
         - render_f32_planar (f32):      {:.2} ms ({:.1}x realtime, {:.2} Mframes/s)\n\
         - render_f32_interleaved (f32): {:.2} ms ({:.1}x realtime, {:.2} Mframes/s)",
        dt_i16 * 1000.0,
        rt_i16,
        (TOTAL_FRAMES as f64 / dt_i16) / 1e6,
        dt_planar * 1000.0,
        rt_planar,
        (TOTAL_FRAMES as f64 / dt_planar) / 1e6,
        dt_inter * 1000.0,
        rt_inter,
        (TOTAL_FRAMES as f64 / dt_inter) / 1e6,
    );
}
