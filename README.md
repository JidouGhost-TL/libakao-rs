# libakao

`libakao` is a standalone, `#![no_std]` (`alloc`-compatible), `#![forbid(unsafe_code)]`
pure-Rust PlayStation 1 **AKAO** music sequencer built directly on top of
[`libspu2`](https://github.com/JidouGhost-TL/libspu2-rs).

## Supported AKAO Generations

`libakao` models all four hardware-verified PlayStation 1 AKAO generations via
[`AkaoVersion`]:

| Version | Header Size | Channels | Channel Offsets | Root Counter 2 Timer | Wave Bank Entry | Extended Prefix |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`V1`** | `0x14` | 24 | Post-entry relative (`entry + 2 + val`) | `243.8569 Hz` (`4,233,600 / 17,361`) | 64 B (max 128) | Direct `0xE8..0xFE` |
| **`V2a`** | `0x20` | 24 | Post-entry relative (`entry + 2 + val`) | `240.0000 Hz` (`4,233,600 / 17,640`) | 64 B (max 128) | `0xFC` |
| **`V2b`** | `0x20` | 24 | Post-entry relative (`entry + 2 + val`) | `240.0000 Hz` (`4,233,600 / 17,640`) | 64 B (max 256, relocatable) | `0xFC` |
| **`V3`** | `0x40` | 32 | Self-relative (`0x40 + idx * 2 + val`) | `240.0000 Hz` (`4,233,600 / 17,640`) | 16 B (relocatable) + embedded 8 B drum/keysplit tables | `0xFE` (`0xF0..0xFD` explicit durations, `0xFE 0x0E`/`0x0F` subroutines) |

## Hardware-Faithful Playback

- **Root Counter 2 timing & 16.16 tempo accumulator**: Fractional-sample stepping at
  `243.8569 Hz` (`V1`) or `240.0000 Hz` (`V2a`/`V2b`/`V3`), advancing musical ticks on
  16.16 fixed-point accumulator overflow (`accum += tempo >> 16`).
- **Per-sequence voice-volume fade-out**: Linear fade-out at 60 Hz applied to each active
  voice's `V_VOLL`/`V_VOLR` registers rather than `MVOLL`/`MVOLR`, allowing SPU reverb
  tails and ADSR release envelopes to decay naturally.
- **Full core opcode set**: 12×11 fixed-duration notes, ties, rests, explicit-duration
  `V3` notes/ties/rests, 4-deep loop stack (`0xC8`/`0xC9`/`0xCA`), `V1`/`V2` `0xFC` and
  `V3` `0xFE` extended jumps/loop-breaks/subroutines, slides (volume, pan, pitch, tempo,
  reverb), 3 LFOs (vibrato, tremolo, auto-pan), ADSR overrides, `PMON`, `NON`, reverb,
  and phase inversion.

## Quick Example

```rust
use libakao::{
    AkaoPlayer, AkaoSequence, AkaoVersion, AkaoWaveBank, Articulation, RenderConfig,
};

// Build a minimal 1-channel V3 sequence playing a single note and halting.
let mut seq_bytes = vec![0u8; 0x40 + 2 + 8];
seq_bytes[0..4].copy_from_slice(b"AKAO");
seq_bytes[0x20..0x24].copy_from_slice(&1u32.to_le_bytes()); // channel 0 active
seq_bytes[0x40..0x42].copy_from_slice(&2u16.to_le_bytes()); // self-relative -> 0x42
seq_bytes[0x42..0x49].copy_from_slice(&[
    0xA1, 0x00, // instrument 0
    0xA5, 0x04, // octave 4
    0xA8, 0x7F, // volume 127
    0x02,       // note C, quarter note
    0xA0,       // end of track
][..7]);
seq_bytes[0x49] = 0xA0;

let seq = AkaoSequence::parse(&seq_bytes, Some(AkaoVersion::V3)).unwrap();

// Build a 1-instrument wave bank with a single looping ADPCM block.
let mut adpcm = vec![0u8; 16];
adpcm[0] = 0x00;
adpcm[1] = 0x03; // loop flag
for b in &mut adpcm[2..] {
    *b = 0x77;
}
let art = Articulation {
    start_addr: 0x1000,
    loop_addr: 0x1000,
    fine_tune: 0x1000,
    adsr1: 0x00FF,
    adsr2: 0x0000,
};
let bank = AkaoWaveBank::from_parts(AkaoVersion::V3, 0x1000, 0, vec![art], adpcm).unwrap();

let mut cfg = RenderConfig::default();
cfg.tail_seconds = 0.05;
cfg.max_seconds = 1.0;

let mut player = AkaoPlayer::new(seq, bank, cfg).unwrap();
let rendered = player.render_all().unwrap();
assert!(rendered.samples.iter().any(|&s| s != 0));
```

## License

Dual-licensed under MIT (`LICENSE-MIT`) or Apache-2.0 (`LICENSE-APACHE`), at your option.
