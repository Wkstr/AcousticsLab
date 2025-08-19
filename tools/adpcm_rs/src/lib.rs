#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub predictor: i16,
    pub step_index: u8,
}

const STEP_TABLE: [i16; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66, 73, 80, 88, 97,
    107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449, 494, 544, 598, 658, 724, 796,
    876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272, 2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871,
    5358, 5894, 6484, 7132, 7845, 8630, 9493, 10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623,
    27086, 29794, 32767,
];

const INDEX_TABLE: [i8; 16] = [-1, -1, -1, -1, 2, 4, 6, 8, -1, -1, -1, -1, 2, 4, 6, 8];

#[inline]
fn clamp_i32_to_i16(x: i32) -> i16 {
    if x > i32::from(i16::MAX) {
        i16::MAX
    } else if x < i32::from(i16::MIN) {
        i16::MIN
    } else {
        x as i16
    }
}

#[inline]
fn clamp_u8(v: i32, min: i32, max: i32) -> u8 {
    let v = v.clamp(min, max);
    v as u8
}

#[inline]
pub fn decode_nibble(state: &mut State, code: u8) -> i16 {
    let code = code & 0x0F;
    let step = i32::from(STEP_TABLE[state.step_index as usize]);

    let mut diff = step >> 3;

    if (code & 0x04) != 0 {
        diff += step;
    }
    let mut s = step >> 1;
    if (code & 0x02) != 0 {
        diff += s;
    }
    s >>= 1;
    if (code & 0x01) != 0 {
        diff += s;
    }

    if (code & 0x08) != 0 {
        diff = -diff;
    }

    let pred = i32::from(state.predictor) + diff;
    state.predictor = clamp_i32_to_i16(pred);

    let idx = i32::from(state.step_index) + i32::from(INDEX_TABLE[code as usize]);
    state.step_index = clamp_u8(idx, 0, (STEP_TABLE.len() - 1) as i32);

    state.predictor
}

pub fn decode_bytes(state: &mut State, input: &[u8]) -> Vec<i16> {
    let mut out = Vec::with_capacity(input.len() * 2);
    for &b in input {
        let hi = b >> 4;
        out.push(decode_nibble(state, hi));
        let lo = b & 0x0F;
        out.push(decode_nibble(state, lo));
    }
    out
}
