#![forbid(unsafe_code)]

#[derive(Clone, Copy, Debug, Default)]
pub struct State {
    pub predictor: i16,
    pub step_index: u8,
}

const STEP_TABLE: [i16; 89] = [
    7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 19, 21, 23, 25, 28, 31, 34, 37, 41, 45, 50, 55, 60, 66,
    73, 80, 88, 97, 107, 118, 130, 143, 157, 173, 190, 209, 230, 253, 279, 307, 337, 371, 408, 449,
    494, 544, 598, 658, 724, 796, 876, 963, 1060, 1166, 1282, 1411, 1552, 1707, 1878, 2066, 2272,
    2499, 2749, 3024, 3327, 3660, 4026, 4428, 4871, 5358, 5894, 6484, 7132, 7845, 8630, 9493,
    10442, 11487, 12635, 13899, 15289, 16818, 18500, 20350, 22385, 24623, 27086, 29794, 32767,
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

cfg_if::cfg_if! {
    if #[cfg(feature = "wasm")] {
        use wasm_bindgen::prelude::*;

        #[wasm_bindgen]
        pub struct WasmState {
            predictor: i16,
            step_index: u8,
        }

        #[wasm_bindgen]
        impl WasmState {
            #[wasm_bindgen(constructor)]
            pub fn new() -> WasmState { WasmState { predictor: 0, step_index: 0 } }
            pub fn predictor(&self) -> i16 { self.predictor }
            pub fn step_index(&self) -> u8 { self.step_index }
            pub fn set_predictor(&mut self, v: i16) { self.predictor = v; }
            pub fn set_step_index(&mut self, v: u8) { self.step_index = v.min(88); }

            #[wasm_bindgen(js_name = "reset")]
            pub fn reset_state(&mut self) {
                self.predictor = 0;
                self.step_index = 0;
            }
        }

        #[wasm_bindgen]
        pub fn decode_adpcm_chunk(state: &mut WasmState, input: &[u8]) -> Vec<i16> {
            let mut st = State { predictor: state.predictor, step_index: state.step_index };
            let out = decode_bytes(&mut st, input);
            state.predictor = st.predictor;
            state.step_index = st.step_index;
            out
        }

        #[wasm_bindgen(js_name = "decodeInto")]
        pub fn decode_into(state: &mut WasmState, input: &[u8], out: &mut [i16]) -> usize {
            let mut st = State { predictor: state.predictor, step_index: state.step_index };
            let mut produced = 0usize;
            for &b in input {
                if produced + 2 > out.len() { break; }
                let hi = b >> 4;
                out[produced] = decode_nibble(&mut st, hi);
                let lo = b & 0x0F;
                out[produced + 1] = decode_nibble(&mut st, lo);
                produced += 2;
            }
            state.predictor = st.predictor;
            state.step_index = st.step_index;
            produced
        }

        #[wasm_bindgen(js_name = "expectedOutputLen")]
        pub fn expected_output_len(input_len: usize) -> usize { input_len.saturating_mul(2) }
    }
}

cfg_if::cfg_if! {
    if #[cfg(feature = "python")] {
        use pyo3::prelude::*;
        use pyo3::types::PyBytes;

        #[pyclass]
        pub struct PyState {
            inner: State,
        }

        #[pymethods]
        impl PyState {
            #[new]
            fn new() -> Self { Self { inner: State::default() } }
            #[getter]
            fn predictor(&self) -> i16 { self.inner.predictor }
            #[getter]
            fn step_index(&self) -> u8 { self.inner.step_index }
            #[setter]
            fn set_predictor(&mut self, v: i16) { self.inner.predictor = v; }
            #[setter]
            fn set_step_index(&mut self, v: u8) { self.inner.step_index = v.min(88); }

            fn reset(&mut self) { self.inner = State::default(); }
        }

        #[pyfunction]
        pub fn decode_adpcm_chunk_py(py_state: &mut PyState, input: &[u8]) -> Vec<i16> {
            decode_bytes(&mut py_state.inner, input)
        }

        #[pyfunction]
        pub fn decode_to_s16le_bytes_py(py: Python<'_>, py_state: &mut PyState, input: &[u8]) -> Py<PyBytes> {
            let mut st = py_state.inner;
            let mut out = Vec::with_capacity(input.len() * 2 * 2);
            for &b in input {
                let hi = b >> 4;
                out.extend_from_slice(&decode_nibble(&mut st, hi).to_le_bytes());
                let lo = b & 0x0F;
                out.extend_from_slice(&decode_nibble(&mut st, lo).to_le_bytes());
            }
            py_state.inner = st;
            PyBytes::new(py, &out).unbind()
        }

        #[pyfunction]
        pub fn expected_output_len_py(input_len: usize) -> usize { input_len.saturating_mul(2) }

        #[pymodule]
        fn adpcm_rs(_py: Python<'_>, m: &Bound<'_, PyModule>) -> PyResult<()> {
            m.add_class::<PyState>()?;
            m.add_function(wrap_pyfunction!(decode_adpcm_chunk_py, m)?)?;
            m.add_function(wrap_pyfunction!(decode_to_s16le_bytes_py, m)?)?;
            m.add_function(wrap_pyfunction!(expected_output_len_py, m)?)?;
            Ok(())
        }
    }
}
