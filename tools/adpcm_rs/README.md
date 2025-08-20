## Build CLI

```
cargo build --release
```

## WASM module

1. Setup target and WASM tools:

    ```sh
    TARGET="wasm32-unknown-unknown"
    rustup target add $TARGET
    cargo install wasm-bindgen-cli
    ```

2. Build for WASM target:

    ```sh
    cargo build --release --features wasm --target $TARGET
    ```

3. Generate WASM bindings:

    - For bundlers (Vite/Webpack/Rollup):
        ```sh
        wasm-bindgen target/wasm32-unknown-unknown/release/adpcm_rs.wasm --target bundler --out-dir pkg --out-name adpcm_rs
        ```
    - For plain browser without bundler:
        ```sh
        wasm-bindgen target/wasm32-unknown-unknown/release/adpcm_rs.wasm --target web --out-dir pkg --out-name adpcm_rs
        ```
    - For Node.js:
        ```sh
        wasm-bindgen target/wasm32-unknown-unknown/release/adpcm_rs.wasm --target nodejs --out-dir pkg --out-name adpcm_rs
        ```

Exports (JS):

- `WasmState`
    - constructor: `new WasmState()`
    - methods: `predictor() -> number`, `step_index() -> number`, `set_predictor(v: number)`, `set_step_index(v: number)`, `reset()`
- Functions
    - `decode_adpcm_chunk(state: WasmState, input: Uint8Array) -> Int16Array`
    - `decodeInto(state: WasmState, input: Uint8Array, out: Int16Array) -> number`
    - `expectedOutputLen(inputLen: number) -> number`


## Python module

1. Build with Python features:

    ```sh
    cargo build --features python
    ```

3. Generate Python bindings:

    ```sh
    pip3 install maturin
    maturin develop --features python
    ```

Exports (Python):

- `PyState`
    - methods: `predictor() -> int`, `step_index() -> int`, `set_predictor(v: int)`, `set_step_index(v: int)`, `reset()`
- Functions
    - `decode_to_s16le_bytes_py(state: PyState, input: bytes) -> bytes`: Returns s16le PCM bytes (2 bytes per sample); efficient for NumPy via `np.frombuffer(..., dtype="<i2")`.
- `decode_adpcm_chunk_py(state: PyState, input: bytes) -> List[int]`
- `expected_output_len_py(n_bytes: int) -> int`
