use std::fs::File;
use std::io::{self, Read, Write};
use std::path::PathBuf;

use adpcm_rs::{decode_bytes, State};

fn parse_args() -> Result<(Option<PathBuf>, Option<PathBuf>, State), String> {
    let mut args = std::env::args().skip(1);
    let mut input: Option<PathBuf> = None;
    let mut output: Option<PathBuf> = None;
    let mut state = State::default();

    while let Some(arg) = args.next() {
        match arg.as_str() {
            "-i" | "--input" => {
                let p = args.next().ok_or("missing value for --input")?;
                input = Some(PathBuf::from(p));
            }
            "-o" | "--output" => {
                let p = args.next().ok_or("missing value for --output")?;
                output = Some(PathBuf::from(p));
            }
            "--predictor" => {
                let v = args.next().ok_or("missing value for --predictor")?;
                state.predictor = v.parse::<i32>().map_err(|_| "invalid predictor")? as i16;
            }
            "--index" => {
                let v = args.next().ok_or("missing value for --index")?;
                let parsed = v.parse::<i32>().map_err(|_| "invalid index")?;
                if !(0..=88).contains(&parsed) { return Err("index out of range (0..=88)".into()); }
                state.step_index = parsed as u8;
            }
            "-h" | "--help" => { return Err(String::new()); }
            other => { return Err(format!("unknown argument: {}", other)); }
        }
    }

    Ok((input, output, state))
}

fn print_usage() {
    eprintln!(
        "Usage: adpcm_rs_cli [options]\n\
         Options:\n\
           -i, --input <path>       ADPCM input file (default: stdin)\n\
           -o, --output <path>      PCM s16le output file (default: stdout)\n\
               --predictor <i16>    Initial predictor (default: 0)\n\
               --index <u8>         Initial step index 0..=88 (default: 0)\n\
           -h, --help               Show this help"
    );
}

fn write_pcm_s16le<W: Write>(mut w: W, samples: &[i16]) -> io::Result<()> {
    let mut buf = vec![0u8; samples.len() * 2];
    for (i, &s) in samples.iter().enumerate() {
        let bytes = s.to_le_bytes();
        buf[2 * i] = bytes[0];
        buf[2 * i + 1] = bytes[1];
    }
    w.write_all(&buf)
}

fn main() {
    let (input_path, output_path, mut state) = match parse_args() {
        Ok(v) => v,
        Err(e) => {
            if e.is_empty() { print_usage(); std::process::exit(0); }
            else { eprintln!("Error: {}\n", e); print_usage(); std::process::exit(2); }
        }
    };

    // Read input
    let mut input_data = Vec::new();
    let result = if let Some(p) = input_path {
        match File::open(p) { Ok(mut f) => f.read_to_end(&mut input_data), Err(e) => Err(e) }
    } else {
        io::stdin().lock().read_to_end(&mut input_data)
    };
    if let Err(e) = result { eprintln!("Failed to read input: {}", e); std::process::exit(1); }

    // Decode
    let samples = decode_bytes(&mut state, &input_data);

    // Write output as s16le
    let write_res = if let Some(p) = output_path {
        match File::create(p) { Ok(file) => write_pcm_s16le(file, &samples), Err(e) => Err(e) }
    } else {
        write_pcm_s16le(io::stdout().lock(), &samples)
    };
    if let Err(e) = write_res { eprintln!("Failed to write output: {}", e); std::process::exit(1); }
}
