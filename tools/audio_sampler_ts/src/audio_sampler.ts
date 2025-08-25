import { SerialPort } from 'serialport';
import { ReadlineParser } from '@serialport/parser-readline';
import { FileWriter } from 'wav';
import * as adpcm from '../../adpcm_rs/pkg/adpcm_rs.js';

interface Options {
    port: string;
    baud: number;
    raw: boolean;
    list: boolean;
    output: string;
    sampleRate: number;
    channels: number;
    bitDepth: number;
}

function parseArgs(argv: string[]): Options {
    const opts: Options = { port: '/dev/ttyACM0', baud: 921600, raw: false, list: false, output: 'output.wav', sampleRate: 44100, channels: 1, bitDepth: 16 };
    for (let i = 2; i < argv.length; i++) {
        const a = argv[i];
        if (a === '-p' || a === '--port') opts.port = argv[++i] ?? opts.port;
        else if (a === '-b' || a === '--baud') opts.baud = parseInt(argv[++i] ?? `${opts.baud}`, 10);
        else if (a === '--raw') opts.raw = true;
        else if (a === '--list') opts.list = true;
        else if (a === '-o' || a === '--output') opts.output = argv[++i] ?? opts.output;
        else if (a === '-sr' || a === '--sample-rate') opts.sampleRate = parseInt(argv[++i] ?? `${opts.sampleRate}`, 10);
        else if (a === '-ch' || a === '--channels') opts.channels = parseInt(argv[++i] ?? `${opts.channels}`, 10);
        else if (a === '-h' || a === '--help') {
            printHelp();
            process.exit(0);
        } else {
            console.warn(`Unknown arg: ${a}`);
        }
    }
    return opts;
}

function printHelp() {
    console.log(`
    Usage: audio_sampler_ts [options]
    Options:
        -p, --port <path>         Serial device path (default: /dev/ttyACM0)
        -b, --baud <num>          Baud rate (default: 921600)
            --raw                 Print raw bytes as hex dump instead of line mode
            --list                List available serial ports and exit
        -o, --output <file>       Output file path (default: output.wav)
        -sr, --sample-rate <num>  Audio sample rate (default: 44100)
        -ch, --channels <num>     Number of audio channels (default: 1)
        -h, --help                Show help
    `);
}

async function listPorts() {
    const ports = await SerialPort.list();
    if (!ports.length) {
        console.log('No serial ports found.');
        return;
    }
    for (const p of ports) {
        const parts = [
            p.path,
            p.manufacturer ? `(mfr: ${p.manufacturer})` : undefined,
            p.vendorId ? `vendorId=${p.vendorId}` : undefined,
            p.productId ? `productId=${p.productId}` : undefined,
        ].filter(Boolean);
        console.log('- ' + parts.join(' '));
    }
}

function hex(buf: Buffer): string {
    return [...buf].map(b => b.toString(16).padStart(2, '0')).join(' ');
}

function decodeSoundPacket(line: string, writer: FileWriter | null, wasmState: adpcm.WasmState) {
    const parts = line.split(' ');
    if (parts.length < 4) {
        process.stderr.write(`Invalid sound packet format: ${line}\n`);
        return;
    }

    const predictor = parseInt(parts[1], 10);
    if (isNaN(predictor) || predictor < -32768 || predictor > 32767) {
        process.stderr.write(`Invalid predictor value: ${parts[1]}\n`);
        return;
    }

    const stepIndex = parseInt(parts[2], 10);
    if (isNaN(stepIndex) || stepIndex < 0 || stepIndex > 88) {
        process.stderr.write(`Invalid step index value: ${parts[2]}\n`);
        return;
    }

    const base64Data = parts[3];
    try {
        const adpcmBuffer = Buffer.from(base64Data, 'base64');

        wasmState.set_predictor(predictor);
        wasmState.set_step_index(stepIndex);

        const pcmI16: Int16Array = adpcm.decode_adpcm_chunk(wasmState, new Uint8Array(adpcmBuffer));
        const pcmBuf = Buffer.from(pcmI16.buffer, pcmI16.byteOffset, pcmI16.byteLength);
        writer?.write(pcmBuf);
        process.stdout.write(`PCM predictor: ${predictor}, step_index: ${stepIndex}, samples: ${pcmI16.length}\n`);

    } catch (err) {
        process.stderr.write(`Failed to decode sound packet: ${parts[3]}\n`);
        return;
    }
}

async function main() {
    const opts = parseArgs(process.argv);
    if (opts.list) {
        await listPorts();
        return;
    }

    console.log(`Opening ${opts.port} @ ${opts.baud}...`);
    const port = new SerialPort({ path: opts.port, baudRate: opts.baud, autoOpen: true });
    port.on('open', () => console.log('Port opened.'));
    port.on('error', (err: Error) => {
        console.error('Serial error:', err.message);
        process.exitCode = 1;
    });

    let writer: FileWriter | null = null;
    if (opts.raw) {
        port.on('data', (chunk: Buffer) => process.stdout.write(hex(chunk) + '\n'));
    } else {
        writer = new FileWriter(opts.output, {
            sampleRate: opts.sampleRate,
            channels: opts.channels,
            bitDepth: opts.bitDepth,
        });
        writer.on('error', (err: Error) => {
            console.error('WAV file write error:', err.message);
        });

        const wasmState = new adpcm.WasmState();
        const parser = port.pipe(new ReadlineParser({ delimiter: '\n' }));
        parser.on('data', (line: string) => {
            if (line.startsWith("ADPCM: ")) {
                decodeSoundPacket(line, writer, wasmState);
            } else {
                process.stdout.write(line.replace(/[\r\n]+$/, '') + '\n');
            }
        });
    }

    const shutdown = () => {
        if (writer) {
            writer.end(() => {
                console.log(`Successfully saved WAV file to: ${opts.output}`);
            });
        }
        if (port.isOpen) port.close(() => process.exit(0));
        else process.exit(0);
    };
    process.on('SIGINT', shutdown);
    process.on('SIGTERM', shutdown);
}

main().catch((e) => {
    console.error(e);
    process.exit(1);
});
