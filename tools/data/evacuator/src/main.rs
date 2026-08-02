use anyhow::{bail, Context, Result};
use arrow_array::{Array, StringArray};
use clap::Parser;
use flate2::read::MultiGzDecoder;
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;
use serde_json::Value;
use sha2::{Digest, Sha256};
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};
use walkdir::WalkDir;

#[derive(Parser, Debug)]
#[command(about = "Streams Parquet/jsonl.gz corpora into resumable JSONL.zst evacuation shards.")]
struct Args {
    /// Directory containing .parquet or .jsonl.gz sources.
    #[arg(long)]
    input: PathBuf,
    /// Output root. One .jsonl.zst file is emitted per input file.
    #[arg(long)]
    output: PathBuf,
    /// Dataset label embedded in every record and used in output paths.
    #[arg(long)]
    source: String,
    /// Number of source files processed concurrently.
    #[arg(long, default_value_t = 4)]
    workers: usize,
    /// zstd compression level; 3 prioritizes network/disk throughput.
    #[arg(long, default_value_t = 3)]
    zstd_level: i32,
}

#[derive(Serialize)]
struct Record<'a> {
    text: &'a str,
    source: &'a str,
    source_file: &'a str,
}

#[derive(Serialize)]
struct ManifestEntry {
    input: String,
    output: String,
    source: String,
    records: u64,
    input_bytes: u64,
    output_bytes: u64,
    output_sha256: String,
}

struct CountingHashWriter<W: Write> {
    inner: W,
    hasher: Sha256,
    bytes: u64,
}

impl<W: Write> CountingHashWriter<W> {
    fn new(inner: W) -> Self { Self { inner, hasher: Sha256::new(), bytes: 0 } }
    fn finish(self) -> (W, u64, String) {
        (self.inner, self.bytes, format!("{:x}", self.hasher.finalize()))
    }
}

impl<W: Write> Write for CountingHashWriter<W> {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        let n = self.inner.write(buf)?;
        self.hasher.update(&buf[..n]);
        self.bytes += n as u64;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> { self.inner.flush() }
}

fn collect_inputs(root: &Path) -> Vec<PathBuf> {
    let mut paths: Vec<_> = WalkDir::new(root).follow_links(false).into_iter().filter_map(|e| e.ok())
        .filter(|e| e.file_type().is_file())
        .filter_map(|e| {
            let p = e.path();
            let n = p.file_name()?.to_string_lossy();
            if n.ends_with(".parquet") || n.ends_with(".jsonl.gz") { Some(p.to_path_buf()) } else { None }
        }).collect();
    paths.sort();
    paths
}

fn output_path(args: &Args, input: &Path) -> Result<PathBuf> {
    let rel = if args.input.is_file() {
        Path::new(input.file_name().context("source file has no name")?)
    } else {
        input.strip_prefix(&args.input).context("input is not under input root")?
    };
    let mut out = args.output.join(&args.source).join(rel);
    let name = out.file_name().context("source file has no name")?.to_string_lossy();
    out.set_file_name(format!("{}.jsonl.zst", name));
    Ok(out)
}

fn write_record<W: Write>(writer: &mut W, text: &str, source: &str, source_file: &str) -> Result<()> {
    if text.trim().is_empty() { return Ok(()); }
    serde_json::to_writer(&mut *writer, &Record { text, source, source_file })?;
    writer.write_all(b"\n")?;
    Ok(())
}

fn text_from_json(value: &Value) -> Option<String> {
    for key in ["text", "content"] {
        if let Some(text) = value.get(key).and_then(Value::as_str) { return Some(text.to_owned()); }
    }
    let conversations = value.get("conversations").or_else(|| value.get("messages")).and_then(Value::as_array);
    let mut text = String::new();
    if let Some(conversations) = conversations {
        for turn in conversations {
            let role = turn.get("from").or_else(|| turn.get("role")).and_then(Value::as_str).unwrap_or("unknown");
            let value = turn.get("value").or_else(|| turn.get("content")).and_then(Value::as_str).unwrap_or("");
            if !value.trim().is_empty() {
                if !text.is_empty() { text.push_str("\n\n"); }
                text.push_str("<|"); text.push_str(role); text.push_str("|>\n"); text.push_str(value);
            }
        }
        return (!text.is_empty()).then_some(text);
    }
    // Instruction corpora often split one sample across these fields.  Preserve
    // each textual field with a label instead of silently emitting zero records.
    for key in ["system", "instruction", "input", "prompt", "question", "context", "article", "title", "output", "response", "answer", "solution", "target", "summary"] {
        if let Some(part) = value.get(key).and_then(Value::as_str) {
            if !part.trim().is_empty() {
                if !text.is_empty() { text.push_str("\n\n"); }
                text.push_str("<|"); text.push_str(key); text.push_str("|>\n"); text.push_str(part);
            }
        }
    }
    if !text.is_empty() { return Some(text); }
    // Never silently discard an unfamiliar JSONL schema during evacuation.
    // Canonical JSON is still a recoverable textual payload for later cleanup.
    Some(value.to_string())
}

fn convert_parquet<W: Write>(input: &Path, writer: &mut W, source: &str, source_file: &str) -> Result<u64> {
    let file = File::open(input)?;
    let builder = ParquetRecordBatchReaderBuilder::try_new(file)?;
    let schema = builder.schema().clone();
    let text_index = ["text", "content"].iter().find_map(|name| schema.index_of(name).ok())
        .context("Parquet has neither text nor content column")?;
    let mut reader = builder.with_batch_size(8_192).build()?;
    let mut records = 0;
    while let Some(batch) = reader.next() {
        let batch = batch?;
        let col = batch.column(text_index).as_any().downcast_ref::<StringArray>().context("text column is not UTF-8")?;
        for row in 0..col.len() {
            if !col.is_null(row) {
                let text = col.value(row);
                if !text.trim().is_empty() { write_record(writer, text, source, source_file)?; records += 1; }
            }
        }
    }
    Ok(records)
}

fn write_json_value<W: Write>(writer: &mut W, bytes: &[u8], source: &str, source_file: &str) -> Result<bool> {
    if bytes.iter().all(u8::is_ascii_whitespace) { return Ok(false); }
    match serde_json::from_slice::<Value>(bytes) {
        Ok(value) => if let Some(text) = text_from_json(&value) {
            write_record(writer, &text, source, source_file)?;
            Ok(true)
        } else { Ok(false) },
        Err(error) => {
            eprintln!("warning: preserve malformed or oversized JSON fragment: {error}");
            let raw = String::from_utf8_lossy(bytes);
            write_record(writer, &raw, source, source_file)?;
            Ok(true)
        }
    }
}

fn convert_jsonl_gz<W: Write>(input: &Path, writer: &mut W, source: &str, source_file: &str) -> Result<u64> {
    // Several v11 exports concatenate gzip members. GzDecoder stops after the
    // first member; MultiGzDecoder is required to evacuate the full corpus.
    let decoder = MultiGzDecoder::new(File::open(input)?);
    let mut reader = BufReader::with_capacity(4 * 1024 * 1024, decoder);
    let mut read_buf = vec![0u8; 1024 * 1024];
    let mut line = Vec::with_capacity(64 * 1024);
    let mut records = 0;
    const MAX_JSON_LINE: usize = 8 * 1024 * 1024;
    loop {
        let n = match reader.read(&mut read_buf) {
            Ok(0) => break,
            Ok(n) => n,
            // Preserve the readable prefix of a damaged download. The original
            // source remains untouched and the warning is visible in lane logs.
            Err(error) => { eprintln!("warning: stop at damaged gzip member in {}: {error}", input.display()); break; }
        };
        let mut start = 0;
        while start < n {
            let relative_end = read_buf[start..n].iter().position(|byte| *byte == b'\n');
            let end = relative_end.map(|offset| start + offset).unwrap_or(n);
            line.extend_from_slice(&read_buf[start..end]);
            if line.len() >= MAX_JSON_LINE {
                // A malformed concatenation can contain a multi-GiB line.
                // Emit bounded raw fragments rather than creating an unbounded
                // allocation or losing the readable payload.
                if write_json_value(writer, &line, source, source_file)? { records += 1; }
                line.clear();
            }
            match relative_end {
                Some(_) => {
                    if write_json_value(writer, &line, source, source_file)? { records += 1; }
                    line.clear();
                    start = end + 1;
                }
                None => start = n,
            }
        }
    }
    if !line.is_empty() && write_json_value(writer, &line, source, source_file)? { records += 1; }
    Ok(records)
}

fn convert_one(args: &Args, input: &Path, manifest: &Mutex<BufWriter<File>>) -> Result<()> {
    let output = output_path(args, input)?;
    if output.exists() && output.metadata()?.len() > 0 { return Ok(()); }
    fs::create_dir_all(output.parent().context("missing output parent")?)?;
    let partial = output.with_extension("zst.partial");
    let source_file = if args.input.is_file() {
        input.file_name().context("source file has no name")?.to_string_lossy().replace('\\', "/")
    } else {
        input.strip_prefix(&args.input)?.to_string_lossy().replace('\\', "/")
    };
    let input_bytes = input.metadata()?.len();
    let file = File::create(&partial)?;
    let hash_writer = CountingHashWriter::new(BufWriter::with_capacity(4 * 1024 * 1024, file));
    let mut encoder = zstd::stream::write::Encoder::new(hash_writer, args.zstd_level)?;
    let records = if input.file_name().unwrap().to_string_lossy().ends_with(".parquet") {
        convert_parquet(input, &mut encoder, &args.source, &source_file)?
    } else {
        convert_jsonl_gz(input, &mut encoder, &args.source, &source_file)?
    };
    let hash_writer = encoder.finish()?;
    let (mut file_writer, output_bytes, output_sha256) = hash_writer.finish();
    file_writer.flush()?;
    drop(file_writer);
    fs::rename(&partial, &output)?;
    let entry = ManifestEntry { input: input.display().to_string(), output: output.display().to_string(), source: args.source.clone(), records, input_bytes, output_bytes, output_sha256 };
    let mut manifest = manifest.lock().unwrap();
    serde_json::to_writer(&mut *manifest, &entry)?;
    manifest.write_all(b"\n")?;
    manifest.flush()?;
    Ok(())
}

fn main() -> Result<()> {
    let args = Args::parse();
    if args.workers == 0 { bail!("workers must be at least 1"); }
    let inputs = collect_inputs(&args.input);
    if inputs.is_empty() { bail!("no .parquet or .jsonl.gz files under {}", args.input.display()); }
    fs::create_dir_all(args.output.join(&args.source))?;
    let manifest_path = args.output.join(&args.source).join("manifest.jsonl");
    let manifest = Arc::new(Mutex::new(BufWriter::new(OpenOptions::new().create(true).append(true).open(manifest_path)?)));
    let queue = Arc::new(Mutex::new(inputs));
    let mut handles = Vec::new();
    for _ in 0..args.workers {
        let queue = Arc::clone(&queue); let manifest = Arc::clone(&manifest);
        let local_args = Args { input: args.input.clone(), output: args.output.clone(), source: args.source.clone(), workers: args.workers, zstd_level: args.zstd_level };
        handles.push(std::thread::spawn(move || -> Result<()> {
            loop {
                let item = { queue.lock().unwrap().pop() };
                match item { Some(input) => convert_one(&local_args, &input, &manifest)?, None => return Ok(()) }
            }
        }));
    }
    for handle in handles { handle.join().map_err(|_| anyhow::anyhow!("worker panicked"))??; }
    Ok(())
}
