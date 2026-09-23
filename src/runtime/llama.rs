use std::ffi::{CStr, CString, c_char, c_float, c_int, c_void};
use std::fs;
use std::ops::Range;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use libloading::Library;

use crate::config::Device;

#[repr(C)]
struct LlamaModel;
#[repr(C)]
struct LlamaContext;

type ProgressCallback = Option<unsafe extern "C" fn(c_float, *mut c_void) -> bool>;
type EvalCallback = Option<unsafe extern "C" fn(*mut c_void, *mut c_void)>;
type AbortCallback = Option<unsafe extern "C" fn(*mut c_void) -> bool>;

#[repr(C)]
struct LlamaModelParams {
    devices: *mut *mut c_void,
    tensor_buft_overrides: *const c_void,
    n_gpu_layers: c_int,
    split_mode: c_int,
    load_mode: c_int,
    lazy_mode: c_int,
    main_gpu: c_int,
    tensor_split: *const f32,
    progress_callback: ProgressCallback,
    progress_callback_user_data: *mut c_void,
    kv_overrides: *const c_void,
    vocab_only: bool,
    check_tensors: bool,
    use_extra_bufts: bool,
    no_host: bool,
    no_alloc: bool,
    load_mtp: bool,
}

#[repr(C)]
struct LlamaContextParams {
    n_ctx: u32,
    n_batch: u32,
    n_ubatch: u32,
    n_seq_max: u32,
    n_rs_seq: u32,
    n_outputs_max: u32,
    n_outputs_max_per_seq: u32,
    n_threads: c_int,
    n_threads_batch: c_int,
    ctx_type: c_int,
    rope_scaling_type: c_int,
    pooling_type: c_int,
    attention_type: c_int,
    flash_attn_type: c_int,
    rope_freq_base: f32,
    rope_freq_scale: f32,
    yarn_ext_factor: f32,
    yarn_attn_factor: f32,
    yarn_beta_fast: f32,
    yarn_beta_slow: f32,
    yarn_orig_ctx: u32,
    defrag_thold: f32,
    cb_eval: EvalCallback,
    cb_eval_user_data: *mut c_void,
    type_k: c_int,
    type_v: c_int,
    abort_callback: AbortCallback,
    abort_callback_data: *mut c_void,
    embeddings: bool,
    offload_kqv: bool,
    no_perf: bool,
    op_offload: bool,
    swa_full: bool,
    kv_unified: bool,
    samplers: *mut c_void,
    n_samplers: usize,
    ctx_other: *mut LlamaContext,
}

#[repr(C)]
#[derive(Clone, Copy)]
struct LlamaBatch {
    n_tokens: c_int,
    token: *mut c_int,
    embd: *mut f32,
    pos: *mut c_int,
    n_seq_id: *mut c_int,
    seq_id: *mut *mut c_int,
    logits: *mut i8,
}

type VersionFn = unsafe extern "C" fn() -> *const c_char;
type ModelDefaultParamsFn = unsafe extern "C" fn() -> LlamaModelParams;
type ContextDefaultParamsFn = unsafe extern "C" fn() -> LlamaContextParams;
type BackendLoadAllFn = unsafe extern "C" fn();
type BackendLoadAllFromPathFn = unsafe extern "C" fn(*const c_char);
type BackendInitFn = unsafe extern "C" fn();
type BackendFreeFn = unsafe extern "C" fn();
type ModelLoadFn = unsafe extern "C" fn(*const c_char, LlamaModelParams) -> *mut LlamaModel;
type ModelFreeFn = unsafe extern "C" fn(*mut LlamaModel);
type InitContextFn = unsafe extern "C" fn(*mut LlamaModel, LlamaContextParams) -> *mut LlamaContext;
type ContextFreeFn = unsafe extern "C" fn(*mut LlamaContext);
type BatchInitFn = unsafe extern "C" fn(c_int, c_int, c_int) -> LlamaBatch;
type BatchFreeFn = unsafe extern "C" fn(LlamaBatch);
type EncodeFn = unsafe extern "C" fn(*mut LlamaContext, LlamaBatch) -> c_int;
type SetEmbeddingsFn = unsafe extern "C" fn(*mut LlamaContext, bool);
type SetCausalFn = unsafe extern "C" fn(*mut LlamaContext, bool);
type SynchronizeFn = unsafe extern "C" fn(*mut LlamaContext);
type GetEmbeddingFn = unsafe extern "C" fn(*mut LlamaContext, c_int) -> *mut f32;
type ModelNEmbdOutFn = unsafe extern "C" fn(*const LlamaModel) -> c_int;
type ModelNParamsFn = unsafe extern "C" fn(*const LlamaModel) -> u64;

struct Api {
    version: VersionFn,
    model_default_params: ModelDefaultParamsFn,
    context_default_params: ContextDefaultParamsFn,
    backend_load_all: Option<BackendLoadAllFn>,
    backend_load_all_from_path: Option<BackendLoadAllFromPathFn>,
    backend_init: BackendInitFn,
    backend_free: BackendFreeFn,
    model_load: ModelLoadFn,
    model_free: ModelFreeFn,
    init_context: InitContextFn,
    context_free: ContextFreeFn,
    batch_init: BatchInitFn,
    batch_free: BatchFreeFn,
    encode: EncodeFn,
    set_embeddings: SetEmbeddingsFn,
    set_causal: SetCausalFn,
    synchronize: SynchronizeFn,
    get_embedding: GetEmbeddingFn,
    model_n_embd_out: ModelNEmbdOutFn,
    model_n_params: ModelNParamsFn,
}

pub struct LlamaLibrary {
    _library: Library,
    api: Api,
    path: PathBuf,
    version: String,
}

// The C library owns the model/context and all access is serialized by the service mutex.
unsafe impl Send for LlamaLibrary {}
unsafe impl Sync for LlamaLibrary {}

impl LlamaLibrary {
    pub fn load(path: &Path) -> Result<Self> {
        // Loading a dynamic library is inherently unsafe. All symbols are resolved here and
        // the resulting function table remains valid for the lifetime of `_library`.
        let library = unsafe { Library::new(path) }
            .with_context(|| format!("loading llama.cpp library {}", path.display()))?;
        unsafe fn symbol<T: Copy>(library: &Library, name: &[u8]) -> Result<T> {
            let symbol = unsafe { library.get::<T>(name) }
                .map_err(|e| anyhow::anyhow!(e.to_string()))
                .with_context(|| format!("missing llama.cpp symbol {:?}", name))?;
            Ok(*symbol)
        }
        unsafe fn optional_symbol<T: Copy>(library: &Library, name: &[u8]) -> Option<T> {
            unsafe { library.get::<T>(name).ok().map(|symbol| *symbol) }
        }
        let api = unsafe {
            Api {
                version: symbol(&library, b"llama_version\0")?,
                model_default_params: symbol(&library, b"llama_model_default_params\0")?,
                context_default_params: symbol(&library, b"llama_context_default_params\0")?,
                backend_load_all: optional_symbol(&library, b"ggml_backend_load_all\0"),
                backend_load_all_from_path: optional_symbol(
                    &library,
                    b"ggml_backend_load_all_from_path\0",
                ),
                backend_init: symbol(&library, b"llama_backend_init\0")?,
                backend_free: symbol(&library, b"llama_backend_free\0")?,
                model_load: symbol(&library, b"llama_model_load_from_file\0")?,
                model_free: symbol(&library, b"llama_model_free\0")?,
                init_context: symbol(&library, b"llama_init_from_model\0")?,
                context_free: symbol(&library, b"llama_free\0")?,
                batch_init: symbol(&library, b"llama_batch_init\0")?,
                batch_free: symbol(&library, b"llama_batch_free\0")?,
                encode: symbol(&library, b"llama_encode\0")?,
                set_embeddings: symbol(&library, b"llama_set_embeddings\0")?,
                set_causal: symbol(&library, b"llama_set_causal_attn\0")?,
                synchronize: symbol(&library, b"llama_synchronize\0")?,
                get_embedding: symbol(&library, b"llama_get_embeddings_ith\0")?,
                model_n_embd_out: symbol(&library, b"llama_model_n_embd_out\0")?,
                model_n_params: symbol(&library, b"llama_model_n_params\0")?,
            }
        };
        let version = unsafe {
            let ptr = (api.version)();
            if ptr.is_null() {
                "unknown".to_owned()
            } else {
                CStr::from_ptr(ptr).to_string_lossy().into_owned()
            }
        };
        Ok(Self {
            _library: library,
            api,
            path: path.to_path_buf(),
            version,
        })
    }

    pub fn version(&self) -> &str {
        &self.version
    }

    pub fn runtime_info(&self, source: String, device: Device) -> super::RuntimeInfo {
        super::RuntimeInfo {
            runtime: "llama.cpp".to_owned(),
            source,
            version: Some(self.version.clone()),
            path: Some(self.path.clone()),
            platform: std::env::consts::OS.to_owned(),
            architecture: std::env::consts::ARCH.to_owned(),
            backend: device.as_str().to_owned(),
            compatible: true,
            error: None,
        }
    }
}

pub struct LlamaEncoder {
    library: LlamaLibrary,
    model: *mut LlamaModel,
    context: *mut LlamaContext,
    embd: usize,
    micro_batch_tokens: usize,
    max_sequences: usize,
}

unsafe impl Send for LlamaEncoder {}
unsafe impl Sync for LlamaEncoder {}

#[derive(Debug, Clone)]
pub struct EncoderOutput {
    pub hidden: Vec<f32>,
    pub tokens: usize,
    pub width: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct EncodeBatch {
    sequence_range: Range<usize>,
    token_count: usize,
}

struct PackedBatch {
    tokens: Vec<i32>,
    positions: Vec<i32>,
    sequence_ids: Vec<i32>,
}

fn plan_encode_batches(
    sequence_lengths: &[usize],
    max_tokens: usize,
    max_sequences: usize,
) -> Result<Vec<EncodeBatch>> {
    if max_tokens == 0 || max_sequences == 0 {
        bail!("llama context dimensions must be positive")
    }
    if sequence_lengths.len() > max_sequences {
        bail!(
            "request contains {} questions; maximum is {}",
            sequence_lengths.len(),
            max_sequences
        )
    }
    let total_tokens = sequence_lengths
        .iter()
        .try_fold(0usize, |total, length| total.checked_add(*length))
        .context("request token count overflowed")?;
    if total_tokens == 0 || total_tokens > max_tokens.saturating_mul(max_sequences) {
        bail!("request contains too many tokens: {total_tokens}")
    }
    for length in sequence_lengths {
        if *length > max_tokens {
            bail!("one question contains {length} tokens; maximum is {max_tokens}")
        }
    }

    let mut batches = Vec::new();
    let mut batch_start = 0usize;
    let mut batch_tokens = 0usize;
    for (sequence_id, length) in sequence_lengths.iter().copied().enumerate() {
        if length > 0 && batch_tokens > 0 && batch_tokens + length > max_tokens {
            batches.push(EncodeBatch {
                sequence_range: batch_start..sequence_id,
                token_count: batch_tokens,
            });
            batch_start = sequence_id;
            batch_tokens = 0;
        }
        batch_tokens += length;
    }
    if batch_tokens > 0 {
        batches.push(EncodeBatch {
            sequence_range: batch_start..sequence_lengths.len(),
            token_count: batch_tokens,
        });
    }
    Ok(batches)
}

fn pack_batch(sequences: &[Vec<i32>], sequence_range: Range<usize>) -> PackedBatch {
    let token_count = sequence_range
        .clone()
        .map(|sequence_id| sequences[sequence_id].len())
        .sum();
    let mut packed = PackedBatch {
        tokens: Vec::with_capacity(token_count),
        positions: Vec::with_capacity(token_count),
        sequence_ids: Vec::with_capacity(token_count),
    };
    for sequence_id in sequence_range {
        for (position, token) in sequences[sequence_id].iter().copied().enumerate() {
            packed.tokens.push(token);
            packed.positions.push(position as c_int);
            packed.sequence_ids.push(sequence_id as c_int);
        }
    }
    packed
}

impl LlamaEncoder {
    pub fn load(
        library: LlamaLibrary,
        model_path: &Path,
        device: Device,
        max_tokens: usize,
        max_sequences: usize,
    ) -> Result<Self> {
        if max_tokens == 0 || max_sequences == 0 {
            bail!("llama context dimensions must be positive")
        }
        let api = &library.api;
        let runtime_dir = library
            .path
            .parent()
            .map(Path::to_path_buf)
            .context("llama.cpp library has no parent directory")?;
        let runtime_dir_c = CString::new(runtime_dir.to_string_lossy().as_bytes())?;
        if let Some(load_all_from_path) = api.backend_load_all_from_path {
            unsafe { load_all_from_path(runtime_dir_c.as_ptr()) };
        } else if let Some(load_all) = api.backend_load_all {
            unsafe { load_all() };
        }
        unsafe { (api.backend_init)() };
        let model_path_c = CString::new(model_path.to_string_lossy().as_bytes())?;
        let mut model_params = unsafe { (api.model_default_params)() };
        model_params.n_gpu_layers = match device {
            Device::Cpu => 0,
            Device::Auto | Device::Metal | Device::Vulkan | Device::Cuda => -1,
        };
        let model = unsafe { (api.model_load)(model_path_c.as_ptr(), model_params) };
        if model.is_null() {
            unsafe { (api.backend_free)() };
            bail!(
                "llama.cpp could not load GGUF model {}",
                model_path.display()
            );
        }
        let embd = unsafe { (api.model_n_embd_out)(model) };
        if embd <= 0 {
            unsafe { (api.model_free)(model) };
            unsafe { (api.backend_free)() };
            bail!("llama.cpp reported an invalid output embedding dimension")
        }
        let mut context_params = unsafe { (api.context_default_params)() };
        let context_tokens = max_tokens.saturating_mul(max_sequences);
        context_params.n_ctx = context_tokens as u32;
        context_params.n_batch = context_tokens as u32;
        context_params.n_ubatch = max_tokens as u32;
        context_params.n_seq_max = max_sequences as u32;
        context_params.pooling_type = 0; // LLAMA_POOLING_TYPE_NONE
        context_params.attention_type = 1; // LLAMA_ATTENTION_TYPE_NON_CAUSAL
        context_params.embeddings = true;
        let context = unsafe { (api.init_context)(model, context_params) };
        if context.is_null() {
            unsafe { (api.model_free)(model) };
            unsafe { (api.backend_free)() };
            bail!("llama.cpp could not create an encoder context")
        }
        unsafe {
            (api.set_embeddings)(context, true);
            (api.set_causal)(context, false);
        }
        let params = unsafe { (api.model_n_params)(model) };
        tracing::info!(
            model_params = params,
            embedding_dim = embd,
            max_tokens,
            max_sequences,
            device = device.as_str(),
            llama_version = %library.version,
            "llama.cpp encoder loaded"
        );
        Ok(Self {
            library,
            model,
            context,
            embd: embd as usize,
            micro_batch_tokens: max_tokens,
            max_sequences,
        })
    }

    pub fn embedding_dim(&self) -> usize {
        self.embd
    }

    pub fn encode(&mut self, sequences: &[Vec<i32>]) -> Result<Vec<EncoderOutput>> {
        if sequences.is_empty() {
            return Ok(Vec::new());
        }
        let sequence_lengths = sequences.iter().map(Vec::len).collect::<Vec<_>>();
        let batches = plan_encode_batches(
            &sequence_lengths,
            self.micro_batch_tokens,
            self.max_sequences,
        )?;
        let mut outputs = (0..sequences.len())
            .map(|_| None)
            .collect::<Vec<Option<EncoderOutput>>>();

        for batch_plan in batches {
            let packed = pack_batch(sequences, batch_plan.sequence_range.clone());
            debug_assert_eq!(packed.tokens.len(), batch_plan.token_count);
            let token_capacity = c_int::try_from(batch_plan.token_count)
                .context("micro-batch token count does not fit in llama.cpp's integer range")?;
            let sequence_capacity = c_int::try_from(batch_plan.sequence_range.len())
                .context("micro-batch sequence count does not fit in llama.cpp's integer range")?;
            let mut batch =
                unsafe { (self.library.api.batch_init)(token_capacity, 0, sequence_capacity) };
            batch.n_tokens = token_capacity;
            if batch.token.is_null()
                || batch.pos.is_null()
                || batch.n_seq_id.is_null()
                || batch.seq_id.is_null()
                || batch.logits.is_null()
            {
                unsafe { (self.library.api.batch_free)(batch) };
                bail!("llama.cpp returned an incomplete batch")
            }
            for index in 0..batch_plan.token_count {
                unsafe {
                    *batch.token.add(index) = packed.tokens[index];
                    *batch.pos.add(index) = packed.positions[index];
                    *batch.n_seq_id.add(index) = 1;
                    **batch.seq_id.add(index) = packed.sequence_ids[index];
                    *batch.logits.add(index) = 1;
                }
            }
            debug_assert!(batch.n_tokens as usize <= self.micro_batch_tokens);
            let result = unsafe { (self.library.api.encode)(self.context, batch) };
            if result != 0 {
                unsafe { (self.library.api.batch_free)(batch) };
                bail!("llama_encode failed with status {result}")
            }
            unsafe { (self.library.api.synchronize)(self.context) };

            // Embeddings are owned by the llama context and can be overwritten by the next
            // llama_encode call, so copy every output before submitting another micro-batch.
            let mut output_index = 0i32;
            for sequence_id in batch_plan.sequence_range.clone() {
                let sequence = &sequences[sequence_id];
                let mut hidden = Vec::with_capacity(sequence.len() * self.embd);
                for _ in sequence {
                    let ptr =
                        unsafe { (self.library.api.get_embedding)(self.context, output_index) };
                    if ptr.is_null() {
                        unsafe { (self.library.api.batch_free)(batch) };
                        bail!("llama.cpp returned a null token embedding at output {output_index}")
                    }
                    let values = unsafe { std::slice::from_raw_parts(ptr, self.embd) };
                    hidden.extend_from_slice(values);
                    output_index += 1;
                }
                outputs[sequence_id] = Some(EncoderOutput {
                    hidden,
                    tokens: sequence.len(),
                    width: self.embd,
                });
            }
            unsafe { (self.library.api.batch_free)(batch) };
        }

        outputs
            .into_iter()
            .enumerate()
            .map(|(sequence_id, output)| {
                output.with_context(|| {
                    format!("llama.cpp did not return an embedding for question {sequence_id}")
                })
            })
            .collect()
    }
}

impl Drop for LlamaEncoder {
    fn drop(&mut self) {
        unsafe {
            if !self.context.is_null() {
                (self.library.api.context_free)(self.context);
            }
            if !self.model.is_null() {
                (self.library.api.model_free)(self.model);
            }
            (self.library.api.backend_free)();
        }
    }
}

pub fn find_library(home: &Path) -> Result<PathBuf> {
    if home.is_file() {
        return Ok(home.to_path_buf());
    }
    if !home.is_dir() {
        bail!("llama.cpp home does not exist: {}", home.display())
    }
    let mut candidates = Vec::new();
    for directory in [home.to_path_buf(), home.join("lib"), home.join("bin")] {
        for name in library_names() {
            candidates.push(directory.join(name));
        }
        if let Some(path) = find_library_in_directory(&directory)? {
            return Ok(path);
        }
    }
    if let Some(path) = find_cached_library(home)? {
        return Ok(path);
    }
    candidates
        .into_iter()
        .find(|path| path.is_file())
        .context("no libllama library found")
}

pub fn find_cached_library(root: &Path) -> Result<Option<PathBuf>> {
    if !root.is_dir() {
        return Ok(None);
    }
    let mut stack = vec![root.to_path_buf()];
    while let Some(directory) = stack.pop() {
        for entry in fs::read_dir(&directory)? {
            let path = entry?.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.is_file() && is_library_name(&path) {
                return Ok(Some(path));
            }
        }
    }
    Ok(None)
}

pub fn find_path_library() -> Result<Option<PathBuf>> {
    let Some(path_value) = std::env::var_os("PATH") else {
        return Ok(None);
    };
    for directory in std::env::split_paths(&path_value) {
        if let Some(path) = find_library_in_directory(&directory)? {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

fn library_names() -> &'static [&'static str] {
    #[cfg(target_os = "windows")]
    {
        &["llama.dll", "libllama.dll"]
    }
    #[cfg(target_os = "macos")]
    {
        &["libllama.dylib", "libllama.0.dylib"]
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        &["libllama.so", "libllama.so.0"]
    }
}

fn is_library_name(path: &Path) -> bool {
    let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
        return false;
    };
    if library_names().contains(&name) {
        return true;
    }
    #[cfg(target_os = "macos")]
    {
        name.starts_with("libllama.") && name.ends_with(".dylib")
    }
    #[cfg(all(unix, not(target_os = "macos")))]
    {
        return name.starts_with("libllama.so.");
    }
    #[cfg(target_os = "windows")]
    {
        return false;
    }
}

fn find_library_in_directory(directory: &Path) -> Result<Option<PathBuf>> {
    if !directory.is_dir() {
        return Ok(None);
    }
    for entry in fs::read_dir(directory)? {
        let path = entry?.path();
        if path.is_file() && is_library_name(&path) {
            return Ok(Some(path));
        }
    }
    Ok(None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::domain::{NormalizedQuestion, QuestionKind, SystemOneRequest};

    const MAX_TOKENS: usize = 1024;
    const MAX_SEQUENCES: usize = 16;

    #[test]
    fn choice_and_three_noul_sequences_over_1024_are_batched_in_order() {
        let request: SystemOneRequest = serde_json::from_value(serde_json::json!({
            "state": "large multi-question request",
            "questions": {
                "choice": {"type": "choice", "options": ["allow", "deny"]},
                "noul-1": {"type": "noul"},
                "noul-2": {"type": "noul"},
                "noul-3": {"type": "noul"}
            }
        }))
        .unwrap();
        let questions = request
            .questions
            .iter()
            .map(|(id, question)| NormalizedQuestion::from_api(id.clone(), question).unwrap())
            .collect::<Vec<_>>();
        assert_eq!(
            questions
                .iter()
                .map(|question| question.kind)
                .collect::<Vec<_>>(),
            [
                QuestionKind::Choice,
                QuestionKind::Noul,
                QuestionKind::Noul,
                QuestionKind::Noul
            ]
        );

        // Four tokenized question sequences representing the request above.
        let sequences = (0..4)
            .map(|sequence_id| {
                (0..300)
                    .map(|position| sequence_id * 1000 + position)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let lengths = sequences.iter().map(Vec::len).collect::<Vec<_>>();
        assert!(lengths.iter().sum::<usize>() > MAX_TOKENS);

        let batches = plan_encode_batches(&lengths, MAX_TOKENS, MAX_SEQUENCES).unwrap();
        assert_eq!(
            batches,
            [
                EncodeBatch {
                    sequence_range: 0..3,
                    token_count: 900
                },
                EncodeBatch {
                    sequence_range: 3..4,
                    token_count: 300
                }
            ]
        );

        let mut sequence_order = Vec::new();
        for batch in batches {
            assert!(batch.token_count <= MAX_TOKENS);
            sequence_order.extend(batch.sequence_range.clone());
            let packed = pack_batch(&sequences, batch.sequence_range.clone());
            let mut token_index = 0;
            for sequence_id in batch.sequence_range {
                for (position, token) in sequences[sequence_id].iter().copied().enumerate() {
                    assert_eq!(packed.sequence_ids[token_index], sequence_id as c_int);
                    assert_eq!(packed.positions[token_index], position as c_int);
                    assert_eq!(packed.tokens[token_index], token);
                    token_index += 1;
                }
            }
            assert_eq!(token_index, batch.token_count);
        }
        assert_eq!(sequence_order, [0, 1, 2, 3]);
    }

    #[test]
    fn packs_multiple_batches_without_splitting_or_reordering_sequences() {
        let sequences = [600, 500, 400, 300, 200]
            .into_iter()
            .enumerate()
            .map(|(sequence_id, length)| vec![sequence_id as i32; length])
            .collect::<Vec<_>>();
        let lengths = sequences.iter().map(Vec::len).collect::<Vec<_>>();
        let batches = plan_encode_batches(&lengths, MAX_TOKENS, MAX_SEQUENCES).unwrap();

        assert_eq!(
            batches,
            [
                EncodeBatch {
                    sequence_range: 0..1,
                    token_count: 600
                },
                EncodeBatch {
                    sequence_range: 1..3,
                    token_count: 900
                },
                EncodeBatch {
                    sequence_range: 3..5,
                    token_count: 500
                }
            ]
        );
        let packed_sequence_ids = batches
            .iter()
            .flat_map(|batch| pack_batch(&sequences, batch.sequence_range.clone()).sequence_ids)
            .collect::<Vec<_>>();
        assert_eq!(
            packed_sequence_ids,
            sequences
                .iter()
                .enumerate()
                .flat_map(|(sequence_id, sequence)| {
                    std::iter::repeat_n(sequence_id as c_int, sequence.len())
                })
                .collect::<Vec<_>>()
        );
    }

    #[test]
    fn accepts_existing_maximum_sequence_and_total_token_limits() {
        let lengths = vec![MAX_TOKENS; MAX_SEQUENCES];
        let batches = plan_encode_batches(&lengths, MAX_TOKENS, MAX_SEQUENCES).unwrap();

        assert_eq!(lengths.iter().sum::<usize>(), MAX_TOKENS * MAX_SEQUENCES);
        assert_eq!(batches.len(), MAX_SEQUENCES);
        assert!(batches.iter().all(|batch| batch.token_count == MAX_TOKENS));
        assert_eq!(
            batches
                .iter()
                .flat_map(|batch| batch.sequence_range.clone())
                .collect::<Vec<_>>(),
            (0..MAX_SEQUENCES).collect::<Vec<_>>()
        );
        assert!(plan_encode_batches(&[MAX_TOKENS + 1], MAX_TOKENS, MAX_SEQUENCES).is_err());
        assert!(
            plan_encode_batches(&[MAX_TOKENS; MAX_SEQUENCES + 1], MAX_TOKENS, MAX_SEQUENCES)
                .is_err()
        );
    }
}
