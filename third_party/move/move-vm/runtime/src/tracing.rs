// Copyright (c) The Diem Core Contributors
// Copyright (c) The Move Contributors
// SPDX-License-Identifier: Apache-2.0
#[cfg(feature = "debugging")]
use crate::debug::DebugContext;
#[cfg(feature = "debugging")]
use crate::{interpreter::InterpreterDebugInterface, loader::LoadedFunction, RuntimeEnvironment};
#[cfg(feature = "debugging")]
use move_vm_types::instr::Instruction;
#[cfg(feature = "debugging")]
use ::{
    move_vm_types::values::{Value, Locals},
    once_cell::sync::Lazy,
    std::{
        cell::RefCell,
        env,
        fs::{File, OpenOptions},
        io::Write,
        sync::Mutex,
    },
};
#[cfg(feature = "debugging")]
const MOVE_VM_TRACING_ENV_VAR_NAME: &str = "MOVE_VM_TRACE";

#[cfg(feature = "debugging")]
const MOVE_VM_STEPPING_ENV_VAR_NAME: &str = "MOVE_VM_STEP";

#[cfg(feature = "debugging")]
static FILE_PATH: Lazy<String> = Lazy::new(|| {
    env::var(MOVE_VM_TRACING_ENV_VAR_NAME).unwrap_or_else(|_| "move_vm_trace.trace".to_string())
});

#[cfg(feature = "debugging")]
pub static TRACING_ENABLED: Lazy<bool> =
    Lazy::new(|| env::var(MOVE_VM_TRACING_ENV_VAR_NAME).is_ok());

#[cfg(feature = "debugging")]
static DEBUGGING_ENABLED: Lazy<bool> =
    Lazy::new(|| env::var(MOVE_VM_STEPPING_ENV_VAR_NAME).is_ok());

#[cfg(feature = "debugging")]
pub static LOGGING_FILE_WRITER: Lazy<Mutex<std::io::BufWriter<File>>> = Lazy::new(|| {
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&*FILE_PATH)
        .unwrap();
    Mutex::new(std::io::BufWriter::with_capacity(
        4096 * 1024, /* 4096KB */
        file,
    ))
});

#[cfg(feature = "debugging")]
static DEBUG_CONTEXT: Lazy<Mutex<DebugContext>> = Lazy::new(|| Mutex::new(DebugContext::new()));


// Thread-local, in-memory pc capture support.
thread_local! {
    static TL_PC_CAPTURE_ENABLED: RefCell<bool> = RefCell::new(false);
    // Packed PCs per step: upper 32 bits = function hash, lower 32 bits = local pc (u16 widened)
    static TL_PC_BUFFER: RefCell<Vec<u64>> = RefCell::new(Vec::new());
}

// Thread-local, in-memory shift event capture support.
thread_local! {
    static TL_SHIFT_CAPTURE_ENABLED: RefCell<bool> = RefCell::new(false);
    static TL_SHIFT_BUFFER: RefCell<Vec<ShiftEvent>> = RefCell::new(Vec::new());
}

#[derive(Clone, Copy, Debug)]
pub enum ShiftOp {
    Shl,
    Shr,
}

#[derive(Clone, Debug)]
pub struct ShiftEvent {
    pub function: String,
    pub pc: u16,
    pub op: ShiftOp,
    pub lhs: String,
    pub rhs: u8,
    pub lost_high_bits: bool,
}

/// Begin capturing shift operations for the current thread.
pub fn begin_shift_capture() {
    TL_SHIFT_CAPTURE_ENABLED.with(|e| *e.borrow_mut() = true);
    TL_SHIFT_BUFFER.with(|buf| buf.borrow_mut().clear());
}

/// Stop capturing and return the captured shift events for the current thread.
pub fn end_shift_capture_take() -> Vec<ShiftEvent> {
    TL_SHIFT_CAPTURE_ENABLED.with(|e| *e.borrow_mut() = false);
    TL_SHIFT_BUFFER.with(|buf| std::mem::take(&mut *buf.borrow_mut()))
}

pub(crate) fn record_shift_event(
    function: &LoadedFunction,
    pc: u16,
    op: ShiftOp,
    lhs: &Value,
    rhs: u8,
) {
    TL_SHIFT_CAPTURE_ENABLED.with(|enabled| {
        if *enabled.borrow() {
            let event = ShiftEvent {
                function: function.name_as_pretty_string(),
                pc,
                op,
                lhs: format_integer_value(lhs),
                rhs,
                lost_high_bits: match op { ShiftOp::Shl => shl_loses_high_bits(lhs, rhs), ShiftOp::Shr => false },
            };
            TL_SHIFT_BUFFER.with(|buf| buf.borrow_mut().push(event));
        }
    });
}

fn format_integer_value(v: &Value) -> String {
    match v {
        Value::U8(x) => x.to_string(),
        Value::U16(x) => x.to_string(),
        Value::U32(x) => x.to_string(),
        Value::U64(x) => x.to_string(),
        Value::U128(x) => x.to_string(),
        Value::U256(x) => format!("{}", x),
        _ => "non-integer".to_string(),
    }
}

fn shl_loses_high_bits(v: &Value, n: u8) -> bool {
    if n == 0 { return false; }
    match v {
        Value::U8(x) => if n < 8 { (*x >> (8 - n)) != 0 } else { false },
        Value::U16(x) => if n < 16 { (*x >> (16 - n)) != 0 } else { false },
        Value::U32(x) => if n < 32 { (*x >> (32 - n)) != 0 } else { false },
        Value::U64(x) => if n < 64 { (*x >> (64 - n)) != 0 } else { false },
        Value::U128(x) => if n < 128 { (*x >> (128 - n)) != 0 } else { false },
        Value::U256(x) => {
            // compare as string against zero after shifting right by (256-n)
            let shift: u8 = (256u32.saturating_sub(n as u32)) as u8;
            let shifted = format!("{}", **x >> shift.into());
            shifted != "0"
        },
        _ => false,
    }
}

/// Begin capturing program counters for the current thread.
pub fn begin_pc_capture() {
    TL_PC_CAPTURE_ENABLED.with(|e| *e.borrow_mut() = true);
    TL_PC_BUFFER.with(|buf| buf.borrow_mut().clear());
}

/// Stop capturing and return the captured program counters for the current thread.
pub fn end_pc_capture_take() -> Vec<u64> {
    TL_PC_CAPTURE_ENABLED.with(|e| *e.borrow_mut() = false);
    TL_PC_BUFFER.with(|buf| std::mem::take(&mut *buf.borrow_mut()))
}


// Only include in debug builds
#[cfg(feature = "debugging")]
pub(crate) fn trace(
    function: &LoadedFunction,
    locals: &Locals,
    pc: u16,
    instr: &Instruction,
    runtime_environment: &RuntimeEnvironment,
    interpreter: &dyn InterpreterDebugInterface,
) {
    TL_PC_CAPTURE_ENABLED.with(|enabled| {
        if *enabled.borrow() {
            // Compute a stable function hash (FNV-1a 32-bit) from module address, module name, and function name.
            fn hash32(data: &[u8]) -> u32 {
                let mut hash: u32 = 0x811C9DC5; // FNV-1a 32-bit offset basis
                for &b in data {
                    hash ^= b as u32;
                    hash = hash.wrapping_mul(0x01000193);
                }
                hash
            }

            let module_id = function.module_or_script_id();
            let func_name = function.name_id();

            // address || module_name || function_name
            let mut bytes = Vec::with_capacity(32 + module_id.name().as_str().len() + func_name.as_str().len());
            bytes.extend_from_slice(module_id.address().as_ref());
            bytes.extend_from_slice(module_id.name().as_str().as_bytes());
            bytes.extend_from_slice(func_name.as_str().as_bytes());
            let function_hash = hash32(&bytes);

            // Pack into u64: upper 32 bits = function_hash, lower 32 bits = local pc (u16 widened)
            let packed = ((function_hash as u64) << 32) | (pc as u32 as u64);
            TL_PC_BUFFER.with(|buf| buf.borrow_mut().push(packed));
        }
    });
    if *TRACING_ENABLED {
        let buf_writer = &mut *LOGGING_FILE_WRITER.lock().unwrap();
        buf_writer
            .write_fmt(format_args!(
                "{},{}\n",
                function.name_as_pretty_string(),
                pc,
            ))
            .unwrap();
        buf_writer.flush().unwrap();
    }
    if *DEBUGGING_ENABLED {
        DEBUG_CONTEXT.lock().unwrap().debug_loop(
            function,
            locals,
            pc,
            instr,
            runtime_environment,
            interpreter,
        );
    }
}

#[macro_export]
macro_rules! trace {
    ($function_desc:expr, $locals:expr, $pc:expr, $instr:tt, $resolver:expr, $interp:expr) => {
        // Only include this code in debug releases
        #[cfg(feature = "debugging")]
        $crate::tracing::trace(&$function_desc, $locals, $pc, &$instr, $resolver, $interp)
    };
}

pub const fn assert_move_vm_tracing_feature_disabled(err_msg: &str) {
    assert!(!cfg!(feature = "debugging"), "{}", err_msg)
}
