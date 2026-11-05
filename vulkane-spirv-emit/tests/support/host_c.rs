// SPDX-License-Identifier: MIT OR Apache-2.0
//! The host reference: every case's `unpopped-cpu-c` kernel in one C program, compiled and run,
//! with each output cell read back as its bit pattern.

use std::fmt::Write as _;
use std::process::Command;

use unpopped_vocab::ElementKind;

use crate::cases::Case;
use crate::cc;

fn ctype(dt: ElementKind) -> &'static str {
    match dt {
        ElementKind::F32 => "float",
        ElementKind::I32 => "int",
        ElementKind::U32 => "unsigned int",
        other => panic!("{other:?} is not run by this oracle"),
    }
}

/// The `main` lines that feed one case's kernel and print `case <index>` then each output cell as
/// eight hex digits of its bit pattern.
fn case_main(ci: usize, case: &Case, kname: &str) -> String {
    let n = case.inputs[0].len();
    let ct = ctype(case.dt);
    let mut s = String::new();
    for (k, col) in case.inputs.iter().enumerate() {
        let lits: Vec<String> = col.iter().map(|w| format!("{w}u")).collect();
        let _ = writeln!(
            s,
            "    static const unsigned in{ci}_{k}[{n}] = {{{}}};",
            lits.join(", ")
        );
    }
    let _ = writeln!(s, "    static unsigned out{ci}[{n}];");
    let args: Vec<String> = (0..case.inputs.len())
        .map(|k| format!("(const {ct}*)in{ci}_{k}"))
        .collect();
    let _ = writeln!(s, "    {kname}({}, ({ct}*)out{ci}, {n});", args.join(", "));
    let _ = writeln!(
        s,
        "    printf(\"case {ci}\\n\"); for (int i = 0; i < {n}; ++i) printf(\"%08x\\n\", out{ci}[i]);"
    );
    s
}

/// One C program holding every case's kernel.
fn c_program(cases: &[Case], sources: &[(String, String)]) -> String {
    let mut s = String::from("#include <stdio.h>\n#include <string.h>\n#include <math.h>\n");
    for (_, src) in sources {
        s.push_str(src);
        s.push('\n');
    }
    s.push_str("int main(void) {\n");
    for (ci, (case, (kname, _))) in cases.iter().zip(sources).enumerate() {
        s.push_str(&case_main(ci, case, kname));
    }
    s.push_str("    return 0;\n}\n");
    s
}

/// Parse the program's output into one column of result bits per case.
fn parse_columns(text: &str) -> Vec<Vec<u32>> {
    let mut cols: Vec<Vec<u32>> = Vec::new();
    for line in text.lines() {
        if line.starts_with("case ") {
            cols.push(Vec::new());
        } else {
            cols.last_mut()
                .expect("a result before any case header")
                .push(u32::from_str_radix(line.trim(), 16).expect("hex cell"));
        }
    }
    cols
}

/// Compile and run the C program; the per-case output columns. `sources` is each case's
/// `(kernel name, C source)`.
pub(crate) fn host_results(
    cc: &cc::CCompiler,
    cases: &[Case],
    sources: &[(String, String)],
) -> Vec<Vec<u32>> {
    let dir = tempfile::tempdir().expect("a private scratch directory");
    let c_file = dir.path().join("oracle.c");
    let exe = dir.path().join("oracle.exe");
    std::fs::write(&c_file, c_program(cases, sources)).unwrap();
    cc.compile(&c_file, &exe, false)
        .unwrap_or_else(|e| panic!("the CPU-C oracle program did not compile:\n{e}"));
    let out = Command::new(&exe).output().expect("run the oracle program");
    assert!(
        out.status.success(),
        "oracle program exited {:?}",
        out.status
    );
    let cols = parse_columns(&String::from_utf8_lossy(&out.stdout));
    assert_eq!(cols.len(), cases.len(), "the oracle printed every case");
    cols
}
