# vulkane-spirv-parse

SPIR-V module model and reflection for [Vulkane](https://github.com/ciresnave/vulkane), built on
[`rspirv`](https://crates.io/crates/rspirv).

```rust
use vulkane_spirv_parse::Module;

# fn demo(words: &[u32]) -> Result<(), vulkane_spirv_parse::ParseError> {
let module = Module::parse(words)?;
for ep in module.entry_points() {
    println!("{} {:?} local_size={:?}", ep.name, ep.model, ep.local_size);
}
for b in module.bindings() {
    println!("set {:?} binding {} {:?}", b.set, b.binding, b.storage_class);
}
assert_eq!(module.to_words(), words); // byte-identical re-encode
# Ok(()) }
```

Parsing does not validate: a well-formed but invalid module (such as vulkane#106's old `softmax_f64`) loads.
Run `spirv-val` before trusting a module.
