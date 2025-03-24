# AmfiRL
Framework library for reinforcement learning using data model from `amfi` crate.
Crate contains traits and generic implementation.
## Torch dependency
This crate depends on [`tch`](https://crates.io/crates/tch)
which requires that you have `Torch` installed. 
Use guide provided by that crate.

For me the procedure looks like this:
1. Unpacking compiled [`torchlib`](https://pytorch.org/get-started/locally/) for C++/Java;
You will get structure like this:
```
/path/to/libtorch
                | -- bin
                | -- include
                | -- lib
                | ...
```
2. Set your environment variables:
```
export LIBTORCH=/path/to/torch
export LD_LIBRARY_PATH=/path/to/libtorch/lib
```

## CUDA support

This crate tries to support CUDA backed tensor operations.
It might be necessary to add following code in `build.rs` script in your crate:
```rust
fn main() {
    let os = std::env::var("CARGO_CFG_TARGET_OS").expect("Unable to get TARGET_OS");
    match os.as_str() {
        "linux" | "windows" => {
            if let Some(lib_path) = std::env::var_os("DEP_TCH_LIBTORCH_LIB") {
                println!("cargo:rustc-link-arg=-Wl,-rpath={}", lib_path.to_string_lossy());
            }
            println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
            println!("cargo:rustc-link-arg=-Wl,--copy-dt-needed-entries");
            println!("cargo:rustc-link-arg=-ltorch");
        }
        _ => {}
    }
}
```
It solved CUDA backend problem for me [link to source](https://github.com/LaurentMazare/tch-rs/issues/923#issuecomment-2669687652).

## Examples
Examples are presented in separate crate - [`amfiteatr_examples`](https://github.com/moroviintaas/amfiteatr_examples)
## Licence: MIT

