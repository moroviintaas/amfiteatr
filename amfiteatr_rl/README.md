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


## Examples
Examples are presented in separate crate - [`amfiteatr_examples`](https://github.com/moroviintaas/amfiteatr_examples)
## Licence: MIT

