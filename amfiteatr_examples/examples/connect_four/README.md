# Game Connect Four

This is one of standard games using AEC API authored by Farama Foundation ([code](https://github.com/Farama-Foundation/PettingZoo)).


This example presents actually 3 implementations of models for connect four game:
1. Original Farama Foundation environment with agents implemented in Python ([here](./python/));
2. Farama Foundiation Environment wrapped as GameState in Rust code of `amfiteatr`, to show how such wrapping could be made ([here](./rust/env_wrapped.rs));
3. Game rewritten in Rust, trying to preserve original code structure by lines ([here](./rust/env.rs))