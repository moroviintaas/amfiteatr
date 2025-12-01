# Amfiteatr
Framework to model game theory problems and apply reinforcement learning to optimise solution. 
It is designed to help model problems involving many players.
This is rather low level modelling framework and in many cases Python equivalent would be more handy however
this one maybe helpful if you want Rust compiler to help you develop liable code.
## Member crates:
1. [`amfiteatr_core`](https://crates.io/crates/amfiteatr_core) 
([github](https://github.com/moroviintaas/amfiteatr_core.git)) - crate
for core traits and generic implementations without reinforcement learning.
2. [`amfiteatr_rl`](https://crates.io/crates/amfiteatr_rl) 
([github](https://github.com/moroviintaas/amfiteatr_rl.git)) - crate extending core features
to provide interface and simple implementations of reinforcement learning (using neural networks backed by 
Torch ([`tch`](https://docs.rs/tch/latest/tch/))).
3. [`amfiteatr_net_ext`](https://crates.io/crates/amfiteatr_net_ext) 
([github](https://github.com/moroviintaas/amfiteatr_net_ext.git)) - currently providing early proof of concept
for using TCP socket to provide communications between entities in game model
4. [`amfiteatr_classic`](https://crates.io/crates/amfiteatr_classic) 
([github](https://github.com/moroviintaas/amfiteatr_classic.git)) - crate providing structures for simulating
classic game theory games (like prisoners' dilemma).
5. [`amfiteatr_examples`](https://github.com/moroviintaas/amfiteatr_examples.git) - repository with some examples
of using the library. Hopefully it will be expanded in the future.


## Linker issue
Since version `1.90` Rust by default uses linker `ldd`, which cannot be 
used in with `libtorch`. 
To use `libtorch` one needs to use `ld`, therefore in the 
crates there are `.cargo/config.toml` files to force usage of `ld`.
If you include these crates in projects, you may need to do  the same
on workspace level. Or on global level (`~/.cargo/config.toml`).


## Development stage
It is my education and research project. Many elements will change or vanish in the future and some breaking changes
may occur often. I will be adding features and documentation in time and I will try to simplify interfaces that currently seems to be
inconvenient. 

__TL;DR__ It's early and unstable stage.


## Licence: MIT

## Other projects:
Currently, I develop some projects using this library, that can show current possibilities
1. [`brydz_model`](https://github.com/moroviintaas/brydz_model) - Simulation and reinforcement learning model
   for contract bridge card game. Can be used as example of implementing 4 player game.
2. [`brydz_dd`](https://github.com/moroviintaas/brydz_model) - early project of Double Dummy solver for contract bridge
   (card faced up analysis of optimal game solution). __Warning__ it uses alpha-beta algorithm variants and on
   current level of optimisation it cannot be used to solve full 52 card problems.



