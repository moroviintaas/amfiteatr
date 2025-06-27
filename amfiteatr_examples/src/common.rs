use clap::ValueEnum;

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum ComputeDevice{
    Cpu,
    Cuda,
}

#[derive(ValueEnum, Copy, Clone, Debug)]
pub enum PolicySelect{
    A2C,
    MaskingA2C,
    PPO,
    MaskingPPO
}