use tch::nn::{Path, Sequential};

pub struct SequentialBuilder<F: Fn(&Path) -> Sequential> {

    sequential_fn: F
}

impl<F: Fn(&Path) -> Sequential> SequentialBuilder<F>{

    pub fn new(f: F) -> Self{
        Self{sequential_fn: f}
    }
    pub fn build(&self, path: &Path) -> Sequential{
        (self.sequential_fn)(path)
    }
}