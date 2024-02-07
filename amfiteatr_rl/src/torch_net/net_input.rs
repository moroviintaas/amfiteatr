use tch::Tensor;


pub trait NetInput{
    fn tensor(self) -> Tensor;
}

impl NetInput for Tensor{
    fn tensor(self) -> Tensor{
        self
    }
}

pub struct TensorStateAction{
    pub state: Tensor,
    pub action: Tensor
}

impl NetInput for TensorStateAction{
    fn tensor(self) -> Tensor {
        Tensor::cat(&[&self.state, &self.action], 0)
    }
}