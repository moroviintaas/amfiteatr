use crate::policy::DiscountFactor;


/// Structure for basic training configuration data.
/// When you implement policy you may use some special training configuration and tunning.
/// This however provide minimum used in policies implemented in this crate.
#[derive(Copy, Clone)]
pub struct TrainConfig{
    pub gamma: f64
}

impl DiscountFactor for TrainConfig{
    fn discount_factor(&self) -> f64 {
        self.gamma
    }
}