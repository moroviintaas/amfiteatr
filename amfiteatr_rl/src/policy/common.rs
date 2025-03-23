use tch::Tensor;

/// Distribution entropy for categorical distribution.
/// Given the probabilities and log probabilities tensors in shape:
/// `BATCH_SIZE x CATEGORY_NUMBER` outputs a [`Tensor`] of size `BATCH_SIZE`.
#[inline]
pub fn categorical_dist_entropy(probabilities: &Tensor, log_probabilities: &Tensor,  kind: tch::Kind) -> Tensor {
    (-log_probabilities * probabilities).sum_dim_intlist(-1, false, kind)
}