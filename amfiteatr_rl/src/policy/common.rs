use tch::Tensor;

/// Distribution entropy for categorical distribution.
/// Given the probabilities and log probabilities tensors in shape:
/// `BATCH_SIZE x CATEGORY_NUMBER` outputs a [`Tensor`] of size `BATCH_SIZE`.
#[inline]
pub fn categorical_dist_entropy(probabilities: &Tensor, log_probabilities: &Tensor,  kind: tch::Kind) -> Tensor {
    (-log_probabilities * probabilities).sum_dim_intlist(-1, false, kind)
}

pub mod serde_kind{
    use tch::Kind;

    pub fn serialize<S> (value: &tch::kind::Kind, serializer: S) -> Result<S::Ok, S::Error>
        where
            S: serde::Serializer,
    {
        match value{
            Kind::Uint8 => serializer.serialize_str("u8"),
            Kind::Int8 => serializer.serialize_str("i8"),
            Kind::Int16 => serializer.serialize_str("i16"),
            Kind::Int => serializer.serialize_str("int"),
            Kind::Int64 => serializer.serialize_str("i64"),
            Kind::Half => serializer.serialize_str("half"),
            Kind::Float => serializer.serialize_str("f32"),
            Kind::Double => serializer.serialize_str("f64"),
            Kind::ComplexHalf => serializer.serialize_str("complex-half"),
            Kind::ComplexFloat => serializer.serialize_str("complex-f32"),
            Kind::ComplexDouble => serializer.serialize_str("complex-f64"),
            Kind::Bool => serializer.serialize_str("bool"),
            Kind::QInt8 => serializer.serialize_str("qi8"),
            Kind::QUInt8 => serializer.serialize_str("qu8"),
            Kind::QInt32 => serializer.serialize_str("qi32"),
            Kind::BFloat16 => serializer.serialize_str("bf16"),
            Kind::Float8e5m2 => serializer.serialize_str("float_8e5m2"),
            Kind::Float8e4m3fn => serializer.serialize_str("float_8e4mfn"),
            Kind::Float8e5m2fnuz => serializer.serialize_str("float_8e5m2fnuz"),
            Kind::Float8e4m3fnuz => serializer.serialize_str("float8e4m3fnuz"),
        }
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<tch::kind::Kind, D::Error>
        where
            D: serde::Deserializer<'de>,

    {

        todo!()
        // Twoja niestandardowa funkcja deserializacji
        // Zwraca zdeserializowaną wartość
    }
}