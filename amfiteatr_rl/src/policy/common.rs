use tch::Tensor;

/// Distribution entropy for categorical distribution.
/// Given the probabilities and log probabilities tensors in shape:
/// `BATCH_SIZE x CATEGORY_NUMBER` outputs a [`Tensor`] of size `BATCH_SIZE`.
#[inline]
pub fn categorical_dist_entropy(probabilities: &Tensor, log_probabilities: &Tensor,  kind: tch::Kind) -> Tensor {
    (-log_probabilities * probabilities).sum_dim_intlist(-1, false, kind)
}

pub mod serde_kind{
    use std::fmt::Formatter;
    use serde::de;
    use serde::de::{Error, Visitor};
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
            Kind::Float8e4m3fnuz => serializer.serialize_str("float_8e4m3fnuz"),
        }
    }

    pub struct KindVisitor;

    impl <'de> Visitor<'de> for KindVisitor{
        type Value = Kind;

        fn expecting(&self, formatter: &mut Formatter) -> std::fmt::Result {
            formatter.write_str("u8, i8, i16, int, i64, half, f32, f64, complex-half, \
            complex-f32, complex-f64, bool, qi8, qu8, qi32, bf16, float_8e5m2, \
            float_8e4m3fn, float_8e5m2fnuz, float_8e4m3fnuz")
        }

        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E> where E: serde::de::Error {
            Ok(match v{
                "u8" => Kind::Uint8,
                "i8" => Kind::Int8,
                "i16" => Kind::Int16,
                "int" => Kind::Int,
                "i64" => Kind::Int64,
                "half" => Kind::Half,
                "f32" => Kind::Float,
                "f64" => Kind::Double,
                "complex-half" => Kind::ComplexHalf,
                "complex-f32" => Kind::ComplexFloat,
                "complex-f64" => Kind::ComplexDouble,
                "bool" => Kind::Bool,
                "qi8" => Kind::QInt8,
                "qu8" => Kind::QUInt8,
                "qi32" => Kind::QInt32,
                "bf16" => Kind::BFloat16,
                "float_8e5m2" => Kind::Float8e5m2,
                "float_8e4m3fn" => Kind::Float8e4m3fn,
                "float_8e5m2fnuz" => Kind::Float8e5m2fnuz,
                "float_8e4m3fnuz" => Kind::Float8e4m3fnuz,
                bad => {
                    return Err(de::Error::unknown_variant(bad, &["u8", "i8", "i16",
                        "int", "i64", "half", "f32", "f64", "complex-half",
                        "complex-f32", "complex-f64", "bool", "qi8", "qu8", "qi32", "bf16",
                        "float_8e5m2", "float_8e4m3fn", "float_8e5m2fnuz", "float_8e4m3fnuz"]))
                }
            })
        }

        fn visit_string<E>(self, v: String) -> Result<Self::Value, E> where E: Error {
            self.visit_str(&v[..])
        }
    }



    pub fn deserialize<'de, D>(deserializer: D) -> Result<tch::kind::Kind, D::Error>
        where
            D: serde::Deserializer<'de>,

    {

        deserializer.deserialize_str(KindVisitor)

    }

    pub fn default_kind() -> Kind{
        Kind::Float
    }
}


