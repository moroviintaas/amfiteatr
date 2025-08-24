use serde::{Serialize, Deserialize};
use amfiteatr_core::domain::Reward;
use enum_map::{Enum, enum_map, EnumMap};
use amfiteatr_core::error::ConvertError;
use amfiteatr_rl::tch::Tensor;
use amfiteatr_rl::tensor_data::{ContextDecodeIndexI64, ContextDecodeTensor, ContextEncodeIndexI64, TensorDecoding, TensorIndexI64Encoding};
use crate::domain::{ClassicAction, IntReward};

/// Enum for representing on which side of encounter is player.
/// This is important for [`AsymmetricRewardTable`]
#[derive(Debug, Copy, Clone, Enum, Serialize, Deserialize, speedy::Writable, speedy::Readable)]
pub enum Side{
    Left,
    Right
}

impl Default for Side{
    fn default() -> Self {
        Self::Left
    }
}

/// This is reward table for games where it is not important on what side the player is.
/// > The reward table would look like this:
/// ```norust
///  --------------------
/// |      |  Up  | Down |
/// |--------------------
/// |  Up  |    A |    B |
/// |      | A    | C    |
/// |--------------------|
/// | Down |    C |    D |
/// |      | B    | D    |
///  --------------------
/// ```
/// Note that you only need 4 numbers to define this table (compare to [`AsymmetricRewardTable`]).
#[derive(Debug, Clone, Serialize, Deserialize, Copy)]
pub struct SymmetricRewardTable<R: Reward + Copy> {

    pub map: EnumMap<ClassicAction, EnumMap<ClassicAction, R>>

    //pub coop_when_coop: R,
    //pub coop_when_defect: R,
    //pub defect_when_coop: R,
    //pub defect_when_defect: R
}
/// Alias for [`SymmetricRewardTable`] using `i64`
pub type SymmetricRewardTableInt = SymmetricRewardTable<IntReward>;


impl<R: Reward + Copy> SymmetricRewardTable<R> {

    pub fn new(coop_when_coop: R, coop_when_defect: R, defect_when_coop: R, defect_when_defect: R) -> Self{
        Self{
            map: enum_map! {
                ClassicAction::Up => enum_map! {
                    ClassicAction::Up => defect_when_defect,
                    ClassicAction::Down => defect_when_coop,
                },
                ClassicAction::Down => enum_map! {
                    ClassicAction::Up => coop_when_defect,
                    ClassicAction::Down => coop_when_coop,
                }
            }
        }
    }

    pub fn reward(&self, action: ClassicAction, other_action: ClassicAction) -> R {
        /*
        match (action, other_action){
            (ClassicAction::Cooperate, ClassicAction::Cooperate) => &self.coop_when_coop,
            (ClassicAction::Cooperate, ClassicAction::Defect) => &self.coop_when_defect,
            (ClassicAction::Defect, ClassicAction::Cooperate) => &self.defect_when_coop,
            (ClassicAction::Defect, ClassicAction::Defect) => &self.defect_when_defect
        }

         */
        self.map[action][other_action]
    }

}

/// This is reward table for games where it is important on what side the player is.
/// > May be used for games that has asymmetric tables - if two players switched certain action it
/// > may not necessarily need to reverted payoffs.
///
/// The reward table would look like this:
/// ```norust
///  --------------------
/// |      |  Up  | Down |
/// |--------------------
/// |  Up  |    A |    C |
/// |      | B    | D    |
/// |--------------------|
/// | Down |    E |    F |
/// |      | G    | H    |
///  --------------------
/// ```
/// Note that `E` can differ from `D`, like `C` and `G`.
///
#[derive(Debug, Copy, Clone, Serialize, Deserialize)]
pub struct AsymmetricRewardTable<R: Reward + Copy>{

    table: EnumMap<Side, SymmetricRewardTable<R>>
}

/// Alias for [`AsymmetricRewardTable`] using `i64`
pub type AsymmetricRewardTableInt = AsymmetricRewardTable<IntReward>;

impl<R: Reward + Copy> AsymmetricRewardTable<R> {

    pub fn new(left_table: SymmetricRewardTable<R>, right_table: SymmetricRewardTable<R>) -> Self{
        Self{
            table: enum_map! {
                Side::Left => left_table,
                Side::Right => right_table
            }
        }
    }

    pub fn reward_for_side(&self, reward_for: Side, left_action: ClassicAction, right_action: ClassicAction) -> R {

        self.table[reward_for].reward(left_action, right_action)
    }

    pub fn rewards(&self, left_action: ClassicAction, right_action: ClassicAction) -> (R, R){
        (
            self.table[Side::Left].reward(left_action, right_action),
            self.table[Side::Right].reward(left_action, right_action)
        )
    }



}

impl<R: Reward + Copy> From<SymmetricRewardTable<R>> for AsymmetricRewardTable<R>{
    fn from(value: SymmetricRewardTable<R>) -> Self {
        let mut reverted = value;
        reverted.map[ClassicAction::Down][ClassicAction::Up] =
            value.map[ClassicAction::Up][ClassicAction::Down];
        reverted.map[ClassicAction::Up][ClassicAction::Down] =
            value.map[ClassicAction::Down][ClassicAction::Up];

        AsymmetricRewardTable::new(value, reverted)
    }
}



#[cfg(test)]
mod tests{
    use std::mem::size_of;
    use crate::{AsymmetricRewardTableInt, SymmetricRewardTableInt};

    #[test]
    fn size_of_symmetric_table(){
        assert_eq!(size_of::<SymmetricRewardTableInt>(), 32);
    }
    #[test]
    fn size_of_asymmetric_table(){
        assert_eq!(size_of::<AsymmetricRewardTableInt>(), 64);
    }
}


pub struct ClassicActionTensorRepresentation{}

impl TensorDecoding for ClassicActionTensorRepresentation{
    fn expected_input_shape(&self) -> &[i64] {
        &[1]
    }
}

impl TensorIndexI64Encoding for ClassicActionTensorRepresentation{
    fn min(&self) -> i64 {
        0
    }

    fn limit(&self) -> i64 {
        1
    }
}

impl ContextDecodeTensor<ClassicActionTensorRepresentation> for ClassicAction{
    fn try_from_tensor(tensor: &Tensor, _decoding: &ClassicActionTensorRepresentation) -> Result<Self, ConvertError>
    where
        Self: Sized
    {
        let id = tensor.f_int64_value(&[0])
            .map_err(|e| ConvertError::ConvertFromTensor {
                origin: format!("{e:}"),
                context: format!("Converting Connect Four action from Tensor {tensor:?}")}
            )?;
        match id{
            0 => Ok(ClassicAction::Up),
            1 => Ok(ClassicAction::Down),
            n => Err(ConvertError::IllegalValue { value: format!("{n}"), context: "ContextDecodeTensor".to_string() })

        }
    }
}

impl ContextDecodeIndexI64<ClassicActionTensorRepresentation> for ClassicAction{
    fn try_from_index(index: i64, _encoding: &ClassicActionTensorRepresentation) -> Result<Self, ConvertError> {
        match index{
            0 => Ok(ClassicAction::Up),
            1 => Ok(ClassicAction::Down),
            n => Err(ConvertError::IllegalValue { value: format!("{n}"), context: "ContextDecodeIndexI64".to_string() })

        }
    }
}

impl ContextEncodeIndexI64<ClassicActionTensorRepresentation> for ClassicAction{
    fn try_to_index(&self, _encoding: &ClassicActionTensorRepresentation) -> Result<i64, ConvertError> {
        match self{
            ClassicAction::Up => Ok(0),
            ClassicAction::Down => Ok(1),
        }
    }
}