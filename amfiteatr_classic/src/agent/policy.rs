use std::fmt::{Debug, Formatter};
use amfiteatr_core::agent::Policy;
use amfiteatr_core::error::AmfiteatrError;
use crate::agent::LocalHistoryInfoSet;
use crate::domain::ClassicAction::{Down, Up};
use crate::domain::{ClassicAction, ClassicGameDomain, UsizeAgentId};


/// Uses last action that enemy used two times in the row, if no such action is found start with
/// [`Down`].
pub struct SwitchAfterTwo{
}

impl<ID: UsizeAgentId> Policy<ClassicGameDomain<ID>> for SwitchAfterTwo{
    type InfoSetType = LocalHistoryInfoSet<ID>;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {

        if let Some(last_report) = state.previous_encounters().last(){
            let mut other_action = last_report.other_player_action;
            for i in (0..state.previous_encounters().len()).rev(){
                if state.previous_encounters()[i].other_player_action == other_action{
                    return Ok(other_action)
                } else {
                    other_action = state.previous_encounters()[i].other_player_action;
                }
            }
            Ok(Down)
        } else {
            Ok(Down)
        }
    }
}
/// Example strategy that prefers action [`Down`], but punishes action [`Up`] with response of [`Up`].
/// Policy forgives previous [`Up`]s after to subsequent [`Down`]s from enemy.
pub struct ForgiveAfterTwo{
}


impl<ID: UsizeAgentId> Policy<ClassicGameDomain<ID>> for ForgiveAfterTwo{
    type InfoSetType = LocalHistoryInfoSet<ID>;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {

        if let Some(_last_report) = state.previous_encounters().last(){
            let mut subsequent_coops = 0;
            for i in (0..state.previous_encounters().len()).rev(){
                if state.previous_encounters()[i].other_player_action == Up {
                    return Ok(Up)
                } else {
                    subsequent_coops += 1;
                    if subsequent_coops >= 2{
                        return Ok(Down)
                    }
                }
            }
            Ok(Down)
        } else {
            Ok(Down)
        }
    }
}

/// Example strategy that prefers action [`Down`], but punishes certain actions of [`Up`] with action
/// of [`Up`]. To forgive and start using [`Down`] once more, other player must use action [`Down`]
/// at lest _Fib(x)_ where _Fib_ is Fibonacci function and _x_ is the number of [`Down`].
/// It is approach to make further forgiveness more difficult to achieve.
#[derive(Clone, Default)]
pub struct FibonacciForgiveStrategy{
}

impl FibonacciForgiveStrategy{

}

impl Debug for FibonacciForgiveStrategy{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "FibonacciForgiveStrategy", )
    }
}

fn fibonacci(index: u64) -> u64{

    match index{
        0 => 0,
        1 => 1,
        n => {
            let mut f1 = 0;
            let mut f2 = 1;
            let mut f = 1;
            for _i in 2..=n{
                f = f1 + f2;
                f1 = f2;
                f2 = f
            }
            f
        }
    }

}


impl<ID: UsizeAgentId> Policy<ClassicGameDomain<ID>> for FibonacciForgiveStrategy{
    type InfoSetType = LocalHistoryInfoSet<ID>;

    fn select_action(&self, state: &Self::InfoSetType) -> Result<ClassicAction, AmfiteatrError<ClassicGameDomain<ID>>> {

        let enemy_defects = state.count_actions_other(Up) as u64;
        let enemy_coops = state.count_actions_other(Down) as u64;

        let penalty = fibonacci(enemy_defects);
        if penalty > enemy_coops{
            Ok(Up)
        } else {
            Ok(Down)
        }


        /*
        let mut subsequent_coops = 1;
        //let mut enemy_defects = 0;
        let mut f1 = 0;
        let mut f2 = 1;
        let mut forgive_target = 1;
        for a in state.previous_encounters(){
            if a.other_player_action == Defect{
                forgive_target = f2 + f1;
                f1 = f2;
                f2 = forgive_target;
                subsequent_coops = 0;

            } else{
                subsequent_coops +=1;
            }
        }
        if subsequent_coops >= forgive_target{
            Some(Cooperate)
        } else {
            Some(Defect)
        }

         */
        //let mut punishing = false;

    }
}

#[cfg(test)]
mod tests{
    use crate::agent::policy::fibonacci;

    #[test]
    fn t_fibonacci(){
        assert_eq!(fibonacci(10), 55)
    }
}