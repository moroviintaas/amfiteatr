use crate::AsymmetricRewardTableInt;
use crate::domain::{AgentNum, UsizeAgentId};

pub trait ReplInfoSet<ID: UsizeAgentId>{

    fn create(agent_id: ID, reward_table: AsymmetricRewardTableInt) -> Self;
}

pub trait ReplInfoSetAgentNum: ReplInfoSet<AgentNum>{}

impl<S: ReplInfoSet<AgentNum>> ReplInfoSetAgentNum for S {}