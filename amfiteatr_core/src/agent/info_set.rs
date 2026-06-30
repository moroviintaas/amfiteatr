use std::fmt::Debug;
use crate::scheme::{Scheme, Reward};

/// Represents agent's point of view on game state.
/// > Formally _information set_ is subset of game _states_ that are indistinguishable
/// > from the point of view of this agent.
/// > Most common case is when agent does not know some detail of the game state.
/// > Game states with identical observable details but differing in unobservable (not known) detail
/// > form single _information set_.
pub trait InformationSet<S: Scheme>: Send + Debug{


    fn agent_id(&self) -> &S::AgentId;
    fn is_action_valid(&self, action: &S::ActionType) -> bool;
    fn update(&mut self, update: S::UpdateType) -> Result<(), S::GameErrorType>;
}
/// Trait for information sets that can provide iterator (Vec or some other kind of list) of
/// actions that are possible in this state of game.
pub trait PresentPossibleActions<S: Scheme>: InformationSet<S>{
    /// Structure that can be transformed into iterator of actions.
    type ActionIteratorType: IntoIterator<Item = S::ActionType>;
    /// Construct and return iterator of possible actions.
    fn available_actions(&self) -> Self::ActionIteratorType;
}

impl<S: Scheme, T: InformationSet<S>> InformationSet<S> for Box<T>{
    fn agent_id(&self) -> &S::AgentId {
        self.as_ref().agent_id()
    }


    fn is_action_valid(&self, action: &S::ActionType) -> bool {
        self.as_ref().is_action_valid(action)
    }

    fn update(&mut self, update: S::UpdateType) -> Result<(), S::GameErrorType> {
        self.as_mut().update(update)
    }
}

impl<S: Scheme, T: PresentPossibleActions<S>> PresentPossibleActions<S> for Box<T>{
    type ActionIteratorType = T::ActionIteratorType;

    fn available_actions(&self) -> Self::ActionIteratorType {
        self.as_ref().available_actions()
    }
}

/// Information Set that can produce score based on it's state.
/// This reward can be in different type that defined in [`DomainParameters`](Scheme).
/// > It can represent different kind of reward than defined in protocol parameters.
/// > Primary use case is to allow agent interpret its situation, for example instead of
/// > one numeric value as reward agent may be interested in some vector of numeric values representing
/// > his multi-criterion view on game's result.
pub trait EvaluatedInformationSet<S: Scheme, R: Reward>: InformationSet<S>{
    fn current_assessment(&self) -> R;
    fn penalty_for_illegal(&self) -> R;
}

impl<T: EvaluatedInformationSet<S, R>, S: Scheme, R: Reward> EvaluatedInformationSet<S, R> for Box<T> {

    fn current_assessment(&self) -> R {
        self.as_ref().current_assessment()
    }

    fn penalty_for_illegal(&self) -> R {
        self.as_ref().penalty_for_illegal()
    }
}


/// Information set that can be constructed using certain (generic type) value to construct new
/// information set instance.
/// > This is meant to be implemented for every information set
/// > used in game by any agent and for state of environment.
/// > Implementing construction based on common seed allows to reload all info sets and states.
pub trait ConstructedInfoSet<S: Scheme, B>: InformationSet<S> + From<B> {}
impl<S: Scheme, B, T: InformationSet<S> + From<B>> ConstructedInfoSet<S, B> for T{}

//impl<S: DomainParameters, B, T: ConstructedInfoSet<S, B>> ConstructedInfoSet<S, B> for Box<T>{}


#[cfg(feature = "mcp")]
mod mcp{
    use std::collections::HashMap;
    use std::sync::Arc;
    use rmcp::{ErrorData, RoleServer};
    use crate::agent::InformationSet;
    use crate::scheme::{Renew, Scheme};
    use serde::{Serialize, Deserialize};

    use tokio::sync::Mutex;
    use schemars::JsonSchema;
    use std::default::Default;
    use std::marker::PhantomData;
    use rmcp::handler::server::wrapper::Parameters;
    use rmcp::model::{CallToolResult, Content, GetPromptResult, PromptMessage, PromptMessageRole};
    use rmcp::service::RequestContext;
    use crate::util::mcp::McpReqArgAgentId;



    #[derive(Clone)]
    pub struct McpCoreInformationSets<
        SC: Scheme,
        IS: InformationSet<SC> + Serialize + for<'a> Deserialize<'a> + JsonSchema + Renew<SC, Seed>,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema
    >
    where
        SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
    {
        internal: std::sync::Arc<Mutex<HashMap<SC::AgentId, IS>>>,
        game_name: String,
        usage: String,
        _seed: PhantomData<Seed>,

    }

    impl<
        SC: Scheme,
        IS: InformationSet<SC> + Serialize + for<'a> Deserialize<'a> + JsonSchema + Renew<SC, Seed>,
        Seed: Serialize + for<'a> Deserialize<'a> + JsonSchema + Clone
    > McpCoreInformationSets<SC, IS, Seed>
    where
        SC::ActionType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::UpdateType: Serialize + for<'a> Deserialize<'a> + JsonSchema,
        SC::AgentId: Serialize + for<'a> Deserialize<'a> + JsonSchema,
    {
        pub fn new(info_set_map: HashMap<SC::AgentId, IS>, game_name: String, usage: String) -> Self{
            Self{game_name, usage, internal: Arc::new(Mutex::new(info_set_map)), _seed: PhantomData::default()}
        }

        pub async fn reset_all_information_sets(&self, seed: Seed) -> Result<(), ErrorData>{
            let mut hm = self.internal.lock().await;

            for is in hm.values_mut(){
                let _ = is.renew_from(seed.clone());
            }
            Ok(())
        }

        pub async fn update_information_set(&self, agent_id: SC::AgentId, updates: Vec<SC::UpdateType>)
        -> Result<(), ErrorData>{
            let mut hm = self.internal.lock().await;

            let mut is = hm.remove(&agent_id).ok_or_else(||ErrorData::internal_error(format!("No information set for player {agent_id}"), None))?;
            for update in updates.into_iter(){
                is.update(update).map_err(|e|{
                    ErrorData::internal_error(format!("Error updating information set: {e}"), None)
                })?;
            }
            hm.insert(agent_id, is);

            Ok(())

        }

        pub async fn serialize_information_set(&self, agent_id: &SC::AgentId)
            -> Result<CallToolResult, ErrorData>{

            let hm = self.internal.lock().await;

            match hm.get(&agent_id){
                None => Err(ErrorData::internal_error(format!("No information set for player {agent_id}"), None)),
                Some(is) => {
                    Ok(CallToolResult::success(vec![Content::json(is)?]))
                }
            }

        }

        pub fn get_usage(&self) -> &str{
            &self.usage[..]
        }

        pub fn game_name(&self) -> &str{
            &self.game_name[..]
        }

        pub async fn information_set_usage(
            &self,
            _ctx: RequestContext<RoleServer>,
        ) -> Result<GetPromptResult, ErrorData> {

            let messages = vec![
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "What is the information set server used for?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "It is used to store and manage information sets for players in some game or simulation.\
                    The information could be described as specific player's view on game state - it stores some\
                    but not necessarily whole information stored in game or simulation state."
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "How is information set kept up to date?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "When game state changes, environment typically issue \"observations\" (updates) to involved players.\
                    These observations should be applied using tool `update_information_set`. \
                    Server can track more than one information set at once, so tool `update_information_set` expects\
                    `agent_id` to identify to which player these observations apply. Observations are applied in the form of list \
                    (it can be list of one element). Information set processes these observation in exact order they appear on list."
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "What can I do with the information set?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Typically it is used as knowledge needed (or at least helpful) to select next action in the game.\
                    You can make it manually by analysing the data or you can use some policy (for example in form of MCP tool).\
                    Policy should accept information set and hint with action proposition. Selecting action is out of scope of information set,\
                    it's policies' role. Search for tools like `select_action` or similar."
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "How do I extract data of information set from this store?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "There is a tool `serialize_information_set` which returns serialized information set.\
                    You can use it then as you wish."
                ),
            ];

            Ok(GetPromptResult::new(messages).with_description("Interaction with players \"information set\"."))

        }

        pub async fn player_automation(
            &self,
            _ctx: RequestContext<RoleServer>,
        ) -> Result<GetPromptResult, ErrorData> {
            let messages = vec![
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "I want to run semi-interactive game using MCP servers. \
                    I want to some players to be fully automated and some interactive (that I control).\
                    How can I use information set MCP server to do it?".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Ok, information set MCP server can be a part of this set up. \
                    Firstly, initialize or restart information set information for every player that is \
                    going to be tracked in this store - both fully automatic players and those you control. \
                    Typically you can do this with tool `reset_all_information_sets`. \
                    Then you can start the game. Information sets stored in the tool will be cyclically updated and used to select next action.\
                    Game is normally run by environment that accepts players' actions and provides observations for players. \
                    When game runs (changes its state) it produces observations for players, usually you have to prompt game environment for \
                    observations. Every observation you can collect you should apply to tracked information sets.\
                    You can do it by calling `update_information_set` tool. Updating information set typically does \
                    not need user information, so MCP client should should be calling this tool automatically, without prompting user.\
                    When you know which player has to make a move in specific time, you need to make sure it has actual information set for this player. \
                    Once you are sure the information set is up to date, select an action for the player. \
                    To do so get information set serialization with tool `serialize_information_set`, then \
                    If the player is interactive decide manually what to do. If the player is automatic \
                    action selection should be made by some external program (for example MCP server dedicated for policy logic - with tool `select_action`). \
                    Typically you need to specify what tool should be used for which player.\
                    "
                ),
                PromptMessage::new_text(
                    PromptMessageRole::User,
                    "Can you make example game round for generic game with 3 players A0, A1 and A2?\
                    Player A0 is interactive, I want to play as him. A1 should run automatically using policy P1 and\
                    A2 should run automatically on policy P2.".to_string()
                ),
                PromptMessage::new_text(
                    PromptMessageRole::Assistant,
                    "Sure. Assume game environment is one MCP server, P1 and P2 are executed on separate MCP servers, \
                    it does not matter but P1 could be DQN based policy and P2 PPO. All information sets are stored in one MCP server dedicated to information sets.

                    1. Call tool `reset_all_information_sets` to reset to starting point all information sets. (Can be done automatically).
                    2. Call tool `reset` on game environment (Not on this server) (Can be done automatically).
                    3. Find current player by calling tool `get_current_player` from environment MCP server. If it does not return any player the game is finished.
                    4. For current player (let's say it's A0 for this example), collect his observations with environment tool `get_updates` called with A0 as argument.
                    5. Update information set for A0 player - call tool `update_information_set` with A0 as argument and collected observations.
                    6. Now you have to analyse situation for A0 and select action, you should call tool `serialize_information_set` with A0 argument.
                    7. Once action is selected pass it to the environment using it's tool `process_action`.
                    8. Now ask which player plays now (like in step 3).
                    9. Let's say now plays A1, collect his observations as in 4. and update his information set as in 5.
                    10. Now since player A1 is automatic you do not need to analyse his information set, you need to serialize it and pass to policy P1. \
                    You collect serialization as in 6. and pass it to policy MCP server with tool `select_action`.
                    11. Pass selected action to environment, similarly like in 7. but you can set up this to be done automatically without your interaction.
                    12. Now ask which player plays now (like in step 3 and 8).
                    13. Let's say it is A2 - perform similarly as in steps 9-11.
                    14. Continue to the moment the game environment returns no current player, that means end of the game.

                    Note: you can ask environment for updates for player at any time, and apply them as the appear, but make sure to do this just before his action so he has the most recent info.
                    "
                ),
            ];
            Ok(GetPromptResult::new(messages).with_description("Role of information set in player's automation."))
        }


    }


}

#[cfg(feature = "mcp")]
pub use mcp::*;