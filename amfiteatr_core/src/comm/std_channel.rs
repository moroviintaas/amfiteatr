use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::mpsc::{channel, Receiver, RecvError, Sender, SendError, TryRecvError};
use crate::env::ListPlayers;
use crate::comm::endpoint::BidirectionalEndpoint;
use crate::error::CommunicationError;
use crate::scheme::{AgentMessage, EnvironmentMessage, Scheme};

use super::{AgentAdapter, EnvironmentAdapter, EnvironmentEndpoint, BroadcastingEnvironmentAdapter};


#[derive(Debug)]


/// This is standard endpoint dedicated for local agents.
/// It uses Rust's built-in [`mpsc`](std::sync::mpsc) communication channel.
pub struct StdEndpoint<OT, IT, E: Error>{
    sender: Sender<OT>,
    receiver: Receiver<IT>,
    _phantom: PhantomData<E>
}

/// Standard endpoint to be used by environment to connect with local agents.
pub type StdEnvironmentEndpoint<S> = StdEndpoint<EnvironmentMessage<S>, AgentMessage<S>, CommunicationError<S>>;
/// Standard endpoint to be used by local agent to connect with environment
pub type StdAgentEndpoint<S> = StdEndpoint<AgentMessage<S>, EnvironmentMessage<S>,  CommunicationError<S>>;

impl<OT, IT, E: Error> StdEndpoint<OT, IT, E>
where StdEndpoint<OT, IT, E> :  BidirectionalEndpoint<OutwardType = OT, InwardType = IT, Error = E>{
    pub fn new(sender: Sender<OT>, receiver: Receiver<IT>) -> Self{
        Self{sender, receiver, _phantom: PhantomData}
    }
    pub fn new_pair() -> (Self, StdEndpoint<IT, OT, E>) {
        let (tx_1, rx_1) = channel();
        let (tx_2, rx_2) = channel();

        (Self{sender: tx_1, receiver: rx_2, _phantom: PhantomData},
         StdEndpoint {sender: tx_2, receiver: rx_1, _phantom: PhantomData})
    }
    pub fn _decompose(self) -> (Sender<OT>, Receiver<IT>){
        (self.sender, self.receiver)
    }
}

impl<OT, IT, E> BidirectionalEndpoint for StdEndpoint<OT, IT, E>
where E: Debug + Error + From<RecvError> + From<SendError<OT>> + From<TryRecvError> + From<SendError<IT>>,
OT: Debug, IT:Debug{
    type OutwardType = OT;

    type InwardType = IT;

    type Error = E;

    fn send(&mut self, message: OT) -> Result<(), E> {
        self.sender.send(message).map_err(|e| e.into())
    }

    fn receive_blocking(&mut self) -> Result<IT, E> {
        self.receiver.recv().map_err(|e| e.into())
    }

    fn receive_non_blocking(&mut self) -> Result<Option<IT>, E> {
        self.receiver.try_recv().map_or_else(
            |e| match e{
                TryRecvError::Empty => Ok(None),
                TryRecvError::Disconnected => Err(e.into())
            },
            |message| Ok(Some(message))
        )

    }


}

/// Dynamic endpoint, actually enum with static variant of using [`StdEndpoint`](crate::comm::StdEndpoint),
/// and [`Box`](std::boxed::Box) with [`BidirectionalEndpoint`](crate::comm::BidirectionalEndpoint)
pub enum DynEndpoint<OT, IT, E: Error>{
    Std(StdEndpoint<OT, IT, E>),
    Dynamic(Box<dyn BidirectionalEndpoint<OutwardType = OT, InwardType = IT, Error = E> + Send>)
}

impl <OT: Debug, IT: Debug, E: Error> BidirectionalEndpoint for DynEndpoint<OT, IT, E>
where E: From<RecvError> + From<SendError<OT>> + From<TryRecvError> + From<SendError<IT>>{
    type OutwardType = OT;
    type InwardType = IT;
    type Error = E;

    fn send(&mut self, message: Self::OutwardType) -> Result<(), Self::Error> {
        match self{
            DynEndpoint::Std(c) => c.send(message),
            DynEndpoint::Dynamic(c) => {c.as_mut().send(message)}
        }
    }

    fn receive_blocking(&mut self) -> Result<Self::InwardType, Self::Error> {
        match self{
            DynEndpoint::Std(c) => c.receive_blocking(),
            DynEndpoint::Dynamic(c) => {c.as_mut().receive_blocking()}
        }
    }

    fn receive_non_blocking(&mut self) -> Result<Option<Self::InwardType>, Self::Error> {
        match self{
            DynEndpoint::Std(c) => c.receive_non_blocking(),
            DynEndpoint::Dynamic(c) => {c.as_mut().receive_non_blocking()}
        }
    }
}

/// Standard agent adapter using [`mpsc`](std::sync::mpsc) to be paired with
/// [`EnvironmentMpscPort`](crate::comm::EnvironmentMpscPort)
pub struct AgentMpscAdapter<S: Scheme>{
    id: S::AgentId,
    sender: Sender<(S::AgentId, AgentMessage<S>)>,
    receiver: Receiver<EnvironmentMessage<S>>,
}

impl<S: Scheme> AgentMpscAdapter<S>{
    pub(crate) fn new(
        id: S::AgentId,
        sender: Sender<(S::AgentId, AgentMessage<S>)>,
        receiver: Receiver<EnvironmentMessage<S>>
    ) -> Self{
        Self{id, sender, receiver}
    }
}

impl<S: Scheme> AgentAdapter<S> for AgentMpscAdapter<S>{
    fn send(&mut self, message: AgentMessage<S>) -> Result<(), CommunicationError<S>> {
        self.sender.send((self.id.to_owned(), message)).map_err(|e| e.into())
    }

    fn receive(&mut self) -> Result<EnvironmentMessage<S>, CommunicationError<S>> {
        self.receiver.recv().map_err(|e| e.into())
    }
}

impl<S: Scheme> BidirectionalEndpoint for AgentMpscAdapter<S> {
    type OutwardType = AgentMessage<S>;
    type InwardType = EnvironmentMessage<S>;
    type Error = CommunicationError<S>;

    fn send(&mut self, message: Self::OutwardType) -> Result<(), Self::Error> {
        self.sender.send((self.id.clone(), message)).map_err(|e|e.into())
    }

    fn receive_blocking(&mut self) -> Result<Self::InwardType, Self::Error> {
        self.receiver.recv().map_err(|e|e.into())
    }

    fn receive_non_blocking(&mut self) -> Result<Option<Self::InwardType>, Self::Error> {
        match self.receiver.try_recv(){
            Ok(message) => Ok(Some(message)),
            Err(TryRecvError::Empty) => Ok(None),
            Err(e) => Err(e.into())
        }
    }
}

/// Implementation of [`EnvironmentAdapter`](crate::comm::EnvironmentAdapter) based on Rust's
/// standard [`mpsc`](std::sync::mpsc).
pub struct EnvironmentMpscPort<S: Scheme>{
    sender_template: Sender<(S::AgentId, AgentMessage<S>)>,
    receiver: Receiver<(S::AgentId, AgentMessage<S>)>,
    senders: HashMap<S::AgentId, Sender<EnvironmentMessage<S>>>
}

impl<S: Scheme> Default for EnvironmentMpscPort<S>{
    fn default() -> Self {
        Self::new()
    }
}
impl<S: Scheme> EnvironmentMpscPort<S>{
    pub fn new() -> Self{
        let (sender_template, receiver) = channel();
        Self{receiver, sender_template, senders: HashMap::new()}
    }
    pub fn register_agent(&mut self, id: S::AgentId) -> Result<AgentMpscAdapter<S>, CommunicationError<S>>{
        /*
        if self.senders.contains_key(&id){
            return Err(CommunicationError::DuplicatedAgent(id));
        } else {
            let (env_tx, agent_rx) = channel();
            let agent_adapter = AgentMpscAdapter::new(
                id.clone(),
                self.sender_template.clone(),
                agent_rx,
            );
            self.senders.insert(id, env_tx);
            Ok(agent_adapter)

        }*/
        if let std::collections::hash_map::Entry::Vacant(e) = self.senders.entry(id.clone()) {
            let (env_tx, agent_rx) = channel();
            let agent_adapter = AgentMpscAdapter::new(
                    id.clone(),
                    self.sender_template.clone(),
                    agent_rx,
                );
            e.insert(env_tx);
            Ok(agent_adapter)

            } else {
            Err(CommunicationError::DuplicatedAgent(id))
        }

    }
}

impl<S: Scheme> EnvironmentAdapter<S> for EnvironmentMpscPort<S>{
    fn send(&mut self, agent: &<S as Scheme>::AgentId, message: EnvironmentMessage<S>)
            -> Result<(), CommunicationError<S>> {
        let s = self.senders.get(agent)
            .ok_or_else(|| CommunicationError::ConnectionToAgentNotFound(agent.to_owned()))?;
        s.send(message).map_err(|e| e.into())
    }

    fn receive_blocking(&mut self) -> Result<(<S as Scheme>::AgentId, AgentMessage<S>), CommunicationError<S>> {
        self.receiver.recv().map_err(|e| e.into())
    }

    fn receive_non_blocking(&mut self) -> Result<Option<(<S as Scheme>::AgentId, AgentMessage<S>)>, CommunicationError<S>> {
        //self.receiver.try_recv().map_err(|e| e.into())
        self.receiver.try_recv().map_or_else(
            |e| match e{
                TryRecvError::Empty => Ok(None),
                TryRecvError::Disconnected => Err(e.into())
            },
            |(id, message)| Ok(Some((id, message)))
        )
    }
    fn is_agent_connected(&self, agent_id: &S::AgentId) -> bool{
        self.senders.contains_key(agent_id)
    }
}

impl<S: Scheme> BroadcastingEnvironmentAdapter<S> for EnvironmentMpscPort<S>{
    fn send_all(&mut self, message: EnvironmentMessage<S>) ->  Result<(), CommunicationError<S>> {
        let mut result = Ok(());
        for (_agent, tx) in self.senders.iter_mut(){
            #[cfg(feature = "log_trace")]
            log::trace!("While broadcasting. Sending to {_agent} message: {message:?}");
            let r = tx.send(message.clone());
            if let Err(e) = r{
                #[cfg(feature = "log_warn")]
                log::warn!("While broadcasting. Error sending to {_agent}: {e}");
                if result.is_ok(){
                    result = Err(CommunicationError::from(e));
                }
            }
               
        }
        #[cfg(feature = "log_trace")]
        log::trace!("Ending broadcast sending.");
        result
    }
}

impl<S: Scheme> ListPlayers<S> for EnvironmentMpscPort<S>{
    type IterType = <Vec<S::AgentId> as IntoIterator>::IntoIter;

    fn players(&self) -> Self::IterType {
        self.senders.keys().map(|r| r.to_owned())
        .collect::<Vec<S::AgentId>>().into_iter()
    }
}

//impl 

/// This is environment communication adapter using HashMap of
/// [`BidirectionalEndpoint`s](crate::comm::BidirectionalEndpoint) and
/// round robin listening strategy.
pub struct EnvRRAdapter<S: Scheme, T: EnvironmentEndpoint<S>>{
    endpoints: HashMap<S::AgentId, T>,
}
impl <S: Scheme, T: EnvironmentEndpoint<S>> Default for EnvRRAdapter<S, T>{
    fn default() -> Self {
        Self::new()
    }
}
impl <S: Scheme, T: EnvironmentEndpoint<S>> EnvRRAdapter<S, T>{

    pub fn new() -> Self{
        Self { endpoints: Default::default() }
    }

    /// Adds new agent, inserts communication endpoint to HashMap.
    /// Returns error if there is already enpdoint dedicated to this agent.
    pub fn register_agent(&mut self, agent_id: S::AgentId, comm: T) -> Result<(), CommunicationError<S>>{
        /*
        if self.endpoints.contains_key(&agent_id){
            return Err(CommunicationError::DuplicatedAgent(agent_id));
        } else {
            self.endpoints.insert(agent_id, comm);
            Ok(())
        }

         */
        if let std::collections::hash_map::Entry::Vacant(e) = self.endpoints.entry(agent_id.clone()) {
            e.insert(comm);
            Ok(())
        } else {
            Err(CommunicationError::DuplicatedAgent(agent_id))
        }
    }
}

impl<S: Scheme> EnvRRAdapter<S, StdEnvironmentEndpoint<S>>{

    /// Creates local connection for agent given his id.
    /// Creates endpoint pair, environment side is stored in HashMap
    /// and endpoint of agent is returned.
    pub fn create_local_connection(&mut self, agent_id: S::AgentId) -> Result<StdAgentEndpoint<S>, CommunicationError<S>>{
        /*
        if self.endpoints.contains_key(&agent_id){
            return Err(CommunicationError::DuplicatedAgent(agent_id));
        } else {
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::<S>::new_pair();
            //let (tx_e, rx_a) = channel();
            //let (tx_a, rx_e) = channel();
            self.endpoints.insert(agent_id, env_comm);
            Ok(agent_comm)
        }

         */
          if let std::collections::hash_map::Entry::Vacant(e) = self.endpoints.entry(agent_id.clone()) {
              let (env_comm, agent_comm) = StdEnvironmentEndpoint::<S>::new_pair();
              e.insert(env_comm);
              Ok(agent_comm)
          } else {
              Err(CommunicationError::DuplicatedAgent(agent_id))
          }
    }
}


impl<S: Scheme> EnvRRAdapter<S, Box<dyn EnvironmentEndpoint<S>>>{

    pub fn create_local_connection(&mut self, agent_id: S::AgentId) -> Result<StdAgentEndpoint<S>, CommunicationError<S>>{
        /*
        if self.endpoints.contains_key(&agent_id){
            return Err(CommunicationError::DuplicatedAgent(agent_id));
        } else {
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::<S>::new_pair();
            self.endpoints.insert(agent_id, Box::new(env_comm));
            Ok(agent_comm)
        }

         */
        if let std::collections::hash_map::Entry::Vacant(e) = self.endpoints.entry(agent_id.clone()) {
            let (env_comm, agent_comm) = StdEnvironmentEndpoint::<S>::new_pair();
            e.insert(Box::new(env_comm));
            Ok(agent_comm)
        } else {
            Err(CommunicationError::DuplicatedAgent(agent_id))
        }
    }
}


impl <S: Scheme, T: EnvironmentEndpoint<S>> EnvironmentAdapter<S> for EnvRRAdapter<S, T>{
    fn send(&mut self, agent: &<S as Scheme>::AgentId, message: EnvironmentMessage<S>)
            -> Result<(), CommunicationError<S>> {
        if let Some(s) = self.endpoints.get_mut(agent){
            s.send(message)
        } else {
            Err(CommunicationError::ConnectionToAgentNotFound(agent.to_owned()))
        }
        
    }

    fn receive_blocking(&mut self) -> Result<(<S as Scheme>::AgentId, AgentMessage<S>), CommunicationError<S>> {
        loop{
            for (agent, endpoint) in self.endpoints.iter_mut(){
                match endpoint.receive_non_blocking(){
                    Ok(None) => {},
                    Ok(Some(message)) => {
                        return Ok((agent.to_owned(), message));
                    },
                    Err(e) => {
                        return Err(e);
                    }
                }
            }
        }
    }

    fn receive_non_blocking(&mut self) -> Result<Option<(<S as Scheme>::AgentId, AgentMessage<S>)>, CommunicationError<S>> {
        for (agent, endpoint) in self.endpoints.iter_mut(){
            match endpoint.receive_non_blocking(){
                Ok(None) => {},
                Ok(Some(message)) => {
                    return Ok(Some((agent.to_owned(), message)))
                },
                Err(e) => {
                    return Err(e)
                }
            }
        }
        Ok(None)

    }

    fn is_agent_connected(&self, agent_id: &<S as Scheme>::AgentId) -> bool {
        self.endpoints.contains_key(agent_id)
    }
}

impl <S: Scheme, T: EnvironmentEndpoint<S>> BroadcastingEnvironmentAdapter<S> for EnvRRAdapter<S, T>{
    fn send_all(&mut self, message: EnvironmentMessage<S>) ->  Result<(), CommunicationError<S>> {
        let mut result = Ok(());
        for (_agent, endpoint) in self.endpoints.iter_mut(){
            let r = endpoint.send(message.clone());
            if let Err(e) = r{
                if result.is_ok(){
                    result = Err(e);
                }
            }
               
        }
        result
    }
}