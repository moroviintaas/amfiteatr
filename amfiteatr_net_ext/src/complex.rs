

use speedy::{Writable, Readable, LittleEndian};
use amfiteatr_core::comm::{BidirectionalEndpoint, StdEndpoint};
use std::{fmt::Debug, sync::mpsc::{RecvError, SendError, TryRecvError}};
use amfiteatr_core::domain::{
    AgentMessage, EnvironmentMessage
};
use amfiteatr_core::error::CommunicationError;

use crate::tcp::{
    PairedTcpEndpoint,
    TcpCommError
};

pub enum ComplexComm<OT, IT, E: std::error::Error, const  SIZE: usize>{
    StdSync(StdEndpoint<OT, IT, E>),
    Tcp(PairedTcpEndpoint<OT, IT, E, SIZE>)

}


impl <'a, OT, IT, E: std::error::Error, const  SIZE: usize> BidirectionalEndpoint for ComplexComm<OT, IT, E, SIZE>
where OT: Writable<LittleEndian> + Debug, IT: Readable<'a, LittleEndian> + Debug,
E: std::error::Error + From<TcpCommError> + From<RecvError> + From<SendError<OT>> + From<TryRecvError> + From<SendError<IT>>{
    type OutwardType = OT;

    type InwardType = IT;

    type Error = E;

    fn send(&mut self, message: Self::OutwardType) -> Result<(), Self::Error> {
        match self{
            ComplexComm::StdSync(comm) => comm.send(message),
            ComplexComm::Tcp(comm) => comm.send(message),
        }
    }

    fn receive_blocking(&mut self) -> Result<Self::InwardType, Self::Error> {
        match self{
            ComplexComm::StdSync(comm) => comm.receive_blocking(),
            ComplexComm::Tcp(comm) => comm.receive_blocking(),
        }
    }

    fn receive_non_blocking(&mut self) -> Result<Option<Self::InwardType>, Self::Error> {
        match self{
            ComplexComm::StdSync(comm) => comm.receive_non_blocking(),
            ComplexComm::Tcp(comm) => comm.receive_non_blocking(),
        }
    }
}

pub type ComplexComm2048<OT, IT, E> = ComplexComm<OT, IT, E, 2048>;
pub type ComplexComm1024<OT, IT, E> = ComplexComm<OT, IT, E, 1024>;
pub type ComplexComm128<OT, IT, E> = ComplexComm<OT, IT, E, 128>;
pub type ComplexComm256<OT, IT, E> = ComplexComm<OT, IT, E, 256>;
pub type ComplexComm512<OT, IT, E> = ComplexComm<OT, IT, E, 512>;

pub type DomainCommE512<DP> = ComplexComm<EnvironmentMessage<DP>, AgentMessage<DP>, CommunicationError<DP>, 512>;
pub type DomainCommA512<DP> = ComplexComm<AgentMessage<DP>, EnvironmentMessage<DP>,  CommunicationError<DP>, 512>;
