use std::error::Error;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::marker::PhantomData;
use speedy::{LittleEndian, Readable, Writable};
use amfiteatr_core::error::CommunicationError;
use thiserror::Error;
use amfiteatr_core::comm::BidirectionalEndpoint;
use amfiteatr_core::domain::DomainParameters;

//const BRIDGE_COMM_BUFFER_SIZE: usize = 256;
#[derive(Debug, Clone, Error)]
pub enum TcpCommError{
    #[error("Serialize Error, text: {0}")]
    SerializeError(String),
    #[error("Deserialize Error, text: {0}")]
    DeserializeError(String),
    #[error("Send Error, text: {0}")]
    SendError(String),
    #[error("Recv Error, text: {0}")]
    RecvError(String),
    #[error("TryRecv Error (empty)")]
    TryRecvEmptyError,
    #[error("TryRecv Error (disconnected)")]
    TryRecvDisconnectedError,
}
/*
impl From<TcpCommError> for CommError{
    fn from(value: TcpCommError) -> Self {
        match value{
            TcpCommError::SerializeError => Self::SerializeError,
            TcpCommError::DeserializeError => Self::DeserializeError,
            TcpCommError::SendError => Self::SendError,
            TcpCommError::RecvError => Self::RecvError,
            TcpCommError::TryRecvEmptyError => Self::TryRecvEmptyError,
            TcpCommError::TryRecvDisconnectedError => Self::TryRecvDisconnectedError
        }
    }
}
*/
pub struct TcpComm<OT, IT, E: Error, const  SIZE: usize>{
    stream: std::net::TcpStream,
    _ot: PhantomData<OT>,
    _it: PhantomData<IT>,
    _e: PhantomData<E>,
    _buff: PhantomData<[u8; SIZE]>
}

impl<OT, IT, E: Error, const  SIZE: usize> TcpComm<OT, IT, E, SIZE>{
    pub fn new(stream : std::net::TcpStream) -> Self{
        Self{stream, _ot: PhantomData::default(), _it: PhantomData::default(), _e: PhantomData::default(), _buff: PhantomData::default()}
    }
}

impl<'a, OT, IT, E: Error, const  SIZE: usize> BidirectionalEndpoint for TcpComm<OT, IT, E, SIZE>
where OT: Writable<LittleEndian> + Debug, IT: Readable<'a, LittleEndian> + Debug,
E: From<TcpCommError>{
    type OutwardType = OT;

    type InwardType = IT;

    type Error = E;


    fn send(&mut self, message: OT) -> Result<(), E> {

        let mut buffer = [0u8; SIZE];
        message.write_to_buffer(&mut buffer)
            .map_err(|e| TcpCommError::SerializeError(format!("{e:}")))?;
        /*
        if message.write_to_buffer(&mut buffer).is_err(){
            return Err(TcpCommError::SerializeError.into())
        }

         */
        match self.stream.write_all(&buffer){
            Ok(_) => Ok(()),
            Err(e) => Err(TcpCommError::SendError(format!("{e:}")).into()),
        }
    }
    fn receive_blocking(&mut self) -> Result<IT, E> {
        self.stream.set_nonblocking(false).unwrap();
        let mut buffer = [0u8; SIZE];
        let mut received = false;
        while !received {
            match self.stream.read(&mut buffer){
                Ok(0) => {},
                Ok(_) => {
                    received = true;
                },
                Err(e) => {return Err(TcpCommError::RecvError(format!("{e:}")).into())}
            }
        }
        match IT::read_from_buffer_copying_data(&buffer){
            Ok(m) => Ok(m),
            Err(e) => Err(TcpCommError::DeserializeError(format!("{e:}")).into())
        }
    }
    fn receive_non_blocking(&mut self) -> Result<Option<IT>, E> {
        let mut buffer = [0u8; SIZE];
        self.stream.set_nonblocking(true).unwrap();

        match self.stream.read(&mut buffer){
            Ok(0) => {
                //debug!("TryRecvError");
                //Err(TcpCommError::TryRecvEmptyError.into())
                Ok(None)
            }
            Ok(_n) => {
                //debug!("Tcp TryRecv with {} bytes", n);
                match IT::read_from_buffer_copying_data(&buffer){
                    Ok(m) => Ok(Some(m)),
                    Err(e) => Err(TcpCommError::DeserializeError(format!("{e:}")).into())
                }
            },
            Err(_e) => {

                Err(TcpCommError::TryRecvDisconnectedError.into())
            }
        }


    }
}

pub type TcpCommK1<OT, IT, E> = TcpComm<OT, IT, E, 1024>;
pub type TcpCommK2<OT, IT, E> = TcpComm<OT, IT, E, 2048>;
pub type TcpComm512<OT, IT, E> = TcpComm<OT, IT, E, 512>;

impl<Spec: DomainParameters> From<TcpCommError> for CommunicationError<Spec>{
    fn from(value: TcpCommError) -> Self {
        match value{
            TcpCommError::SerializeError(s) => CommunicationError::SerializeError(s),
            TcpCommError::DeserializeError(s) => CommunicationError::DeserializeError(s),
            TcpCommError::SendError(s) => CommunicationError::SendErrorUnspecified(s),
            TcpCommError::RecvError(s) => CommunicationError::RecvErrorUnspecified(s),
            TcpCommError::TryRecvEmptyError => CommunicationError::RecvEmptyBufferErrorUnspecified,
            TcpCommError::TryRecvDisconnectedError => CommunicationError::RecvPeerDisconnectedErrorUnspecified
        }
    }
}