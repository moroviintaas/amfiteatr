use std::error::Error;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::sync::mpsc;
use std::thread;
use log::{debug, error, trace};
use speedy::{LittleEndian, Readable, Writable};
use amfiteatr_core::error::CommunicationError;
use thiserror::Error;
use zeroize::Zeroize;
use amfiteatr_core::comm::BidirectionalEndpoint;
use amfiteatr_core::domain::{AgentMessage, DomainParameters, EnvironmentMessage};

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

#[derive(Debug)]
pub struct BoundedTcpEndpoint<OT, IT, E: Error, const  SIZE: usize>{
    stream: std::net::TcpStream,
    _ot: PhantomData<OT>,
    _it: PhantomData<IT>,
    _e: PhantomData<E>,
    _buff: PhantomData<[u8; SIZE]>
}

impl<OT, IT, E: Error, const  SIZE: usize> BoundedTcpEndpoint<OT, IT, E, SIZE>{
    pub fn new(stream : std::net::TcpStream) -> Self{
        Self{stream, _ot: PhantomData::default(), _it: PhantomData::default(), _e: PhantomData::default(), _buff: PhantomData::default()}
    }


}
pub type BoundedTcpEnvironmentEndpoint<DP, const SIZE: usize> = BoundedTcpEndpoint<EnvironmentMessage<DP>, AgentMessage<DP>, CommunicationError<DP>, SIZE>;
pub type BoundedTcpAgentEndpoint<DP, const SIZE: usize> = BoundedTcpEndpoint<AgentMessage<DP>, EnvironmentMessage<DP>, CommunicationError<DP>, SIZE>;

impl<DP: DomainParameters, const SIZE: usize> BoundedTcpEnvironmentEndpoint<DP, SIZE>{



    /// TODO proper error handling
    pub fn create_local_pair(port: u16) -> Result<(BoundedTcpEnvironmentEndpoint<DP, SIZE>, BoundedTcpAgentEndpoint<DP, SIZE>), CommunicationError<DP>>{
        let (tx_env, rx_env) = mpsc::channel();
        let (tx_agent, rx_agent) = mpsc::channel();

        //let mut hold_env = None;
        //let mut hold_agent = None;
        let tcp_listener = std::net::TcpListener::bind(&format!("127.0.0.1:{}", port))
            .expect(&format!("Failed listening on local port {port:}"));
        let handle_env = thread::spawn(move ||{


            let (stream, _) = tcp_listener.accept()
                .expect("Failed creating stream on responder side");
            tx_env.send(stream)
                .expect("Failed sending responder socket");


        });


        let handle_agent = thread::spawn(move ||{
            let agent_connector = std::net::TcpStream::connect(&format!("127.0.0.1:{}", port))
                .expect("Failed connecting from client");
            tx_agent.send(agent_connector)
                .expect("Failed sending initiator socket");

        });

            let env_endpoint = rx_env.recv().expect("Failed receiving env stream");
            let agent_endpoint = rx_agent.recv().expect("Failed receiving agent stream");
        handle_env.join().unwrap();
        handle_agent.join().unwrap();

        Ok((BoundedTcpEndpoint::new(env_endpoint), BoundedTcpEndpoint::new(agent_endpoint)))
        //Ok((TcpEndpoint::new(hold_env.unwrap()), TcpEndpoint::new(hold_agent.unwrap())))
    }
}


impl<'a, OT, IT, E: Error, const  SIZE: usize> BidirectionalEndpoint for BoundedTcpEndpoint<OT, IT, E, SIZE>
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

pub type TcpCommK1<OT, IT, E> = BoundedTcpEndpoint<OT, IT, E, 1024>;
pub type TcpCommK2<OT, IT, E> = BoundedTcpEndpoint<OT, IT, E, 2048>;
pub type TcpComm512<OT, IT, E> = BoundedTcpEndpoint<OT, IT, E, 512>;

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
/*
#[derive(Debug)]
pub struct TcpEndpoint<OT, IT, E: Error>{
    stream: std::net::TcpStream,
    _ot: PhantomData<OT>,
    _it: PhantomData<IT>,
    _e: PhantomData<E>,
    buffer: Vec<u8>,
}

impl<OT, IT, E: Error,> TcpEndpoint<OT, IT, E>{
    pub fn new(stream : std::net::TcpStream) -> Self{
        Self{
            stream,
            _ot: PhantomData::default(),
            _it: PhantomData::default(),
            _e: PhantomData::default(),
            buffer: vec![0;2048],
        }
    }
}

impl<'a, OT, IT, E: Error> BidirectionalEndpoint for TcpEndpoint<OT, IT, E>
    where OT: Writable<LittleEndian> + Debug, IT: Readable<'a, LittleEndian> + Debug,
          E: From<TcpCommError>{
    type OutwardType = OT;

    type InwardType = IT;

    type Error = E;


    fn send(&mut self, message: OT) -> Result<(), E> {

        trace!("Sending message via TCP");
        match message.write_to_stream(&mut self.stream){
            Ok(_) => Ok(()),
            Err(e) => Err(TcpCommError::SendError(format!("{e:}")).into())
        }
    }
    fn receive_blocking(&mut self) -> Result<IT, E> {
        trace!("Receiving blocking on tcp");
        self.stream.set_nonblocking(false).unwrap();


        IT::read_from_stream_buffered(&mut self.stream)
            .map_err(|e| TcpCommError::RecvError(format!("{e:}")).into())
    }
    fn receive_non_blocking(&mut self) -> Result<Option<IT>, E> {

        self.stream.set_nonblocking(true).unwrap();
        //self.buffer.clear();
        self.buffer.fill(0);
        //self.buffer.zeroize();

        match self.stream.read(&mut self.buffer){
            Ok(0) => {
                //trace!("Receiving non-blocking on tcp, no data");
                Ok(None)
            }
            Ok(_n) => {
                //trace!("Receiving non-blocking on tcp, data of len: {}", _n);

                match IT::read_from_buffer_copying_data(&self.buffer){
                    Ok(m) => Ok({
                        self.buffer.fill(0);

                        Some(m)
                    }),
                    Err(e) => Err(TcpCommError::DeserializeError(format!("{e:}")).into())
                }
            },
            Err(_) => Err(TcpCommError::TryRecvDisconnectedError.into())
        }




    }
}

pub type TcpEnvironmentEndpoint<DP> = TcpEndpoint<EnvironmentMessage<DP>, AgentMessage<DP>, CommunicationError<DP>>;
pub type TcpAgentEndpoint<DP> = TcpEndpoint<AgentMessage<DP>, EnvironmentMessage<DP>, CommunicationError<DP>>;

impl<DP: DomainParameters> TcpEnvironmentEndpoint<DP>{


    /// TODO proper error handling
    pub fn create_local_pairs(number_of_connections: usize, port: u16) -> Result<(Vec<(TcpEnvironmentEndpoint<DP>, TcpAgentEndpoint<DP>)>), CommunicationError<DP>>{
        let (tx_env, rx_env) = mpsc::channel();
        let (tx_agent, rx_agent) = mpsc::channel();

        //let mut hold_env = None;
        //let mut hold_agent = None;

        let handle_env = thread::spawn(move ||{
            let tcp_listener = std::net::TcpListener::bind(&format!("127.0.0.1:{}", port))
                .expect(&format!("Failed listening on local port {port:}"));
            for _ in 0..number_of_connections{
                let (stream, _) = tcp_listener.accept()
                    .expect("Failed creating stream on responder side");
                tx_env.send(stream)
                    .expect("Failed sending responder socket");
            }

        });


        let handle_agent = thread::spawn(move ||{
            for _ in 0..number_of_connections{
                let agent_connector = std::net::TcpStream::connect(&format!("127.0.0.1:{}", port))
                    .expect("Failed connecting from client");
                tx_agent.send(agent_connector)
                    .expect("Failed sending initiator socket");
            }

        });

        let mut connections = Vec::with_capacity(number_of_connections);
        for _ in 0..number_of_connections{
            let env_endpoint = rx_env.recv().expect("Failed receiving env stream");
            let agent_endpoint = rx_agent.recv().expect("Failed receiving agent stream");
            connections.push((TcpEndpoint::new(env_endpoint), TcpEndpoint::new(agent_endpoint)));
        }
        handle_env.join().unwrap();
        handle_agent.join().unwrap();

        Ok(connections)
        //Ok((TcpEndpoint::new(hold_env.unwrap()), TcpEndpoint::new(hold_agent.unwrap())))
    }
}

 */
