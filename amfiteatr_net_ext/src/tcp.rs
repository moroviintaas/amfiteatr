use std::collections::HashMap;
use std::error::Error;
use std::fmt::Debug;
use std::io::{Read, Write};
use std::marker::PhantomData;
use std::net::TcpStream;
use std::sync::mpsc;
use std::thread;
use log::{error};
use speedy::{LittleEndian, Readable, Writable};
use amfiteatr_core::error::CommunicationError;
use thiserror::Error;

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
pub struct PairedTcpEndpoint<OT, IT, E: Error, const  SIZE: usize>{
    stream: std::net::TcpStream,
    _ot: PhantomData<OT>,
    _it: PhantomData<IT>,
    _e: PhantomData<E>,
    _buff: PhantomData<[u8; SIZE]>
}

impl<OT, IT, E: Error, const  SIZE: usize> PairedTcpEndpoint<OT, IT, E, SIZE>{
    pub fn new(stream : std::net::TcpStream) -> Self{
        Self{stream, _ot: PhantomData, _it: PhantomData, _e: PhantomData, _buff: PhantomData}
    }


}
pub type PairedTcpEnvironmentEndpoint<DP, const SIZE: usize> = PairedTcpEndpoint<EnvironmentMessage<DP>, AgentMessage<DP>, CommunicationError<DP>, SIZE>;
pub type PairedTcpAgentEndpoint<DP, const SIZE: usize> = PairedTcpEndpoint<AgentMessage<DP>, EnvironmentMessage<DP>, CommunicationError<DP>, SIZE>;

pub type MappedEnvironmentTcpEndpoints<DP, const SIZE: usize> = HashMap<<DP as DomainParameters>::AgentId, PairedTcpEnvironmentEndpoint<DP, SIZE>>;
pub type MappedAgentTcpEndpoints<DP, const SIZE: usize> = HashMap<<DP as DomainParameters>::AgentId, PairedTcpAgentEndpoint<DP, SIZE>>;

impl<DP: DomainParameters, const SIZE: usize> PairedTcpEnvironmentEndpoint<DP, SIZE>{


    pub fn create_local_net<'a>(port: u16, ids: impl Iterator<Item=&'a DP::AgentId>) -> Result<(MappedEnvironmentTcpEndpoints<DP, SIZE>, MappedAgentTcpEndpoints<DP, SIZE>), CommunicationError<DP>>{


        let mut results_environment_connections = HashMap::new();
        let mut results_agent_connections = HashMap::new();


        let tcp_listener = std::net::TcpListener::bind(format!("127.0.0.1:{}", port))
            .map_err(|e| CommunicationError::ConnectionInitialization {
                description: format!("Failed binding tcp listener to 127.0.0.1:{port}: {e:}")})?;

        let ids_set_env: Vec<DP::AgentId> = ids.cloned().collect();
        let ids_set_agents = ids_set_env.clone();

        thread::scope(|s|{
            s.spawn(||{
                for id in ids_set_env{
                    match tcp_listener.accept(){
                        Ok((stream, _addr)) => {
                            results_environment_connections.insert(id.clone(), Ok(stream));
                            //tx_env.send(Ok(id)).unwrap();
                        }
                        Err(e) => {

                            results_environment_connections.insert(id.clone(), Err(CommunicationError::<DP>::ConnectionInitialization {
                                description: format!("Failed accepting connection: {e:}")
                            }));
                        }
                    }
                }
            });
            s.spawn(||{
                for id in ids_set_agents{
                    match TcpStream::connect(format!("127.0.0.1:{port}")){
                        Ok(s) => {
                            results_agent_connections.insert(id, Ok(s));
                        }
                        Err(e) => {
                            results_agent_connections.insert(id.clone(), Err(CommunicationError::<DP>::ConnectionInitialization {
                                description: format!("Failed accepting connection: {e:}")
                            }));
                        }
                    }
                }
            });

        });

        let r_env_connections: Result<MappedEnvironmentTcpEndpoints<DP, SIZE>, CommunicationError<DP>>
            = results_environment_connections.into_iter().map(|(id, r)|{
                r.map(|stream| (id, PairedTcpEnvironmentEndpoint::new(stream)))

        }).collect();

        let r_agents_connections: Result<MappedAgentTcpEndpoints<DP, SIZE>, CommunicationError<DP>>
            = results_agent_connections.into_iter().map(|(id, r)| {
            r.map(|stream| (id, PairedTcpAgentEndpoint::new(stream)))
        }).collect();

        r_env_connections.and_then(|env|{
            r_agents_connections.map(|agent| (env, agent))
        })


    }

    pub fn create_local_pair(port: u16) -> Result<(PairedTcpEnvironmentEndpoint<DP, SIZE>, PairedTcpAgentEndpoint<DP, SIZE>), CommunicationError<DP>>{
        let (tx_env, rx_env) = mpsc::channel();
        let (tx_agent, rx_agent) = mpsc::channel();

        //let mut hold_env = None;
        //let mut hold_agent = None;
        let tcp_listener = std::net::TcpListener::bind(format!("127.0.0.1:{}", port))
            .unwrap_or_else(|_| panic!("Failed listening on local port {port:}"));
        let handle_env = thread::spawn(move ||{


            let (stream, _) = tcp_listener.accept()
                .expect("Failed creating stream on responder side");
            tx_env.send(stream)
                .expect("Failed sending responder socket");


        });


        let handle_agent = thread::spawn(move ||{
            let agent_connector = std::net::TcpStream::connect(format!("127.0.0.1:{}", port))
                .expect("Failed connecting from client");
            tx_agent.send(agent_connector)
                .expect("Failed sending initiator socket");

        });

            let env_endpoint = rx_env.recv().expect("Failed receiving env stream");
            let agent_endpoint = rx_agent.recv().expect("Failed receiving agent stream");
        handle_env.join().unwrap();
        handle_agent.join().unwrap();

        Ok((PairedTcpEndpoint::new(env_endpoint), PairedTcpEndpoint::new(agent_endpoint)))
        //Ok((TcpEndpoint::new(hold_env.unwrap()), TcpEndpoint::new(hold_agent.unwrap())))
    }
}


impl<'a, OT, IT, E: Error, const  SIZE: usize> BidirectionalEndpoint for PairedTcpEndpoint<OT, IT, E, SIZE>
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

pub type TcpCommK1<OT, IT, E> = PairedTcpEndpoint<OT, IT, E, 1024>;
pub type TcpCommK2<OT, IT, E> = PairedTcpEndpoint<OT, IT, E, 2048>;
pub type TcpComm512<OT, IT, E> = PairedTcpEndpoint<OT, IT, E, 512>;

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
