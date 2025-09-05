use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

pub trait TensorboardSupport<DP: Scheme>{

    fn add_tboard_directory<P: AsRef<std::path::Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>>;
    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>>;
}

impl<DP: Scheme, T: TensorboardSupport<DP>> TensorboardSupport<DP> for Arc<Mutex<T>>{
    fn add_tboard_directory<P: AsRef<Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        match self.as_ref().lock(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>> {

        match self.as_ref().lock(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}

impl<DP: Scheme, T: TensorboardSupport<DP>> TensorboardSupport<DP> for Mutex<T>{
    fn add_tboard_directory<P: AsRef<Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        match self.lock(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>> {

        match self.lock(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}

impl<DP: Scheme, T: TensorboardSupport<DP>> TensorboardSupport<DP> for RwLock<T>{
    fn add_tboard_directory<P: AsRef<Path>>(&mut self, directory_path: P) -> Result<(), AmfiteatrError<DP>> {
        match self.write(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<DP>> {

        match self.write(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}