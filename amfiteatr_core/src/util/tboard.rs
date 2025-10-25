use std::path::Path;
use std::sync::{Arc, Mutex, RwLock};
use crate::scheme::Scheme;
use crate::error::AmfiteatrError;

pub trait TensorboardSupport<S: Scheme>{

    fn add_tboard_directory(&mut self, directory_path: &Path) -> Result<(), AmfiteatrError<S>>;
    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>>;
}

impl<S: Scheme, T: TensorboardSupport<S>> TensorboardSupport<S> for Arc<Mutex<T>>{
    fn add_tboard_directory(&mut self, directory_path: &Path) -> Result<(), AmfiteatrError<S>> {
        match self.as_ref().lock(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>> {

        match self.as_ref().lock(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}

impl<S: Scheme, T: TensorboardSupport<S>> TensorboardSupport<S> for Mutex<T>{
    fn add_tboard_directory(&mut self, directory_path: &Path) -> Result<(), AmfiteatrError<S>> {
        match self.lock(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>> {

        match self.lock(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}

impl<S: Scheme, T: TensorboardSupport<S>> TensorboardSupport<S> for RwLock<T>{
    fn add_tboard_directory(&mut self, directory_path: &Path) -> Result<(), AmfiteatrError<S>> {
        match self.write(){
            Ok(mut guard) => guard.add_tboard_directory(directory_path),
            Err(e) => Err(AmfiteatrError::Lock { description: "Add tboard directory".to_string(), object: format!("{e}") })
        }
    }

    fn t_write_scalar(&mut self, index: i64, tag: &str, value: f32) -> Result<bool, AmfiteatrError<S>> {

        match self.write(){
            Ok(mut guard) => guard.t_write_scalar(index, tag, value),
            Err(e) => Err(AmfiteatrError::Lock { description: "Write scalar".to_string(), object: format!("{e}") })
        }
    }
}