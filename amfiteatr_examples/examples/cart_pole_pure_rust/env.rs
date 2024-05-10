
//const GRAVITY: f64 = 9.8;
//const MASS_CART: f64 = 1.0;
//const MASSPOLE: f64

use std::fmt::{Debug, Formatter};
use amfiteatr_core::domain::DomainParameters;
use amfiteatr_core::env::SequentialGameState;
use crate::common::{CartPoleAction, CartPoleDomain, CartPoleObservation, CartPoleRustError, SINGLE_PLAYER_ID};
#[derive(Debug, Clone)]
enum KinematicsIntegrator{
    Euler,
    SemiImplicitEuler
}
#[derive(Debug, Clone)]
pub struct CartPoleEnvStateRust {
    sutton_barto_reward: bool,

    gravity: f32,
    mass_cart: f32,
    mass_pole: f32,
    length: f32,
    pole_mass_length: f32,
    force_mag: f32,
    tau: f32,
    kinematics_integrator: KinematicsIntegrator,

    theta_threshold_radians: f32,
    x_threshold: f32,

    is_open: bool,
    state: Option<CartPoleObservation>,
    terminated: bool,
    max_episode_steps: usize,
    steps_made: usize,


}


impl CartPoleEnvStateRust {
    #[inline]
    fn total_mass(&self) -> f32{
        self.mass_cart + self.mass_pole
    }
    pub fn new() -> Self{
        let x_threshold = 2.4f32;
        let theta_threshold_radians = 12.0 * 2.0 * std::f64::consts::PI / 360.0;

        let high = CartPoleObservation{
            position: x_threshold as f32* 2.0,
            velocity: f32::MAX,
            angle: (theta_threshold_radians  * 2.0) as f32,
            angular_velocity: f32::MAX,
        };
        let low = CartPoleObservation{
            position: - high.position,
            velocity: - high.velocity,
            angle: - high.angle,
            angular_velocity: - high.angular_velocity,
        };

        Self{
            sutton_barto_reward: false,
            gravity: 9.8,
            mass_cart: 0.0,
            mass_pole: 0.0,
            length: 0.0,
            pole_mass_length: 0.0,
            force_mag: 0.0,
            tau: 0.0,
            kinematics_integrator: KinematicsIntegrator::Euler,
            theta_threshold_radians: 0.0,
            x_threshold,
            is_open: false,
            state: None,
            terminated: false,
            max_episode_steps: 500,
            steps_made: 0,
        }

    }
}




impl SequentialGameState<CartPoleDomain> for CartPoleEnvStateRust{
    type Updates = [(<CartPoleDomain as DomainParameters>::AgentId, <CartPoleDomain as DomainParameters>::UpdateType );1];

    fn current_player(&self) -> Option<<CartPoleDomain as DomainParameters>::AgentId> {
        if self.terminated || self.steps_made > self.max_episode_steps{
            None
        } else {
            Some(SINGLE_PLAYER_ID)
        }
    }

    fn is_finished(&self) -> bool {
        self.terminated || self.steps_made > self.max_episode_steps
    }

    fn forward(&mut self, _agent: <CartPoleDomain as DomainParameters>::AgentId, action: <CartPoleDomain as DomainParameters>::ActionType) -> Result<Self::Updates, <CartPoleDomain as DomainParameters>::GameErrorType> {

        let s = self.state.as_ref().ok_or(CartPoleRustError::GameStateNotInitialized)?;
        //let CartPoleObservation{position: x, velocity: x_dot, angle: theta, angular_velocity: theta_dot} = s;
        let (mut x, mut x_dot, mut theta, mut theta_dot) = (s.position, s.velocity, s.angle, s.angular_velocity);
        let force = match action {
            CartPoleAction::Left => -self.force_mag,
            CartPoleAction::Right => self.force_mag,
        };
        let costheta = f32::cos(theta);
        let sintheta = f32::sin(theta);

        let temp = ( force + (self.pole_mass_length * theta_dot * theta_dot * sintheta))/self.total_mass();




        todo!()
    }
}