


use std::fmt::Debug;
use rand::Rng;
use amfiteatr_core::scheme::{Renew, Scheme};
use amfiteatr_core::env::{GameStateWithPayoffs, SequentialGameState};
use amfiteatr_core::error::AmfiteatrError;
use crate::cart_pole::common::{CartPoleAction, CartPoleScheme, CartPoleObservation, CartPoleRustError, SINGLE_PLAYER_ID};
#[derive(Debug, Clone)]
enum KinematicsIntegrator{
    #[warn(dead_code)]
    Euler,
    #[warn(dead_code)]
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

    //is_open: bool,
    state: Option<CartPoleObservation>,
    terminated: bool,
    max_episode_steps: usize,
    steps_made: usize,

    payoff: f32,
    steps_beyond_terminated: Option<usize>,


}


impl CartPoleEnvStateRust {
    #[inline]
    fn total_mass(&self) -> f32{
        self.mass_cart + self.mass_pole
    }
    pub fn new(sutton_barto_reward: bool) -> Self{
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

        let d = rand::distr::Uniform::new(-0.05, 0.05).unwrap();
        let mut rng = rand::rng();
        let state = CartPoleObservation{
            position: rng.sample(d),
            velocity: rng.sample(d),
            angle: rng.sample(d),
            angular_velocity: rng.sample(d),
        };

        let mass_pole = 0.1;
        let length = 0.5;
        Self{
            sutton_barto_reward,
            gravity: 9.8,
            mass_cart: 1.0,
            mass_pole,
            length,
            pole_mass_length: mass_pole * length,
            force_mag: 10.0,
            tau: 0.02,
            kinematics_integrator: KinematicsIntegrator::Euler,
            theta_threshold_radians: 12.0 * 2.0 * std::f32::consts::PI / 360.0,
            x_threshold,
            //is_open: false,
            state: Some(state),
            terminated: false,
            max_episode_steps: 500,
            steps_made: 0,
            payoff: 0.0,
            steps_beyond_terminated: None,
        }

    }
}




impl SequentialGameState<CartPoleScheme> for CartPoleEnvStateRust{
    type Updates = [(<CartPoleScheme as Scheme>::AgentId, <CartPoleScheme as Scheme>::UpdateType );1];

    fn current_player(&self) -> Option<<CartPoleScheme as Scheme>::AgentId> {
        if self.terminated || self.steps_made > self.max_episode_steps{
            None
        } else {
            Some(SINGLE_PLAYER_ID)
        }
    }

    fn is_finished(&self) -> bool {
        self.terminated || self.steps_made > self.max_episode_steps
    }

    fn forward(&mut self, _agent: <CartPoleScheme as Scheme>::AgentId, action: <CartPoleScheme as Scheme>::ActionType) -> Result<Self::Updates, <CartPoleScheme as Scheme>::GameErrorType> {

        log::trace!("In state: {:?} making step: {:?}", self.state, action);
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

        let thetaacc = (self.gravity * sintheta - (costheta * temp)) / (
            self.length * (4.0/3.0 - (self.mass_pole * costheta * costheta / self.total_mass()))
            );
        let xacc = temp - self.pole_mass_length * thetaacc * costheta / self.total_mass();

        match self.kinematics_integrator{
            KinematicsIntegrator::Euler => {
                x = x + (self.tau * x_dot);
                x_dot = x_dot + (self.tau * xacc);
                theta = theta + (self.tau * theta_dot);
                theta_dot = theta_dot + (self.tau * thetaacc);
            },
            KinematicsIntegrator::SemiImplicitEuler => {
                x_dot = x_dot + self.tau * xacc;
                x = x+(self.tau * x_dot);
                theta_dot = theta_dot + (self.tau * thetaacc);
                theta = theta + (self.tau * theta_dot);
            }


        }

        let s = self.state.as_mut().ok_or(CartPoleRustError::GameStateNotInitialized)?;
        s.position = x;
        s.velocity = x_dot;
        s.angle = theta;
        s.angular_velocity = theta_dot;

        log::trace!("Cart Pole observation: {:?}", self.state);

        self.terminated =
                x < -self.x_threshold
                || x > self.x_threshold
                || theta < -self.theta_threshold_radians
                || theta > self.theta_threshold_radians;



        let reward = match self.terminated{
            false => match self.sutton_barto_reward{
                true => 0.0,
                false => 1.0,
            },
            true => match self.sutton_barto_reward{
                true => -1.0,
                false => match &mut self.steps_beyond_terminated{
                    None => {
                        self.steps_beyond_terminated = Some(0);
                        1.0
                    },
                    Some(n) => {
                        *n+=1;
                        0.0
                    }
                }
            },

        };
        

        self.payoff += reward;

        Ok([(SINGLE_PLAYER_ID, CartPoleObservation{
            position: x,
            velocity: x_dot,
            angle: theta,
            angular_velocity: theta_dot,
        })])
    }

    fn first_observations(&self) -> Option<Self::Updates> {
        self.state.clone().and_then(|s| Some([(SINGLE_PLAYER_ID, s)]))
    }
}

impl Renew<CartPoleScheme, (), > for CartPoleEnvStateRust {
    fn renew_from(&mut self, _base: ()) -> Result<(), AmfiteatrError<CartPoleScheme>> {
        let d = rand::distr::Uniform::new(-0.05, 0.05).unwrap();
        let mut rng = rand::rng();
        let state = CartPoleObservation{
            position: rng.sample(d),
            velocity: rng.sample(d),
            angle: rng.sample(d),
            angular_velocity: rng.sample(d),
        };
        self.terminated = false;
        self.steps_made = 0;
        self.state = Some(state);
        self.payoff = 0.0;
        self.steps_beyond_terminated = None;

        Ok(())
    }
}

impl GameStateWithPayoffs<CartPoleScheme> for CartPoleEnvStateRust {
    fn state_payoff_of_player(&self, _agent: &<CartPoleScheme as Scheme>::AgentId) -> <CartPoleScheme as Scheme>::UniversalReward {
        self.payoff
    }
}