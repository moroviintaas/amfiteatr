use tch::{Device, Kind, Tensor};
use amfiteatr_rl::torch_net::TensorMultiParamActorCritic;
use amfiteatr_rl::torch_net::ActorCriticOutput;

fn main() {
    // Assume we have 2 game steps by actor critic
    // We have then 2 critic values (Tensor 2x1)
    // We have 3 parameters of action, of 4,3 and 2 respectively.
    // Then first actor is shape 2x4.
    // The second actor is in shape 2x3.
    // The third us 2x2.
    //
    // Now masks:
    // Assume we took actions:
    // 1. [0,2,0] (Index 0 on first param, index 2 on second, index 0 in third]
    // 2. [2,-,1] (Index 2 on first action, second param is unknown since it not a param wof action type 0, index 1 on third

    let critic = Tensor::from_slice(&[0.7, 0.21]); //first critic calculation, second criric calculation
    let actor = vec![
        {
            let state1_dist0 = Tensor::from_slice(&[-3.0, -4.0, -1.0, -1.0]); //parameter 1 on state1
            let state2_dist0 = Tensor::from_slice(&[-5.0, -2.0, -4.0, -1.0]); //parameter 1 on state2
            Tensor::vstack(&[state1_dist0, state2_dist0])
        },   //first parameter
        {
            let state1_dist1 = Tensor::from_slice(&[-3.0, -4.0, -1.0]); //parameter 2 on state1
            let state2_dist1 = Tensor::from_slice(&[-5.0, -2.0, -4.0]); //parameter 2 on state2
            Tensor::vstack(&[state1_dist1, state2_dist1])
        },
        {
            let state1_dist2 = Tensor::from_slice(&[-3.0, -4.0]); //parameter 3 on state1
            let state2_dist2 = Tensor::from_slice(&[-5.0, -2.0]); //parameter 3 on state2
            Tensor::vstack(&[state1_dist2, state2_dist2])
        },
    ];
    let mca = TensorMultiParamActorCritic { critic, actor, };
    let reverse_masks = vec![
        Tensor::from_slice(&[true, true]),
        Tensor::from_slice(&[true, false]),
        Tensor::from_slice(&[true, true]),
    ];
    let forward_masks = vec![
        {
            let state1_dist0 = Tensor::from_slice(&[true, false, true, false]);
            let state2_dist0 = Tensor::ones(&[4], (Kind::Bool, Device::Cpu));

            //In round (0) oarameter (0) of values (1,3) are illegal
            Tensor::vstack(&[state1_dist0, state2_dist0])
        },
        {
            let state1_dist1 = Tensor::ones(&[3], (Kind::Bool, Device::Cpu));
            let state2_dist1 = Tensor::from_slice(&[false, false, true]);
            //in round (1) parameter (1) of value 0  and 1 are illegal
            Tensor::vstack(&[state1_dist1, state2_dist1])
        },
        {
            let state1_dist2 = Tensor::from_slice(&[false, true]);
            //in round (0) parameter (2) of value (0) is illegal
            let state2_dist2 = Tensor::ones(&[2], (Kind::Bool, Device::Cpu));
            Tensor::vstack(&[state1_dist2, state2_dist2])
        },
    ];
    let entropy_unmasked = mca.batch_entropy_masked(None, None).unwrap();
    let entropy_masked_actions = mca.batch_entropy_masked(Some(&forward_masks), None).unwrap();
    let entropy_masked_reverse = mca.batch_entropy_masked(None, Some(&reverse_masks)).unwrap();
    let entropy_masked = mca.batch_entropy_masked(Some(&forward_masks), Some(&reverse_masks)).unwrap();
    assert_eq!(entropy_unmasked.size2().unwrap(), (2, 3));

    println!("Entropy without masks: (dim=0 is batch dimension, dim=1 is category dimension");
    println!("{}", entropy_unmasked);
    println!("Entropy with illegal actions masked out");
    println!("{}", entropy_masked_actions);
    println!("Entropy with unknown parameters masked out");
    println!("{}", entropy_masked_reverse);
    println!("Entropy with both illegal actions and  unknown parameters masked out");
    println!("{}", entropy_masked);


}
