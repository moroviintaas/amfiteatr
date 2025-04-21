



class PolicyConfig:
    def __init__(self, gamma=0.99,  vf_coef=0.5, entropy_coef=0.01, \
                 learning_rate=5e-4, masking=False, batch_size=64, minibatch_size=64, \
                 update_epochs=4, clip_coef=0.2, norm_adv=False, clip_vloss=True,\
                 max_grad_norm = 0.5, target_kl=None, gae_lambda=0.95):
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.clip_vloss = clip_vloss
        self.entropy_coef = entropy_coef
        self.learning_rate = learning_rate
        self.masking = masking
        self.batch_size = batch_size
        self.minibatch_size= minibatch_size
        self.update_epochs = update_epochs
        self.clip_coef = clip_coef
        self.gae_lambda = gae_lambda

        self.norm_adv = norm_adv
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
