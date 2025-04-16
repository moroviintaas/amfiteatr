



class PolicyConfig:
    def __init__(self, gamma=0.99,  vf_coef=0.5, entropy_coef=0.01, lr=5e-4, masking=False):
        self.gamma = gamma
        self.vf_coef = vf_coef
        self.entropy_coef = entropy_coef
        self.learning_rate = lr
        self.masking = masking

