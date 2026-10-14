import torch

class AdamW:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01):
        self.params = list(params)
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.weight_decay = weight_decay

        self.m = [torch.zeros_like(p) for p in self.params]
        self.v = [torch.zeros_like(p) for p in self.params]
        self.steps = [0] * len(self.params)

    @torch.no_grad()
    def step(self):
        for i, p in enumerate(self.params):
            if p.grad is None:
                continue

            g = p.grad
            self.steps[i] += 1
            t = self.steps[i]

            self.m[i].mul_(self.beta1).add_(g, alpha=1 - self.beta1)
            self.v[i].mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

            m_hat = self.m[i] / (1 - self.beta1 ** t)
            v_hat = self.v[i] / (1 - self.beta2 ** t)

            update = m_hat / (v_hat.sqrt() + self.eps)

            p.mul_(1 - self.lr * self.weight_decay)
            p.add_(update, alpha=-self.lr)

    def zero_grad(self):
        for p in self.params:
            p.grad = None

    def state_dict(self):
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "m": [x.clone() for x in self.m],
            "v": [x.clone() for x in self.v],
            "steps": self.steps.copy(),
        }

    def load_state_dict(self, state):
        if len(state["m"]) != len(self.params):
            raise ValueError("Optimizer parameter count mismatch")

        self.lr = state["lr"]
        self.beta1, self.beta2 = state["betas"]
        self.eps = state["eps"]
        self.weight_decay = state["weight_decay"]

        for i, p in enumerate(self.params):
            if state["m"][i].shape != p.shape or state["v"][i].shape != p.shape:
                raise ValueError("Optimizer parameter shape mismatch")

        self.m = [x.to(p.device, dtype=p.dtype).clone() for x, p in zip(state["m"], self.params)]
        self.v = [x.to(p.device, dtype=p.dtype).clone() for x, p in zip(state["v"], self.params)]
        self.steps = state["steps"].copy()
