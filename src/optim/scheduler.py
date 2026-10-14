import math


def get_lr(step, max_lr, min_lr, warmup_steps, max_steps):

    if step < warmup_steps:
        return max_lr * step / warmup_steps

    if step >= max_steps:
        return min_lr

    p = (step - warmup_steps ) / (max_steps - warmup_steps)
    return min_lr + 0.5 * (max_lr - min_lr) * (1 + math.cos(p * math.pi))

if __name__ == '__main__':
    for step in [0, 50, 100, 550, 1000, 1200]:
        print(step, get_lr(step, 0.001, 0.0001, 100, 1000))