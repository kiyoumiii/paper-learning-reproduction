def smooth_loss(l, f, i, r):
    return (l + r * min(f-1, i))/(min(f, i+1))
                