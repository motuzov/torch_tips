import torch
from torch import nn
from torch import autograd


class MPointwiseG(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(
            torch.ones(1, dtype=torch.float),
            requires_grad=True
        )

    def reset_w(self):
        self.w = nn.Parameter(
            torch.ones(1, torch.float),
            requires_grad=True
        )

    def forward(self, x: torch.tensor) -> torch.tensor:
        return (self.w ** 2) * (x ** 3)


def main():
    m = MPointwiseG()
    x = torch.tensor(2.)
    print(x)
    vx = autograd.Variable(x, requires_grad=True)
    # out is a graph
    f_out = m(vx)
    print(f'out_graph={f_out}')
    grad_x = autograd.grad(f_out, vx, create_graph=True, retain_graph=True)
    vx.requires_grad = False
    # df/dx = w^2*3*x^2(x=2)
    print(f'grad_x={grad_x}')
    print(f'grad_x.grad_fn={grad_x[0].grad_fn}')
    g_out = f_out + grad_x[0]
    g_out.backward()
    # g = f + df/dx = w^2*x^3 + w^2*3*x^2(x=2)
    #
    # dg/dw(x=2) = 2*w*x^3(w=1,x=2) + 2*w*3*x^2(w=1,x=2)
    #              2 * 8            + 2 * 3*4 = 16 + 2*12 = 16 + 24
    # 40
    print(f'g_grad_w(x)={m.w.grad.data}')
    print(f'g_grad_w(x)={m.w.grad}')
    print('m')


if __name__ == '__main__':
    main()
