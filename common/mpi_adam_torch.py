import numpy as np
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
import torch


# total number of elements in a tensor as int (multiple all dims together)
def numel(x):
    shape = [*x.shape]
    return int(np.prod(shape))

def intprod(x):
    return int(np.prod(x))

def var_shape(x):
    # return x.get_shape().as_list()
    return [*x.shape]


class MpiAdam(object):
    def __init__(self, network, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-08, scale_grad_by_procs=False, comm=None):
        self.network = network
        var_list = self.get_vars()
        self.var_list = var_list
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.scale_grad_by_procs = scale_grad_by_procs  # always False
        size = sum(numel(v) for v in var_list)  # total number of params and biases in network
        assert size in [136196, 136449]
        self.m = np.zeros(size, 'float32')
        self.v = np.zeros(size, 'float32')
        self.t = 0
        # always MPI.COMM_WORLD. Input comm is always None
        self.comm = MPI.COMM_WORLD if comm is None and MPI is not None else comm
        self.step_update = None

    def get_flat(self):
        # these params are in weight1, bias1, weight2... order
        params = list(self.network.parameters())
        flat_grads = torch.cat([param.view([numel(param)]) for param in params], dim=0)
        return flat_grads.clone().detach().numpy()

    def get_grads(self):
        params = list(self.network.parameters())
        grads = [param.grad for param in params]
        flat_grads = torch.cat([grad.view([numel(grad)]) for grad in grads], dim=0)
        return flat_grads.clone().detach().numpy()

    def set_from_flat(self, theta):
        shapes = list(map(var_shape, self.var_list))
        start = 0
        updated_weights = list()
        for (shape, v) in zip(shapes, self.var_list):  # shape is shape of that weight/bias variable
            size = intprod(shape)  # total elements in variable
            new_weights = theta[start:start + size].reshape(shape)
            updated_weights.append(torch.nn.Parameter(torch.tensor(new_weights)))
            start += size
        self.network.linear1.weight = updated_weights[0]
        self.network.linear1.bias = updated_weights[1]
        self.network.linear2.weight = updated_weights[2]
        self.network.linear2.bias = updated_weights[3]
        self.network.linear3.weight = updated_weights[4]
        self.network.linear3.bias = updated_weights[5]
        self.network.linear4.weight = updated_weights[6]
        self.network.linear4.bias = updated_weights[7]


    def get_vars(self):
        vars = list()
        vars.append(self.network.linear1.weight)
        vars.append(self.network.linear1.bias)
        vars.append(self.network.linear2.weight)
        vars.append(self.network.linear2.bias)
        vars.append(self.network.linear3.weight)
        vars.append(self.network.linear3.bias)
        vars.append(self.network.linear4.weight)
        vars.append(self.network.linear4.bias)
        return vars


    def update(self):
        localg = self.get_grads()  # this is totally flat
        if self.t % 100 == 0:
            self.check_synced()
        localg = localg.astype('float32')
        globalg = np.zeros_like(localg)
        self.comm.Allreduce(localg, globalg, op=MPI.SUM)

        self.t += 1
        a = self.learning_rate * np.sqrt(1 - self.beta2**self.t)/(1 - self.beta1**self.t)
        self.m = self.beta1 * self.m + (1 - self.beta1) * globalg
        self.v = self.beta2 * self.v + (1 - self.beta2) * (globalg * globalg)
        step = (- a) * self.m / (np.sqrt(self.v) + self.epsilon)
        self.step_update = step
        self.set_from_flat(self.get_flat() + step)

    def sync(self):
        theta = self.get_flat()
        self.comm.Bcast(theta, root=0)
        self.set_from_flat(theta)

    def check_synced(self):
        if self.comm.Get_rank() == 0: # this is root
            theta = self.get_flat()
            self.comm.Bcast(theta, root=0)
        else:
            thetalocal = self.get_flat()
            thetaroot = np.empty_like(thetalocal)
            self.comm.Bcast(thetaroot, root=0)
            assert (thetaroot == thetalocal).all(), (thetaroot, thetalocal)
