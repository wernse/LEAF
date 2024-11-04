from utils.buffer.adaptive_retrieve import Adaptive_retrieve
from utils.buffer.gss_greedy_task_boundary import GSSGreedyUpdateTaskBoundary
from utils.buffer.random_retrieve import Random_retrieve
from utils.buffer.reservoir_update import Reservoir_update
from utils.buffer.mir_retrieve import MIR_retrieve
from utils.buffer.weighted_reservoir_update import Weighted_Reservoir_update

retrieve_methods = {
    'MIR': MIR_retrieve,
    'random': Random_retrieve,
    'adaptive': Adaptive_retrieve,
}

update_methods = {
    'random': Reservoir_update,
    'weighted': Weighted_Reservoir_update,
    'GSSTask': GSSGreedyUpdateTaskBoundary,
}

