"""
Depracated, using ``vectorized_value_and_grad`` instead for batched optimization
"""

import sys

sys.path.insert(0, "../")

import tyxonq as tq
from tyxonq.applications.dqas import (
    parallel_qaoa_train,
    single_generator,
    set_op_pool,
)
from tyxonq.applications.layers import *  # pylint: disable=wildcard-import
from tyxonq.applications.graphdata import get_graph

tq.set_backend("pytorch")

set_op_pool([Hlayer, rxlayer, rylayer, rzlayer, xxlayer, yylayer, zzlayer])


if __name__ == "__main__":
    # old fashion example, prefer vvag for new generation software for this task
    parallel_qaoa_train(
        [0, 6, 1, 6, 1],
        single_generator(get_graph("8B")),
        tries=4,
        cores=2,
        batch=1,
        epochs=5,
        scale=0.8,
    )
