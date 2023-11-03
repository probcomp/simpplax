# Copyright 2022 MIT Probabilistic Computing Project
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import abc
import jax.tree_util as jtu
from jax.util import safe_zip
import typing

import beartype.typing as btyping
import jax
import jax.numpy as jnp
import jaxtyping as jtyping
import numpy as np
from beartype import BeartypeConf, beartype
from plum import dispatch
import copy
import dataclasses
import functools
from contextlib import contextmanager

from jax import api_util, lax
from jax import core as jc
from jax import tree_util as jtu
from jax.extend import linear_util as lu
from jax.interpreters import partial_eval as pe
from jax.util import safe_map


##########
# Typing #
##########


Dataclass = typing.Any
PrettyPrintable = typing.Any
PRNGKey = jtyping.UInt[jtyping.Array, "..."]
FloatArray = typing.Union[float, jtyping.Float[jtyping.Array, "..."]]
BoolArray = typing.Union[bool, jtyping.Bool[jtyping.Array, "..."]]
IntArray = typing.Union[int, jtyping.Int[jtyping.Array, "..."]]
Any = typing.Any
Union = typing.Union
Callable = btyping.Callable
Sequence = typing.Sequence
Tuple = btyping.Tuple
NamedTuple = btyping.NamedTuple
Dict = btyping.Dict
List = btyping.List
Iterable = btyping.Iterable
Generator = btyping.Generator
Hashable = btyping.Hashable
FrozenSet = btyping.FrozenSet
Optional = btyping.Optional
Type = btyping.Type
Int = int
Float = float
Bool = bool
String = str
Value = Any
Generic = btyping.Generic
TypeVar = btyping.TypeVar

conf = BeartypeConf()
typecheck = beartype(conf=conf)


#################
# Hashable dict #
#################


class HashableDict(dict):
    """
    A hashable dictionary class - allowing the
    usage of `dict`-like instances as JAX JIT cache keys
    (and allowing their usage with JAX `static_argnums` in `jax.jit`).
    """

    def __key(self):
        return tuple((k, self[k]) for k in sorted(self, key=hash))

    def __hash__(self):
        return hash(self.__key())

    def __eq__(self, other):
        return self.__key() == other.__key()


# This ensures that static keys are always sorted
# in a pre-determined order - which is important
# for `Pytree` structure comparison.
def _flatten(x: HashableDict):
    s = {k: v for (k, v) in sorted(x.items(), key=lambda v: hash(v[0]))}
    return (list(s.values()), list(s.keys()))


jtu.register_pytree_node(
    HashableDict,
    _flatten,
    lambda keys, values: HashableDict(safe_zip(keys, values)),
)


def hashable_dict():
    return HashableDict({})


###########
# Pytrees #
###########


class Pytree:
    """> Abstract base class which registers a class with JAX's `Pytree`
    system.

    Users who mixin this ABC for class definitions are required to
    implement `flatten` below. In turn, instances of the class gain
    access to a large set of utility functions for working with `Pytree`
    data, as well as the ability to use `jax.tree_util` Pytree
    functionality.
    """

    def __init_subclass__(cls, **kwargs):
        jtu.register_pytree_node(
            cls,
            cls.flatten,
            cls.unflatten,
        )

    @abc.abstractmethod
    def flatten(self) -> Tuple[Tuple, Tuple]:
        pass

    @classmethod
    def unflatten(cls, data, xs):
        return cls(*data, *xs)

    @classmethod
    def new(cls, *args, **kwargs):
        return cls(*args, **kwargs)


############################################################
# Staging a function to a closed Jaxpr, for interpretation #
############################################################


def get_shaped_aval(x):
    return jc.raise_to_shaped(jc.get_aval(x))


# This will create a cache entry for a function with abstract
# arguments (each has a dtype and shape).
# It is useful to cache the result, because you may stage the same
# function multiple times - if you write and compose multiple transformations.
@lu.cache
def cached_stage_dynamic(flat_fun, in_avals):
    # https://github.com/google/jax/blob/1c66ac532b4ef2ad64f9e0859ede329f4fbd0041/jax/_src/interpreters/partial_eval.py#L2268-L2305
    jaxpr, _, consts = pe.trace_to_jaxpr_dynamic(flat_fun, in_avals)

    typed_jaxpr = jc.ClosedJaxpr(jaxpr, consts)
    return typed_jaxpr


def stage(f):
    """Returns a function that stages a function to a ClosedJaxpr."""

    def wrapped(*args, **kwargs):
        fun = lu.wrap_init(f, kwargs)
        flat_args, in_tree = jtu.tree_flatten(args)
        flat_fun, out_tree = api_util.flatten_fun_nokwargs(fun, in_tree)
        flat_avals = safe_map(get_shaped_aval, flat_args)
        typed_jaxpr = cached_stage_dynamic(flat_fun, tuple(flat_avals))
        return typed_jaxpr, (flat_args, in_tree, out_tree)

    return wrapped


#######################
# Forward interpreter #
#######################

VarOrLiteral = Union[jc.Var, jc.Literal]


@dataclasses.dataclass
class Environment(Pytree):
    """Keeps track of variables and their values during interpretation."""

    env: HashableDict[jc.Var, Value]

    def flatten(self):
        return (self.env,), ()

    @classmethod
    def new(cls):
        return Environment(hashable_dict())

    def read(self, var: VarOrLiteral) -> Value:
        if isinstance(var, jc.Literal):
            return var.val
        else:
            return self.env.get(var.count)

    def write(self, var: VarOrLiteral, cell: Value) -> Value:
        if isinstance(var, jc.Literal):
            return cell
        cur_cell = self.read(var)
        if isinstance(var, jc.DropVar):
            return cur_cell
        self.env[var.count] = cell
        return self.env[var.count]

    def __getitem__(self, var: VarOrLiteral) -> Value:
        return self.read(var)

    def __setitem__(self, key, val):
        raise ValueError(
            "Environments do not support __setitem__. Please use the "
            "`write` method instead."
        )

    def __contains__(self, var: VarOrLiteral):
        if isinstance(var, jc.Literal):
            return True
        return var in self.env

    def copy(self):
        return copy.copy(self)


###############################
# Forward masking interpreter #
###############################


@dataclasses.dataclass
class ForwardInterpreter(Pytree):
    def flatten(self):
        return (), ()

    # This produces an instance of `Interpreter`
    # as a context manager - to allow us to control error stack traces,
    # if required.
    @classmethod
    @contextmanager
    def new(cls):
        try:
            yield ForwardInterpreter()
        except Exception as e:
            raise e

    def _eval_jaxpr_forward(
        self,
        jaxpr: jc.Jaxpr,
        consts: List[Value],
        flat_args: List[Value],
    ):
        env = Environment.new()
        safe_map(env.write, jaxpr.constvars, consts)
        safe_map(env.write, jaxpr.invars, flat_args)
        for eqn in jaxpr.eqns:
            invals = safe_map(env.read, eqn.invars)
            subfuns, params = eqn.primitive.get_bind_params(eqn.params)
            args = subfuns + invals
            custom_rule = masking_registry[eqn.primitive]
            outvals = eqn.primitive.bind(*args, **params)
            if not eqn.primitive.multiple_results:
                outvals = [outvals]
            safe_map(env.write, eqn.outvars, outvals)

        return safe_map(env.read, jaxpr.outvars)

    def run_interpreter(self, fn, args, **kwargs):
        def _inner(*args):
            return fn(*args, **kwargs)

        closed_jaxpr, (flat_args, _, out_tree) = stage(_inner)(*args)
        flat_mask_flags = jtu.tree_leaves(mask_flags)
        jaxpr, consts = closed_jaxpr.jaxpr, closed_jaxpr.literals
        flat_out = self._eval_jaxpr_forward(jaxpr, consts, flat_args)
        return jtu.tree_unflatten(out_tree(), flat_out)


def forward(f: Callable):
    @functools.wraps(f)
    def wrapped(args, mask_flags):
        with ForwardInterpreter.new() as interpreter:
            return interpreter.run_interpreter(f, args, mask_flags)

    return wrapped
