# type: ignore

from typing import Callable, Generic, Iterable, Tuple, TypeVar

import minitorch.operators as operators

A = TypeVar("A")


class MathTest(Generic[A]):
    @staticmethod
    def neg1(a: A) -> A:
        "Negate the argument"
        return -a
    
    @staticmethod
    def sig1(a: A) -> A:
        "Apply sigmoid"
        return operators.sigmoid(a)

    @staticmethod
    def relu2(a: A, b: A) -> A:
        "Apply relu"
        return operators.relu(a + b + 1)

    @staticmethod
    def add2(a: A, b: A) -> A:
        "Add two arguments"
        return a + b

    @staticmethod
    def mul2(a: A, b: A) -> A:
        "Mul two arguments"
        return a * b

    @staticmethod
    def div2(a: A, b: A) -> A:
        "Divide two arguments"
        return a / (b + 5.5)

    @staticmethod
    def gt2(a: A, b: A) -> A:
        return operators.lt(b, a + 1.2)

    @staticmethod
    def lt2(a: A, b: A) -> A:
        return operators.lt(a + 1.2, b)

    @staticmethod
    def eq2(a: A, b: A) -> A:
        return operators.eq(a, (b + 5.5))

    @staticmethod
    def sum_red(a: Iterable[A]) -> A:
        return operators.sum(a)

    @staticmethod
    def mean_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @staticmethod
    def mean_full_red(a: Iterable[A]) -> A:
        return operators.sum(a) / float(len(a))

    @classmethod
    def _tests(
        cls,
    ) -> Tuple[
        Tuple[str, Callable[[A], A]],
        Tuple[str, Callable[[A, A], A]],
        Tuple[str, Callable[[Iterable[A]], A]],
    ]:
        """
        Returns a list of all the math tests.
        """
        one_arg = []
        two_arg = []
        red_arg = []
        for k in dir(MathTest):
            if callable(getattr(MathTest, k)) and not k.startswith("_"):
                base_fn = getattr(cls, k)
                # scalar_fn = getattr(cls, k)
                tup = (k, base_fn)
                if k.endswith("2"):
                    # two_arg
                    two_arg.append(tup)
                elif k.endswith("red"):
                    # reduce
                    red_arg.append(tup)
                elif k.endswith("1"):
                    # one_arg
                    one_arg.append(tup)
                else:
                    raise ValueError(f"Unknown test: {k}")
        return one_arg, two_arg, red_arg

    @classmethod
    def _comp_testing(cls):
        one_arg, two_arg, red_arg = cls._tests()
        one_argv, two_argv, red_argv = MathTest._tests()
        one_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(one_arg, one_argv)]
        two_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(two_arg, two_argv)]
        red_arg = [(n1, f2, f1) for (n1, f1), (n2, f2) in zip(red_arg, red_argv)]
        return one_arg, two_arg, red_arg


class MathTestVariable(MathTest):
    @staticmethod
    def neg1(x):
        return -x

    @staticmethod
    def sig1(x):
        return x.sigmoid()
    
    @staticmethod
    def relu2(x, y):
        return (x + y + 1).relu()

    @staticmethod
    def sum_red(a):
        return a.sum(0)

    @staticmethod
    def mean_red(a):
        return a.mean(0)

    @staticmethod
    def mean_full_red(a):
        return a.mean()

    @staticmethod
    def eq2(a, b):
        return a == (b + 5.5)

    @staticmethod
    def gt2(a, b):
        return a + 1.2 > b

    @staticmethod
    def lt2(a, b):
        return a + 1.2 < b
