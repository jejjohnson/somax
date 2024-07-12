from somax.domain import Domain


class DiscretizationError(Exception):
    def __init__(self, d1, d2):
        self.message = f"Mismatched spatial discretizations\n{d1}\n{d2}"
        self.message += f"\nd1: {d1.Nx} | d2: {d2.Nx}"
        super().__init__(self.message)


def check_discretization(d1: Domain, d2: Domain):
    assert type(d1) == type(d2)

    if not (
        # check type
        type(d1) == type(d2)
        or
        # check shape
        d1.shape == d2.shape
        or
        # check size
        d1.size == d2.size
        or
        # check bounds
        d1.bounds == d2.bounds
    ):
        print(d1, d2)
        raise DiscretizationError(d1, d2)
