"""Measure the 0.39 first-iteration loop entry and sparse-table tranche.

The inherited 0.38 cases remain available as regression controls. The focused
case is the typed Sieve workload that motivated these two generated-code paths.
Run this unchanged against baseline and candidate trees.
"""
from __future__ import annotations

import sys

import speed_038_ab as _base
from speed_038_ab import *  # noqa: F401,F403


def sieve_reference():
    composite = set()
    count = 0
    for prime in range(2, 5001):
        if prime not in composite:
            count += 1
            multiple = prime * prime
            while multiple <= 5000:
                composite.add(multiple)
                multiple += prime
    return count


CASES_039 = {
    "sieve_sparse": Case("""
local n: integer = 5000
local composite: table = {}
local count: integer = 0
for p = 2, n do
    if not composite[p] then
        count = count + 1
        local multiple: integer = p * p
        while multiple <= n do
            composite[multiple] = true
            multiple = multiple + p
        end
    end
end
return count
""", sieve_reference, 669,
        "First-entry nested loop with sparse integer-to-boolean storage"),
}

_base.CASES.update(CASES_039)
CASES = _base.CASES


if __name__ == "__main__":
    if "--case" not in sys.argv:
        sys.argv.extend(("--case", "sieve_sparse"))
    main()
