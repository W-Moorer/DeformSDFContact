# Prototype Load Stepping

## Scope

This document defines the minimal load-stepping / continuation path used by the
current monolithic prototype.

The implementation lives in:

- `src/deformsdfcontact/solvers/load_stepping.py`

## Current Policy

The current stepping path is intentionally simple:

- one scalar continuation parameter
- a fixed user-provided step schedule
- the converged state from one step is reused as the initial guess of the next
  step

No adaptive step control is implemented.

## Current Continuation Parameter

The current toy and mini benchmark problems use a manufactured SDF scaling
parameter `phi_scale`.

At each step:

- a new transition problem is built for the requested `phi_scale`
- the previous converged state is clamped to the new boundary conditions
- the same monolithic solve path is reused

This gives a minimal continuation path without introducing a more elaborate
load-control system.

## Step Results

Each step records:

- the step index
- the continuation parameter value
- whether the solve converged
- the full prototype solver result

If one step fails, the stepping path stops and reports the failed step index.
The current prototype also reports:

- the last converged step index
- the last converged continuation parameter value

## What This Layer Does Not Do

This layer does **not** implement:

- cutback
- adaptive continuation
- arc-length methods
- production load-control policies

It is a prototype continuation path for the current transition environment.
