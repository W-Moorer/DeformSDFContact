# Transition Monolithic Problem Wrapper

## Scope

This document defines the minimal problem wrapper used by the first solver-ready
prototype in the current transition environment.

The implementation lives in:

- `src/deformsdfcontact/backend/dolfinx0p3/problem.py`

## Responsibilities

`TransitionMonolithicProblem` only packages the information needed to expose:

- a monolithic residual callable
- a monolithic Jacobian callable
- PETSc vector/matrix templates
- a prototype initial guess

It holds:

- the mesh and function spaces
- the frozen monolithic block layout
- the transition residual/Jacobian callables
- the prototype external loads
- the prototype boundary conditions

## What It Is Not

This wrapper is intentionally **not**:

- a general research framework
- a production multi-backend problem abstraction
- the final long-term architecture

It is a transition-environment problem wrapper whose only goal is to expose a
clean solver entry point for the first monolithic prototype.

## Current Toy Problem

The current unit-square toy problem is a manufactured transition case:

- structural exact state `u* = 0`
- SDF exact state `phi*(x, y) = y`
- transition contact offset chosen so that part of the boundary remains active
- structural external load manufactured from the assembled contact contribution
  at the reference state

This gives:

- a known reference state for regression
- nontrivial monolithic block coupling
- a small problem that is still easy to verify
