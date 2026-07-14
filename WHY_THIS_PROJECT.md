# Why This Project

Existing frameworks provide strong implementations and deployment stacks. This project focuses on making implementation decisions explicit, IR-rooted, evidence-backed, provenance-tracked, and materialized into an exact execution contract.

## Why not simply use ExecuTorch?

ExecuTorch is a strong runtime/export stack. This project can use ExecuTorch/XNNPACK as an implementation candidate. The research question here is how a compiler should represent, validate, select, and contractually execute such choices, not whether every kernel should be reimplemented.

## Why not simply use XNNPACK?

XNNPACK is a kernel library. It does not by itself provide the project-level IR decision pipeline: semantic identity, candidate enumeration across possible implementations, feasibility, provenance, policy, contract generation, and runtime validation.

## Why not simply use vLLM?

vLLM is a serving runtime. This project can materialize vLLM configurations and use vLLM evidence. It does not claim to replace vLLM internals.

## Why not simply use TVM, ONNX Runtime, or TensorRT?

Those systems can be candidates or baselines. The project’s focus is the implementation-decision compiler layer above and around such systems: when is a backend legal, what evidence supports it, what exactly was selected, and how does Runtime prove it executed the selected contract?

## Core Question

Can an Edge AI compiler make backend/kernel/thread/artifact decisions explicit, inspectable, calibrated, and enforceable without hiding those decisions inside benchmark scripts or runtime fallback behavior?
