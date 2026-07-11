// QuantizationCoDesignPass — quantization co-design evidence and contract
// (quantization_codesign_contract_v1).
//
// For each matmul-like constant-weight op, this pass answers — as SEPARATE
// facts, never collapsed into one string:
//   1. representation      — which numeric representation is under
//                            consideration (weight_only_int8 / _int4)
//   2. algorithm           — whether any quantization ALGORITHM is even
//                            declared (only the forced-AWQ profile declares
//                            one today; none is implemented in this repo)
//   3. backend legality    — does the declared backend capability expose
//                            the representation
//   4. kernel support      — is there a CONCRETE runtime kernel descriptor
//                            (target.runtime_kernels) that can consume
//                            quantized weights — distinct from backend
//                            capability
//   5. systems cost        — static before/after estimate via
//                            estimateShapeCost (see the cost-term table in
//                            docs/QUANTIZATION_CODESIGN.md for every term,
//                            source, unit, and inclusion boundary)
//   6. accuracy evidence   — honest status; no calibration or measured
//                            accuracy exists in this repository, so the
//                            only possible values today are
//                            no_accuracy_evidence and
//                            algorithm_declared_not_calibrated
//   7. materialization     — whether a float dequant intermediate would
//                            have to be materialized, and why it currently
//                            cannot be (no scale/zero-point metadata)
//
// The pass is INERT unless the module opts in via
// quant.codesign.policy ∈ {planning_only, systems_cost_only,
// require_dispatchable_kernel, require_accuracy_evidence}. It never
// modifies quant.strategy or any other existing planning attr, and its
// quant_codesign.est.* evidence is never read by CandidateEvaluation or
// PlanSelection (ranking invariance is enforced by a test).
//
// Unknown metadata (granularity, group size, axis, symmetric/asymmetric,
// scale, zero point) is OMITTED, not defaulted. scale/zero_point sources
// are reported as not available: no calibration exists.
//
// Cost-model honesty note: a dispatchable weight-only kernel eliminates the
// MATERIALIZED float dequant intermediate (boundary traffic term -> 0). It
// does NOT imply zero scale/metadata/unpacking/inline-conversion cost —
// that term is not modeled and is explicitly listed in
// quant_codesign.est.excluded_terms rather than silently treated as free.

#include "serving/OpShapeFacts.h"
#include "serving/ShapeCostModel.h"
#include "FusionPasses.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include <memory>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {

#define GEN_PASS_DEF_QUANTIZATIONCODESIGN
#include "FusionPasses.h.inc"

static constexpr StringLiteral kContractVersion =
    "quantization_codesign_contract_v1";
static constexpr StringLiteral kTruth =
    "quantization_codesign_static_planning_no_calibration_no_measured_accuracy";

static std::vector<std::string> readStringArr(Operation* op, StringRef name) {
  std::vector<std::string> out;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (Attribute e : a)
      if (auto s = dyn_cast<StringAttr>(e)) out.push_back(s.getValue().str());
  return out;
}

static bool inList(const std::vector<std::string>& v, StringRef s) {
  for (const auto& e : v)
    if (e == s) return true;
  return false;
}

// One quantized-representation candidate under evaluation.
struct QuantCandidate {
  std::string representation; // "weight_only_int8" | "weight_only_int4"
  std::string weight_dtype;   // "int8" | "int4"
  int64_t weight_bits = 0;
};

// Kernel-support fact for a candidate — structured, not an enum-encoded id.
struct KernelSupport {
  // "runtime_dispatchable" | "backend_supported_but_not_dispatchable" |
  // "no_kernel_registry_declared"
  std::string status;
  std::string kernel_id; // only when runtime_dispatchable
  std::string source;    // descriptor source, only when runtime_dispatchable
};

struct QuantizationCoDesignPass
    : impl::QuantizationCoDesignBase<QuantizationCoDesignPass> {
  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext* ctx = funcOp.getContext();
    if (funcOp.getBody().empty()) return;
    Operation* module = funcOp->getParentOp();

    // Opt-in policy gate: absent attr -> the pass is inert and existing
    // artifacts stay byte-identical.
    std::string policy;
    if (module)
      if (auto a = module->getAttrOfType<StringAttr>("quant.codesign.policy"))
        policy = a.getValue().str();
    if (policy.empty()) return;
    if (policy != "planning_only" && policy != "systems_cost_only" &&
        policy != "require_dispatchable_kernel" &&
        policy != "require_accuracy_evidence") {
      funcOp->emitError()
          << "quantization-codesign: unknown quant.codesign.policy '"
          << policy
          << "' (expected planning_only | systems_cost_only | "
             "require_dispatchable_kernel | require_accuracy_evidence)";
      signalPassFailure();
      return;
    }

    auto S = [&](StringRef s) { return StringAttr::get(ctx, s); };
    auto I64 = [&](int64_t v) {
      return IntegerAttr::get(IntegerType::get(ctx, 64), v);
    };
    auto B = [&](bool v) { return BoolAttr::get(ctx, v); };

    StaticCostProfileNums nums = readProfileNums(module);

    std::string backend;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.source_backend"))
      backend = a.getValue().str();
    else if (auto a =
                 funcOp->getAttrOfType<StringAttr>("execution_provider.primary"))
      backend = a.getValue().str();

    std::string effectiveDtype;
    if (auto a =
            funcOp->getAttrOfType<StringAttr>("representation.effective_dtype"))
      effectiveDtype = a.getValue().str();

    // Backend-declared quantization capability (declared_profile facts).
    std::vector<std::string> quantModes, backendDtypes, accumDtypes;
    if (module && !backend.empty()) {
      std::string p = "target.backend_capabilities." + backend + ".";
      quantModes    = readStringArr(module, p + "supported_quant_modes");
      backendDtypes = readStringArr(module, p + "supported_dtypes");
      accumDtypes   = readStringArr(module, p + "accumulation_dtypes");
    }
    bool int8Legal = inList(quantModes, "weight_only") ||
                     inList(backendDtypes, "int8") ||
                     inList(backendDtypes, "i8");
    bool int4Legal = inList(quantModes, "weight_only_int4") ||
                     inList(backendDtypes, "int4") ||
                     inList(backendDtypes, "i4");

    // Algorithm DECLARATION (never implementation): only the forced-quant
    // profile path declares one via module attrs.
    std::string algorithmName;
    std::string artifactRef;
    if (module) {
      if (auto a = module->getAttrOfType<StringAttr>("quantization.algorithm"))
        algorithmName = a.getValue().str();
      if (auto a = module->getAttrOfType<StringAttr>(
              "quantization.quantized_model_artifact_ref"))
        artifactRef = a.getValue().str();
    }
    // Honest accuracy evidence: nothing in this repository is calibrated or
    // accuracy-measured. A declared external algorithm upgrades the status
    // only to algorithm_declared_not_calibrated.
    std::string accuracyStatus = algorithmName.empty()
                                     ? "no_accuracy_evidence"
                                     : "algorithm_declared_not_calibrated";
    bool accuracyEvidenceSufficient = false; // calibrated/measured: never today

    // Runtime kernel registry (kernel_selection_contract_v1 descriptors).
    bool registryDeclared = false;
    ArrayAttr registry;
    if (module)
      if ((registry =
               module->getAttrOfType<ArrayAttr>("target.runtime_kernels")))
        registryDeclared = true;

    for (Operation& op : funcOp.getBody().front().without_terminator()) {
      if (classifyOpKind(op) != "matmul_like")
        continue; // V1 scope: matmul-like constant-weight ops only.

      op.setAttr("quant_codesign.contract_version", S(kContractVersion));
      op.setAttr("quant_codesign.policy", S(policy));
      op.setAttr("quant_codesign.truth_boundary", S(kTruth));
      op.setAttr("quant_codesign.accuracy_evidence.status", S(accuracyStatus));
      if (!artifactRef.empty())
        op.setAttr("quant_codesign.accuracy_evidence.artifact_ref",
                   S(artifactRef));
      op.setAttr("quant_codesign.algorithm.status",
                 S(algorithmName.empty() ? "not_declared"
                                         : "declared_external"));
      if (!algorithmName.empty())
        op.setAttr("quant_codesign.algorithm.name", S(algorithmName));

      SmallVector<Attribute> reasons;
      auto finish = [&](StringRef status) {
        op.setAttr("quant_codesign.status", S(status));
        if (!reasons.empty())
          op.setAttr("quant_codesign.rejection_reasons",
                     ArrayAttr::get(ctx, reasons));
      };

      // A. Graph legality: constant weights required. Absent classification
      // is unknown — deferred, never assumed constant.
      bool funcWeightsConstant = false;
      if (auto a = funcOp->getAttrOfType<BoolAttr>(
              "representation.weights_are_constant"))
        funcWeightsConstant = a.getValue();
      auto opConst = op.getAttrOfType<BoolAttr>("weight.constant_satisfied");
      if (opConst && !opConst.getValue()) {
        reasons.push_back(S("weight_not_compile_time_constant"));
        finish("rejected_weight_not_constant");
        continue;
      }
      if (!opConst && !funcWeightsConstant) {
        reasons.push_back(S("weight_classification_not_available"));
        finish("deferred_missing_weight_classification");
        continue;
      }

      // B. Backend legality (declared profile capability only).
      SmallVector<QuantCandidate, 2> candidates;
      if (int8Legal)
        candidates.push_back({"weight_only_int8", "int8", 8});
      if (int4Legal)
        candidates.push_back({"weight_only_int4", "int4", 4});
      op.setAttr("quant_codesign.backend_legality",
                 S(candidates.empty() ? "not_legal" : "legal"));
      if (candidates.empty()) {
        reasons.push_back(
            S("backend_declares_no_quantized_weight_representation"));
        finish("rejected_backend_not_legal");
        continue;
      }

      // Shared per-op facts.
      ShapeFacts facts = computeShapeFacts(op);
      std::string actDtype = resolveOpActivationDtype(op, effectiveDtype);
      int64_t actBits = dtypeBits(actDtype);

      // C+D. For each backend-legal candidate: concrete kernel support and
      // static systems-cost estimate; keep the best by after-cost.
      struct Evaluated {
        QuantCandidate cand;
        KernelSupport kernel;
        bool hasBytes = false;
        int64_t wBytesBefore = 0, wBytesAfter = 0, boundaryBytes = 0;
        bool hasNanos = false;
        int64_t beforeNanos = 0, afterNanos = 0, benefitNanos = 0;
      };
      SmallVector<Evaluated, 2> evals;
      for (const QuantCandidate& cand : candidates) {
        Evaluated ev;
        ev.cand = cand;

        // Concrete runtime-kernel support: a descriptor for this op name and
        // backend whose supported_quant_modes contain "weight_only" and whose
        // supported_dtypes accept the activation dtype. Backend-library
        // capability never counts as dispatchable.
        if (!registryDeclared) {
          ev.kernel.status = "no_kernel_registry_declared";
        } else {
          ev.kernel.status = "backend_supported_but_not_dispatchable";
          StringRef fullName = op.getName().getStringRef();
          StringRef shortName = fullName;
          if (auto dot = fullName.find('.'); dot != StringRef::npos)
            shortName = fullName.substr(dot + 1);
          for (Attribute e : registry) {
            auto dict = dyn_cast<DictionaryAttr>(e);
            if (!dict) continue;
            auto dS = [&](StringRef k) -> std::string {
              if (auto a = dict.get(k))
                if (auto s = dyn_cast<StringAttr>(a))
                  return s.getValue().str();
              return {};
            };
            auto dArr = [&](StringRef k) -> std::vector<std::string> {
              std::vector<std::string> r;
              if (auto a = dict.get(k))
                if (auto arr = dyn_cast<ArrayAttr>(a))
                  for (auto el : arr)
                    if (auto s = dyn_cast<StringAttr>(el))
                      r.push_back(s.getValue().str());
              return r;
            };
            if (dS("op_name") != shortName.str()) continue;
            if (dS("backend") != backend) continue;
            auto modes = dArr("supported_quant_modes");
            if (!inList(modes, "weight_only")) continue;
            auto dtypes = dArr("supported_dtypes");
            if (!dtypes.empty() && !inList(dtypes, actDtype)) continue;
            ev.kernel.status = "runtime_dispatchable";
            ev.kernel.kernel_id = dS("kernel_id");
            ev.kernel.source = dS("source");
            break;
          }
        }

        // Static systems-cost estimate (see docs/QUANTIZATION_CODESIGN.md
        // for term definitions). Only with usable static shapes and a
        // resolvable dtype; nanos additionally require declared profile
        // peaks. Nothing is estimated for dynamic shapes.
        if (facts.usable() && actBits > 0) {
          ev.hasBytes = true;
          ev.wBytesBefore = facts.weight_elems * actBits / 8;
          ev.wBytesAfter = facts.weight_elems * cand.weight_bits / 8;
          // Boundary term: the MATERIALIZED float dequant intermediate
          // (write quantized->float weights to memory before the kernel),
          // same formula as ShapeCostModel's dequant boundary. A
          // dispatchable weight-only kernel consumes quantized weights
          // directly, so that materialized intermediate — and only it —
          // disappears; inline conversion cost is NOT modeled (see
          // est.excluded_terms).
          ev.boundaryBytes = ev.kernel.status == "runtime_dispatchable"
                                 ? 0
                                 : 2 * ev.wBytesBefore;
          ShapeCostEstimateResult before =
              estimateShapeCost(facts, actBits, actBits, 0, nums);
          ShapeCostEstimateResult after = estimateShapeCost(
              facts, actBits, cand.weight_bits, ev.boundaryBytes, nums);
          if (before.has_time_estimates && after.has_time_estimates) {
            ev.hasNanos = true;
            ev.beforeNanos = before.estimated_total_cost_nanos;
            ev.afterNanos = after.estimated_total_cost_nanos;
            ev.benefitNanos = ev.beforeNanos - ev.afterNanos;
          }
        }
        evals.push_back(std::move(ev));
      }

      // Choose the candidate to report: best benefit when estimable,
      // otherwise the first (int8 precedes int4 by construction).
      const Evaluated* best = &evals[0];
      for (const Evaluated& ev : evals)
        if (ev.hasNanos && (!best->hasNanos ||
                            ev.benefitNanos > best->benefitNanos))
          best = &ev;

      // Emit the separated facts for the reported candidate. Unknown
      // metadata (granularity, group size, axis, symmetric/asymmetric) is
      // omitted — no pass in this repository produces it.
      op.setAttr("quant_codesign.candidate.representation",
                 S(best->cand.representation));
      op.setAttr("quant_codesign.candidate.weight_dtype",
                 S(best->cand.weight_dtype));
      if (actBits > 0)
        op.setAttr("quant_codesign.candidate.activation_dtype", S(actDtype));
      if (!accumDtypes.empty())
        op.setAttr("quant_codesign.candidate.accumulator_dtype",
                   S(accumDtypes[0]));
      op.setAttr("quant_codesign.scale_source",
                 S("not_available_no_calibration"));
      op.setAttr("quant_codesign.zero_point_source",
                 S("not_available_no_calibration"));

      op.setAttr("quant_codesign.kernel_support.status",
                 S(best->kernel.status));
      if (!best->kernel.kernel_id.empty()) {
        op.setAttr("quant_codesign.kernel_support.kernel_id",
                   S(best->kernel.kernel_id));
        op.setAttr("quant_codesign.kernel_support.source",
                   S(best->kernel.source));
      }

      bool dispatchable = best->kernel.status == "runtime_dispatchable";
      op.setAttr("quant_codesign.materialization.required",
                 B(!dispatchable));
      op.setAttr("quant_codesign.materialization.status",
                 S(dispatchable
                       ? "not_required_kernel_consumes_quantized_weights"
                       : "deferred_missing_quant_params"));

      if (best->hasBytes) {
        op.setAttr("quant_codesign.est.weight_bytes_before",
                   I64(best->wBytesBefore));
        op.setAttr("quant_codesign.est.weight_bytes_after",
                   I64(best->wBytesAfter));
        op.setAttr("quant_codesign.est.boundary_bytes",
                   I64(best->boundaryBytes));
      }
      if (best->hasNanos) {
        op.setAttr("quant_codesign.est.total_cost_before_nanos",
                   I64(best->beforeNanos));
        op.setAttr("quant_codesign.est.total_cost_after_nanos",
                   I64(best->afterNanos));
        op.setAttr("quant_codesign.est.systems_benefit_nanos",
                   I64(best->benefitNanos));
      }
      if (dispatchable) {
        op.setAttr(
            "quant_codesign.est.excluded_terms",
            ArrayAttr::get(
                ctx, {S("inline_dequant_unpack_conversion_cost")}));
      }

      // F. Selection by explicit policy.
      if (policy == "planning_only") {
        finish("planning_only_not_selected");
        continue;
      }
      if (policy == "require_accuracy_evidence" &&
          !accuracyEvidenceSufficient) {
        reasons.push_back(
            S("no_calibrated_or_measured_accuracy_evidence_in_repository"));
        finish("deferred_missing_accuracy_evidence");
        continue;
      }
      if (policy == "require_dispatchable_kernel" && !dispatchable) {
        if (!registryDeclared) {
          reasons.push_back(S("no_runtime_kernel_registry_declared"));
          finish("deferred_no_runtime_kernel");
        } else {
          reasons.push_back(
              S("backend_capability_is_not_a_dispatchable_kernel"));
          finish("backend_supported_but_not_dispatchable");
        }
        continue;
      }
      // systems_cost_only and require_dispatchable_kernel both gate on the
      // static cost comparison.
      if (!best->hasNanos) {
        reasons.push_back(
            facts.usable()
                ? S("profile_peak_numbers_not_declared")
                : S("shapes_not_static"));
        finish("deferred_missing_cost_estimates");
        continue;
      }
      if (best->benefitNanos <= 0) {
        reasons.push_back(
            best->boundaryBytes > 0
                ? S("materialized_dequant_boundary_traffic_exceeds_weight_savings")
                : S("estimated_cost_not_reduced"));
        finish("rejected_no_systems_benefit");
        continue;
      }
      finish("selected");
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createQuantizationCoDesignPass() {
  return std::make_unique<QuantizationCoDesignPass>();
}

} // namespace mlir::hir
