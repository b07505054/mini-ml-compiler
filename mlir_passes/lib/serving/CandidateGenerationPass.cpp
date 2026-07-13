#include "FusionPasses.h"
#include "serving/ImplementationCandidate.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Operation.h"
#include "mlir/Pass/Pass.h"
#include <algorithm>
#include <set>
#include <string>
#include <vector>

namespace mlir::hir {
namespace {
#define GEN_PASS_DEF_CANDIDATEGENERATION
#include "FusionPasses.h.inc"

static constexpr StringLiteral kTruth =
    "candidate_generation_static_constraints_not_cost_evaluated";

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static std::string bcAttr(const std::string &backend, const std::string &field) {
  return "target.backend_capabilities." + backend + "." + field;
}

static std::vector<std::string> readStrArr(Operation *op, const std::string &name) {
  std::vector<std::string> r;
  if (auto a = op->getAttrOfType<ArrayAttr>(name))
    for (auto e : a)
      if (auto s = dyn_cast<StringAttr>(e))
        r.push_back(s.getValue().str());
  return r;
}

static std::string readStr(Operation *op, const std::string &name) {
  if (auto a = op->getAttrOfType<StringAttr>(name))
    return a.getValue().str();
  return "";
}

static bool readBool(Operation *op, const std::string &name) {
  if (auto a = op->getAttrOfType<BoolAttr>(name))
    return a.getValue();
  return false;
}

static std::optional<bool> readOptBool(Operation *op, const std::string &name) {
  if (auto a = op->getAttrOfType<BoolAttr>(name))
    return a.getValue();
  return std::nullopt;
}

// Merge lists: items from `preferred` first, then items from `rest` not already in result.
static std::vector<std::string> mergeDedup(const std::vector<std::string> &preferred,
                                           const std::vector<std::string> &acceptable,
                                           const std::vector<std::string> &rest = {}) {
  std::vector<std::string> result;
  std::set<std::string> seen;
  for (const auto &lists : {&preferred, &acceptable, &rest}) {
    for (const auto &s : *lists) {
      if (seen.insert(s).second)
        result.push_back(s);
    }
  }
  return result;
}

static bool isFloat(const std::string &d) {
  return d == "fp16" || d == "fp32" || d == "bf16" || d == "tf32" || d == "fp8";
}

static bool isQuantized(const std::string &d) {
  return d == "int8" || d == "int4" || d == "uint8";
}

// Strip "dialect." prefix from MLIR op names to get the base op name.
static std::string stripDialectPrefix(const std::string &opName) {
  auto pos = opName.rfind('.');
  return (pos != std::string::npos) ? opName.substr(pos + 1) : opName;
}

// Return true if any block arg or op result has a dynamic tensor dimension.
static bool hasDynamicShapes(func::FuncOp funcOp) {
  for (auto arg : funcOp.getArguments()) {
    if (auto tt = dyn_cast<RankedTensorType>(arg.getType()))
      if (!tt.hasStaticShape())
        return true;
    if (isa<UnrankedTensorType>(arg.getType()))
      return true;
  }
  if (funcOp.getBody().empty())
    return false;
  for (auto &block : funcOp.getBody()) {
    for (auto &op : block) {
      for (auto result : op.getResults()) {
        if (auto tt = dyn_cast<RankedTensorType>(result.getType()))
          if (!tt.hasStaticShape())
            return true;
        if (isa<UnrankedTensorType>(result.getType()))
          return true;
      }
    }
  }
  return false;
}

// Return the set of base op names present in the func body.
static std::set<std::string> collectOpNames(func::FuncOp funcOp) {
  std::set<std::string> names;
  if (funcOp.getBody().empty())
    return names;
  for (auto &block : funcOp.getBody())
    for (auto &op : block)
      names.insert(stripDialectPrefix(op.getName().getStringRef().str()));
  return names;
}

// Return true if any op in func body has boundary.layout_transform_required=true.
static bool anyLayoutTransformRequired(func::FuncOp funcOp) {
  if (funcOp.getBody().empty())
    return false;
  for (auto &block : funcOp.getBody())
    for (auto &op : block)
      if (readBool(&op, "boundary.layout_transform_required"))
        return true;
  return false;
}

// Select the appropriate quant_mode for a given candidate dtype.
static std::string selectQuantMode(const std::string &dtype,
                                   const std::vector<std::string> &supportedQuantModes,
                                   const std::string &requiredQuantMode) {
  if (!requiredQuantMode.empty())
    return requiredQuantMode;
  if (!isQuantized(dtype) && !isFloat(dtype))
    return "none";
  if (isFloat(dtype))
    return "none";
  // dtype is quantized — pick first applicable supported mode
  for (const auto &m : supportedQuantModes) {
    if ((dtype == "int8" || dtype == "uint8") &&
        (m == "static_int8" || m == "dynamic_int8" || m == "int8"))
      return m;
    if (dtype == "int4" && (m == "int4_weight_only" || m == "int4"))
      return m;
  }
  return supportedQuantModes.empty() ? "none" : supportedQuantModes[0];
}

// ---------------------------------------------------------------------------
// Per-op compiler candidate helpers (consumes alternative.* from
// AlternativeLoweringPlanningPass).
// ---------------------------------------------------------------------------

static std::string dictStr(DictionaryAttr dict, StringRef key) {
  if (!dict) return {};
  if (auto a = dict.get(key))
    if (auto s = dyn_cast<StringAttr>(a)) return s.getValue().str();
  return {};
}

static std::vector<std::string> dictStrs(DictionaryAttr dict, StringRef key) {
  std::vector<std::string> out;
  if (!dict) return out;
  if (auto a = dict.get(key))
    if (auto arr = dyn_cast<ArrayAttr>(a))
      for (auto e : arr)
        if (auto s = dyn_cast<StringAttr>(e)) out.push_back(s.getValue().str());
  return out;
}

// Build a per-op executable candidate dict.
static ImplementationCandidate buildOpCandidate(
    StringRef candidate_type,
    StringRef source_op,
    StringRef candidate_reason,
    const std::vector<std::string> &required_boundary_ops) {
  ImplementationCandidate candidate;
  candidate.providerId = "candidate_generation_pass";
  candidate.scopeKind = CandidateScopeKind::Operator;
  candidate.semanticTargetRef = source_op.str();
  candidate.implementationKind = candidate_type.str();
  candidate.candidateReason = candidate_reason.str();
  candidate.requiredBoundaryOps = required_boundary_ops;
  candidate.truthBoundary = kTruth.str();
  candidate.candidateId = makeFallbackCandidateId(candidate);
  candidate.feasibility.status =
      candidate.implementationKind == "unsupported"
          ? CandidateFeasibilityStatus::Unsupported
          : CandidateFeasibilityStatus::Feasible;
  candidate.feasibility.reason = candidate.candidateReason;
  return candidate;
}

static DictionaryAttr buildOpExecCandidate(
    MLIRContext *ctx,
    StringRef candidate_type,
    StringRef source_op,
    StringRef candidate_reason,
    const std::vector<std::string> &required_boundary_ops) {
  return encodeImplementationCandidate(
      ctx, buildOpCandidate(candidate_type, source_op, candidate_reason,
                            required_boundary_ops));
}

// Build a per-op rejected candidate dict (invalid alternative, not promoted).
static DictionaryAttr buildOpRejectedCandidate(
    MLIRContext *ctx,
    StringRef candidate_type,
    StringRef source_op,
    StringRef rejection_reason) {
  ImplementationCandidate candidate =
      buildOpCandidate(candidate_type, source_op, "rejected_alternative", {});
  candidate.feasibility.status = CandidateFeasibilityStatus::Rejected;
  candidate.feasibility.reason = rejection_reason.str();
  auto dict = encodeImplementationCandidate(ctx, candidate);
  SmallVector<NamedAttribute> entries(dict.begin(), dict.end());
  upsertCandidateAttr(entries, ctx, "rejection_reason",
                      StringAttr::get(ctx, rejection_reason));
  return DictionaryAttr::get(ctx, entries);
}

// ---------------------------------------------------------------------------
// Candidate struct
// ---------------------------------------------------------------------------

struct Candidate {
  std::string id;
  std::string backend;
  std::string backend_api;
  std::string dtype;
  std::string accumulation_dtype;
  std::string activation_layout;
  std::string weight_layout;
  std::string quant_mode;
  std::string fusion_pattern;
  std::string fallback_backend;
  bool requires_cast             = false;
  bool requires_dequant          = false;
  bool requires_layout_transform = false;
  std::string constraint_status;  // "pass" | "fail" | "unknown"
  std::vector<std::string> failed_constraints;
  std::string source_level;
  std::string truth_boundary;
};

static DictionaryAttr candidateToDict(MLIRContext *ctx, const Candidate &c) {
  SmallVector<NamedAttribute> fields;
  // Fields are added in alphabetical key order so the DictionaryAttr printer
  // emits them in a consistent, readable sequence.
  fields.push_back({StringAttr::get(ctx, "accumulation_dtype"),
                    StringAttr::get(ctx, c.accumulation_dtype)});
  fields.push_back({StringAttr::get(ctx, "activation_layout"),
                    StringAttr::get(ctx, c.activation_layout)});
  fields.push_back({StringAttr::get(ctx, "backend"),
                    StringAttr::get(ctx, c.backend)});
  fields.push_back({StringAttr::get(ctx, "backend_api"),
                    StringAttr::get(ctx, c.backend_api)});
  fields.push_back({StringAttr::get(ctx, "constraint_status"),
                    StringAttr::get(ctx, c.constraint_status)});
  std::string failedStr;
  for (size_t i = 0; i < c.failed_constraints.size(); ++i) {
    if (i > 0) failedStr += ",";
    failedStr += c.failed_constraints[i];
  }
  fields.push_back({StringAttr::get(ctx, "failed_constraints"),
                    StringAttr::get(ctx, failedStr)});
  fields.push_back({StringAttr::get(ctx, "fallback_backend"),
                    StringAttr::get(ctx, c.fallback_backend)});
  fields.push_back({StringAttr::get(ctx, "fusion_pattern"),
                    StringAttr::get(ctx, c.fusion_pattern)});
  fields.push_back({StringAttr::get(ctx, "id"),
                    StringAttr::get(ctx, c.id)});
  fields.push_back({StringAttr::get(ctx, "quant_mode"),
                    StringAttr::get(ctx, c.quant_mode)});
  fields.push_back({StringAttr::get(ctx, "requires_cast"),
                    BoolAttr::get(ctx, c.requires_cast)});
  fields.push_back({StringAttr::get(ctx, "requires_dequant"),
                    BoolAttr::get(ctx, c.requires_dequant)});
  fields.push_back({StringAttr::get(ctx, "requires_layout_transform"),
                    BoolAttr::get(ctx, c.requires_layout_transform)});
  fields.push_back({StringAttr::get(ctx, "source_level"),
                    StringAttr::get(ctx, c.source_level)});
  fields.push_back({StringAttr::get(ctx, "truth_boundary"),
                    StringAttr::get(ctx, c.truth_boundary)});
  fields.push_back({StringAttr::get(ctx, "dtype"),
                    StringAttr::get(ctx, c.dtype)});
  fields.push_back({StringAttr::get(ctx, "weight_layout"),
                    StringAttr::get(ctx, c.weight_layout)});
  return DictionaryAttr::get(ctx, fields);
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct CandidateGenerationPass
    : impl::CandidateGenerationBase<CandidateGenerationPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
  }

  void runOnOperation() override {
    func::FuncOp funcOp = getOperation();
    MLIRContext *ctx = funcOp.getContext();
    Operation *module = funcOp->getParentOp();

    // Read per-func context from prior passes.
    std::string effectiveDtype =
        readStr(funcOp.getOperation(), "representation.effective_dtype");
    bool layoutTransformActive = anyLayoutTransformRequired(funcOp);
    bool dynamicShapes         = hasDynamicShapes(funcOp);
    bool weightsAreConstant    = true; // optimistic default
    if (auto a = funcOp->getAttrOfType<BoolAttr>("representation.weights_are_constant"))
      weightsAreConstant = a.getValue();
    // Explicit false overrides the optimistic default.
    bool weightsConstantDeclared =
        funcOp->hasAttr("representation.weights_are_constant");

    std::set<std::string> presentOpNames = collectOpNames(funcOp);

    // Read backend names from module.
    std::vector<std::string> backendNames =
        readStrArr(module, "target.backend_capability_names");

    std::vector<Candidate> candidates;

    for (const auto &bn : backendNames) {
      // --- Read all capability fields for this backend ---
      std::string backendApi    = readStr(module, bcAttr(bn, "backend_api"));
      std::string sourceLevel   = readStr(module, bcAttr(bn, "source_level"));
      std::string fallbackBe    = readStr(module, bcAttr(bn, "fallback_backend"));

      auto supportedDtypes      = readStrArr(module, bcAttr(bn, "supported_dtypes"));
      auto preferredDtypes      = readStrArr(module, bcAttr(bn, "preferred_dtypes"));
      auto acceptableDtypes     = readStrArr(module, bcAttr(bn, "acceptable_dtypes"));
      auto accumDtypes          = readStrArr(module, bcAttr(bn, "accumulation_dtypes"));
      std::string accumDtype    = accumDtypes.empty() ? "fp32" : accumDtypes[0];

      auto prefActLayouts       = readStrArr(module, bcAttr(bn, "preferred_activation_layouts"));
      auto accActLayouts        = readStrArr(module, bcAttr(bn, "acceptable_activation_layouts"));
      auto prefWtLayouts        = readStrArr(module, bcAttr(bn, "preferred_weight_layouts"));
      std::string primaryWt     = prefWtLayouts.empty() ? "unknown" : prefWtLayouts[0];

      auto supportedQuantModes  = readStrArr(module, bcAttr(bn, "supported_quant_modes"));
      auto prefFusionPatterns   = readStrArr(module, bcAttr(bn, "preferred_fusion_patterns"));
      auto suppFusionPatterns   = readStrArr(module, bcAttr(bn, "supports_fusion_patterns"));
      // Fusion: prefer preferred_fusion_patterns, else first from supports_fusion_patterns.
      std::string primaryFusion = "none";
      if (!prefFusionPatterns.empty())      primaryFusion = prefFusionPatterns[0];
      else if (!suppFusionPatterns.empty()) primaryFusion = suppFusionPatterns[0];

      // Level 2 constraint fields.
      auto unsupportedOps           = readStrArr(module, bcAttr(bn, "unsupported_ops"));
      std::set<std::string> unsupSet(unsupportedOps.begin(), unsupportedOps.end());
      std::string reqActQuantMode   = readStr(module, bcAttr(bn, "required_activation_quant_mode"));
      std::string reqWtQuantMode    = readStr(module, bcAttr(bn, "required_weight_quant_mode"));
      auto reqStaticShape           = readOptBool(module, bcAttr(bn, "requires_static_shape"));
      auto reqConstantWeight        = readOptBool(module, bcAttr(bn, "requires_constant_weight"));

      // Check whether any present op name is in this backend's unsupported set.
      bool hasUnsupportedOp = false;
      std::string firstUnsup;
      for (const auto &opName : presentOpNames) {
        if (unsupSet.count(opName)) {
          hasUnsupportedOp = true;
          firstUnsup = opName;
          break;
        }
      }

      // Ordered dtype list: preferred → acceptable → supported (deduped).
      auto allDtypes = mergeDedup(preferredDtypes, acceptableDtypes, supportedDtypes);
      if (allDtypes.empty())
        allDtypes = supportedDtypes;

      // Ordered layout list: preferred → acceptable (deduped).
      auto allLayouts = mergeDedup(prefActLayouts, accActLayouts);
      if (allLayouts.empty())
        allLayouts = {"unknown"};

      // Generate one candidate per (dtype, activation_layout) combination.
      for (const auto &dtype : allDtypes) {
        for (const auto &layout : allLayouts) {
          Candidate cand;
          cand.id = bn + "_" + dtype + "_" + layout;
          cand.backend              = bn;
          cand.backend_api          = backendApi;
          cand.dtype                = dtype;
          cand.accumulation_dtype   = accumDtype;
          cand.activation_layout    = layout;
          cand.weight_layout        = primaryWt;
          cand.fallback_backend     = fallbackBe;
          cand.source_level         = sourceLevel;
          cand.truth_boundary       = kTruth.str();

          std::string quantMode = selectQuantMode(dtype, supportedQuantModes, reqWtQuantMode);
          cand.quant_mode       = quantMode;
          cand.fusion_pattern   = primaryFusion;

          // requires_cast: dtype → effectiveDtype, both float, different.
          cand.requires_cast = !effectiveDtype.empty()
              && isFloat(dtype) && isFloat(effectiveDtype)
              && dtype != effectiveDtype;

          // requires_dequant: plan was quantized, candidate dtype is float.
          cand.requires_dequant = isQuantized(effectiveDtype) && isFloat(dtype);

          // requires_layout_transform: any existing boundary said so, OR
          // this candidate uses a non-preferred layout.
          bool nonPrefLayout = (!prefActLayouts.empty() && layout != prefActLayouts[0]);
          cand.requires_layout_transform = layoutTransformActive || nonPrefLayout;

          // --- Constraint checking (mark, do NOT discard) ---
          std::vector<std::string> fails;

          if (reqStaticShape.has_value() && *reqStaticShape && dynamicShapes)
            fails.push_back("requires_static_shape");

          if (reqConstantWeight.has_value() && *reqConstantWeight) {
            if (weightsConstantDeclared && !weightsAreConstant)
              fails.push_back("requires_constant_weight");
          }

          if (hasUnsupportedOp)
            fails.push_back("unsupported_op:" + firstUnsup);

          if (!reqActQuantMode.empty() && quantMode != reqActQuantMode)
            fails.push_back("required_activation_quant_mode:" + reqActQuantMode);

          if (!reqWtQuantMode.empty() && quantMode != reqWtQuantMode &&
              quantMode != "none")
            fails.push_back("required_weight_quant_mode:" + reqWtQuantMode);

          cand.failed_constraints = fails;
          cand.constraint_status  = fails.empty() ? "pass" : "fail";

          candidates.push_back(std::move(cand));
        }
      }
    }

    // Attach candidates to the FuncOp.
    SmallVector<Attribute> dicts;
    dicts.reserve(candidates.size());
    for (const auto &c : candidates)
      dicts.push_back(candidateToDict(ctx, c));

    MLIRContext *mctx = funcOp.getContext();
    funcOp->setAttr("candidates", ArrayAttr::get(mctx, dicts));
    funcOp->setAttr("candidates.count",
                    IntegerAttr::get(IntegerType::get(mctx, 64),
                                     static_cast<int64_t>(candidates.size())));
    funcOp->setAttr("candidates.truth_boundary",
                    StringAttr::get(mctx, kTruth));

    // -------------------------------------------------------------------
    // Per-op compiler candidates from alternative.* attrs.
    // Consumes AlternativeLoweringPlanningPass output to produce per-op
    // compiler.candidates and compiler.rejected_candidates.
    // -------------------------------------------------------------------
    if (funcOp.getBody().empty()) return;
    for (Operation &op : funcOp.getBody().front().without_terminator()) {
      auto lsAttr = op.getAttrOfType<StringAttr>("kernel.lowering_status");
      if (!lsAttr) continue;

      std::string ls = lsAttr.getValue().str();
      bool kernelExists = false;
      if (auto a = op.getAttrOfType<BoolAttr>("kernel.exists"))
        kernelExists = a.getValue();

      // Op short name (everything after last '.').
      llvm::StringRef fullName = op.getName().getStringRef();
      std::string opType = fullName.str();
      if (auto dot = fullName.rfind('.'); dot != llvm::StringRef::npos)
        opType = fullName.substr(dot + 1).str();

      // Boundary ops already required by the lowering decision (direct path).
      std::vector<std::string> directBoundaryOps;
      if (readBool(&op, "lowering.requires_cast"))
        directBoundaryOps.push_back("cast");
      if (readBool(&op, "lowering.requires_dequant"))
        directBoundaryOps.push_back("dequant");
      if (readBool(&op, "lowering.requires_layout_transform"))
        directBoundaryOps.push_back("layout_transform");

      SmallVector<Attribute> execCandidates;
      SmallVector<Attribute> rejectedCandidates;
      bool hasDirectLower    = false;
      bool hasValidNonFallback = false;

      // Rule 1: emit direct_lower when the kernel is an exact match.
      if (kernelExists && ls == "lowerable") {
        hasDirectLower = true;
        execCandidates.push_back(buildOpExecCandidate(
            ctx, "direct_lower", opType, "kernel_exact_match",
            directBoundaryOps));
      }

      // Rule 2+3: process alternative.candidates.
      auto altArr = op.getAttrOfType<ArrayAttr>("alternative.candidates");
      if (altArr) {
        for (auto elem : altArr) {
          auto altDict = dyn_cast<DictionaryAttr>(elem);
          if (!altDict) continue;
          std::string altType   = dictStr(altDict, "alternative_type");
          std::string altStatus = dictStr(altDict, "validation_status");
          std::string altFails  = dictStr(altDict, "validation_failures");
          auto altBoundary      = dictStrs(altDict, "required_boundary_ops");

          if (altStatus == "valid") {
            if (altType == "backend_fallback") {
              // Deferred: only emitted by Rule 4.
            } else {
              hasValidNonFallback = true;
              execCandidates.push_back(buildOpExecCandidate(
                  ctx, altType, opType, "from_alternative_lowering",
                  altBoundary));
            }
          } else {
            // Invalid: record in rejected_candidates, not executable.
            rejectedCandidates.push_back(buildOpRejectedCandidate(
                ctx, altType, opType, altFails));
          }
        }
      }

      // Rule 4: backend_fallback only when there is no direct_lower and no
      //         valid non-fallback alternative.
      if (!hasDirectLower && !hasValidNonFallback && altArr) {
        for (auto elem : altArr) {
          auto altDict = dyn_cast<DictionaryAttr>(elem);
          if (!altDict) continue;
          if (dictStr(altDict, "alternative_type") == "backend_fallback" &&
              dictStr(altDict, "validation_status") == "valid") {
            execCandidates.push_back(buildOpExecCandidate(
                ctx, "backend_fallback", opType, "last_resort_fallback", {}));
            break;
          }
        }
      }

      // Emit unsupported sentinel when no viable path was found.
      if (execCandidates.empty()) {
        execCandidates.push_back(buildOpExecCandidate(
            ctx, "unsupported", opType, "no_viable_lowering_path", {}));
      }

      op.setAttr("compiler.candidates",
                 ArrayAttr::get(ctx, execCandidates));
      op.setAttr("compiler.candidates.count",
                 IntegerAttr::get(IntegerType::get(ctx, 64),
                                  static_cast<int64_t>(execCandidates.size())));
      op.setAttr("compiler.candidates.truth_boundary",
                 StringAttr::get(ctx, kTruth));
      op.setAttr("compiler.rejected_candidates",
                 ArrayAttr::get(ctx, rejectedCandidates));
    }
  }
};

} // namespace

std::unique_ptr<::mlir::Pass> createCandidateGenerationPass() {
  return std::make_unique<CandidateGenerationPass>();
}

} // namespace mlir::hir
