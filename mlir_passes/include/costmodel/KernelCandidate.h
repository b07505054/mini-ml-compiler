#pragma once

// Cost Model Slice 1: Candidate Generation.
// (Slice 5 addition: reduction-tile (tileK) candidate-space expansion --
// see the TileShape doc comment and reductionTileChoicesForProblem()
// below; this is a documented, intentional candidate-schema change, not
// a silent reinterpretation of Slices 1-4's original full-K model.)
//
// This is a new, independent compiler track: an accelerator-style kernel
// candidate space for HIR conv2d, modeling precision, dataflow, PE array
// shape, and tile shape (spatial/channel AND, since Slice 5, reduction).
// It is deliberately unrelated to the existing "serving" track
// (mlir_passes/include/serving/ImplementationCandidate.h, ServingEnums.h,
// CandidateGenerationPass.cpp) which models whole-model artifact/Hailo-
// HEF/thread-schedule serving decisions -- nothing here reads or writes
// Hailo HEF metadata, runs a simulator, or touches any execution backend.
// This slice stops at candidate enumeration: no scoring, no pruning, no
// selection, no lowering.

#include "HIR/IR/HIROps.h"

#include <cstdint>
#include <string>
#include <vector>

namespace mlir::costmodel {

// Fixed enumeration order below is part of this module's determinism
// contract -- generateCandidates() iterates candidates in exactly this
// nested order: Precision outer, then Dataflow, then PEArrayShape, then
// base spatial/channel tile shape, then (Slice 5, innermost) reduction-
// tile choice -- so that repeated calls on the same input always produce
// the same list in the same order.
enum class Precision { INT8, FP16 };

enum class Dataflow { WeightStationary, OutputStationary, InputStationary };

// A square systolic PE array, e.g. 16x16 or 32x32.
struct PEArrayShape {
  int64_t rows = 0;
  int64_t cols = 0;

  bool operator==(const PEArrayShape &other) const {
    return rows == other.rows && cols == other.cols;
  }
};

// A candidate on-chip tile shape for the conv2d problem: how many output
// channels, output-row positions, and output-column positions are
// computed per tile, PLUS (Slice 5 addition) an explicit physical
// reduction-tile depth. See costmodel/CandidateLegality.h and
// costmodel/CandidateCostModel.h for the documented M/N/K mapping of
// these fields onto conv2d work and the memory-footprint/cost formulas
// built on top of them.
//
// `reductionDepth` (tileK): chosen name over the alternatives
// "reductionElements"/"tileK" for the struct field itself (tileK remains
// the short name used in prose/formulas throughout the documentation and
// code comments) because it reads unambiguously next to `height`/`width`/
// `outputChannels` as "how deep is one reduction tile", without needing
// the reader to already know the M/N/K convention. This is a Slice 5
// **intentional candidate-schema change**: TileShape previously had no
// reduction-tiling field at all (Slices 1-4 always used the problem's
// full reduction K for every candidate); every TileShape now carries an
// explicit, positive, physical `reductionDepth` -- 0 is never valid and
// is never interpreted as "use the full reduction". A candidate whose
// `reductionDepth` is <= 0, or exceeds the problem's full reduction K
// (see costmodel::extractConv2DProblemShape's inputChannels/kernelHeight/
// kernelWidth), is rejected by Slice 2's legality checker
// (LegalityReason::InvalidReductionTile), never silently reinterpreted.
struct TileShape {
  int64_t outputChannels = 0;
  int64_t height = 0;
  int64_t width = 0;
  int64_t reductionDepth = 0;

  bool operator==(const TileShape &other) const {
    return outputChannels == other.outputChannels && height == other.height &&
           width == other.width && reductionDepth == other.reductionDepth;
  }
};

// Static conv2d problem dimensions, extracted from a real HIR conv2d op's
// operand tensor shapes. Layout convention (matching this repository's
// existing NCHW/KCRS conventions elsewhere): input is [N, C, H, W],
// filter is [K, C, R, S] (K = output channels, R/S = kernel height/width).
//
// Slice 2 addition (additive, does not change Slice 1's input/filter-
// derived fields above): outputHeight/outputWidth are read from the same
// real Conv2dOp's statically-shaped *output* tensor type, used only by
// Slice 2's legality checker for the (documented, non-enforced) logical
// boundary-extent context -- see CandidateLegality.h. HIR_Conv2dOp's
// tablegen definition (HIROps.td) carries no typed stride/padding/
// dilation attributes (it is a metadata-only op), so per this task's
// explicit fallback instruction, the statically-shaped output tensor
// type is used directly as the source of truth instead of deriving
// output shape from conv arithmetic. Unlike the input/filter fields
// (which abort via llvm::report_fatal_error on a bad shape, Slice 1's
// original, unchanged behavior), a bad/dynamic output shape does NOT
// abort here: it is recorded gracefully via
// outputShapeIsStaticAndSupported = false (with outputHeight/outputWidth
// left at their -1 sentinel), so Slice 2 can fail closed with an
// observable, testable illegality reason (DynamicOrUnsupportedShape)
// instead of crashing the process.
struct Conv2DProblemShape {
  int64_t batch = 0;
  int64_t inputChannels = 0;
  int64_t inputHeight = 0;
  int64_t inputWidth = 0;
  int64_t outputChannels = 0;
  int64_t kernelHeight = 0;
  int64_t kernelWidth = 0;

  int64_t outputHeight = -1;
  int64_t outputWidth = -1;
  bool outputShapeIsStaticAndSupported = false;

  bool operator==(const Conv2DProblemShape &other) const {
    return batch == other.batch && inputChannels == other.inputChannels &&
           inputHeight == other.inputHeight &&
           inputWidth == other.inputWidth &&
           outputChannels == other.outputChannels &&
           kernelHeight == other.kernelHeight &&
           kernelWidth == other.kernelWidth &&
           outputHeight == other.outputHeight &&
           outputWidth == other.outputWidth &&
           outputShapeIsStaticAndSupported ==
               other.outputShapeIsStaticAndSupported;
  }
};

// One unscored candidate execution plan: a specific (precision, dataflow,
// PE array shape, tile shape) point in the design space for one conv2d
// problem. Deliberately carries no cost/score/latency field -- this
// slice enumerates only, it never evaluates.
struct KernelCandidate {
  std::string candidateId;
  Conv2DProblemShape problem;
  Precision precision = Precision::INT8;
  Dataflow dataflow = Dataflow::WeightStationary;
  PEArrayShape peArray;
  TileShape tile;

  bool operator==(const KernelCandidate &other) const {
    return candidateId == other.candidateId && problem == other.problem &&
           precision == other.precision && dataflow == other.dataflow &&
           peArray == other.peArray && tile == other.tile;
  }
};

llvm::StringRef toString(Precision precision);
llvm::StringRef toString(Dataflow dataflow);

// The fixed candidate axes. Exposed so tests can assert coverage/count
// directly against the same source of truth the generator uses, instead
// of hard-coding the expected values twice.
const std::vector<Precision> &allPrecisions();
const std::vector<Dataflow> &allDataflows();
const std::vector<PEArrayShape> &allPEArrayShapes();

// Base spatial/output-channel tile shapes (Slice 1's original 3-entry
// axis, unchanged values: {16,8,8}, {32,16,16}, {64,32,32}). Each entry's
// `reductionDepth` is left at its 0 placeholder -- these are NOT
// directly usable TileShapes (0 is never a valid reductionDepth, see the
// TileShape doc comment above). generateCandidates() always combines
// each of these with one of reductionTileChoicesForProblem()'s values
// before constructing a real candidate's TileShape.
const std::vector<TileShape> &allBaseSpatialTileShapes();

// Fixed reduction-tile SEED values, in this exact order: {16, 32, 64}.
// Parameterless and problem-independent (unlike
// reductionTileChoicesForProblem() below) -- exposed so tests can assert
// against the same seed list the generator uses.
const std::vector<int64_t> &reductionTileSeeds();

// The actual deterministic reduction-tile (tileK) choices for ONE
// conv2d problem's full reduction K (`problemK`, must be > 0):
// reductionTileSeeds() filtered to values <= problemK (Slice 5's
// documented policy: never generate a reduction tile larger than the
// full reduction unless padded-K candidates are intentionally
// supported -- they are not, yet), in seed order, followed by problemK
// itself with any duplicate removed (e.g. if problemK already equals a
// seed, the list is unchanged, not lengthened). For the reference
// problem (problemK=576) this is exactly {16, 32, 64, 576}.
std::vector<int64_t> reductionTileChoicesForProblem(int64_t problemK);

// Extracts a Conv2DProblemShape from a real mlir::hir::Conv2dOp, reading
// its input/filter operand tensor shapes directly (metadata-only op --
// no attributes are consulted). Both operands must be ranked tensors
// with fully static rank-4 shapes; this slice does not handle dynamic
// shapes or non-rank-4 operands (fail-closed: aborts via
// llvm::report_fatal_error rather than silently guessing).
Conv2DProblemShape extractConv2DProblemShape(mlir::hir::Conv2dOp op);

// Enumerates the full Cartesian product of Precision x Dataflow x
// PEArrayShape x (base spatial/channel tile) x (reduction-tile choice,
// Slice 5) for the given conv2d problem, in the fixed nested order
// documented above. `problemK` (inputChannels * kernelHeight *
// kernelWidth) is computed once per call via overflow-checked
// arithmetic; aborts via llvm::report_fatal_error on overflow, matching
// this module's existing fail-closed convention for a malformed real
// conv2d op (Slice 1's extractConv2DProblemShape already aborts before a
// candidate would ever be generated for such an op). Does not score,
// prune, sort, or filter candidates -- every combination is returned
// exactly once, in the same order on every call. For the reference
// problem (3 base tiles x 4 reduction choices = 12 TileShapes) this
// produces 2*3*2*12 = 144 candidates.
std::vector<KernelCandidate> generateCandidates(const Conv2DProblemShape &problem);

} // namespace mlir::costmodel
