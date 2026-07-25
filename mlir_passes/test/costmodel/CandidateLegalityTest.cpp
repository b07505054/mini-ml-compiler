// Cost Model Slice 2: Candidate Legality Checking unit tests.
//
// Uses the real Slice 1 candidate-generation path (a real
// mlir::hir::Conv2dOp -> extractConv2DProblemShape -> generateCandidates)
// and classifies every candidate with checkCandidateLegality() against
// the synthetic generic_npu_v1 reference target (and a handful of
// dedicated, narrower test-only targets used to isolate one rejection
// category at a time). See CandidateLegality.h for the documented M/N/K
// mapping, byte-footprint formulas, and rejection-order contract this
// test pins down with exact assertions.

#include "costmodel/CandidateLegality.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <cstdio>
#include <optional>

using namespace mlir;
using namespace mlir::costmodel;

namespace {

mlir::hir::Conv2dOp buildConv2dOp(MLIRContext &ctx, Block &block,
                                   std::optional<int64_t> dynamicOutputHeight =
                                       std::nullopt) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  auto elementType = builder.getIntegerType(8);
  auto inputType = RankedTensorType::get({1, 64, 56, 56}, elementType);
  auto filterType = RankedTensorType::get({128, 64, 3, 3}, elementType);
  int64_t outH = dynamicOutputHeight.has_value() ? ShapedType::kDynamic : 54;
  auto outputType = RankedTensorType::get({1, 128, outH, 54}, elementType);

  block.addArgument(inputType, loc);
  block.addArgument(filterType, loc);
  builder.setInsertionPointToStart(&block);

  return builder.create<mlir::hir::Conv2dOp>(loc, outputType,
                                              block.getArgument(0),
                                              block.getArgument(1));
}

const KernelCandidate &findCandidate(const std::vector<KernelCandidate> &candidates,
                                      Precision precision, Dataflow dataflow,
                                      const PEArrayShape &pe,
                                      const TileShape &tile) {
  for (const auto &c : candidates) {
    if (c.precision == precision && c.dataflow == dataflow &&
        c.peArray == pe && c.tile == tile)
      return c;
  }
  std::fprintf(stderr, "findCandidate: no matching candidate found\n");
  std::abort();
}

NPUTargetConfig makeGenerousTarget() {
  // A target that supports everything and never constrains capacity or
  // alignment -- used to isolate exactly one rejection category at a
  // time in the tests below.
  NPUTargetConfig t;
  t.targetId = "test_generous_target";
  t.scratchpadBytes = 1ull << 40;
  t.weightBufferBytes = 1ull << 40;
  t.inputBufferBytes = 1ull << 40;
  t.outputBufferBytes = 1ull << 40;
  t.dmaAlignmentBytes = 1;
  t.channelAlignmentInt8 = 1;
  t.channelAlignmentFp16 = 1;
  t.supportsInt8 = true;
  t.supportsFp16 = true;
  t.supportedDataflows = {Dataflow::WeightStationary, Dataflow::OutputStationary,
                           Dataflow::InputStationary};
  t.supportedPEArrays = {PEArrayShape{16, 16}, PEArrayShape{32, 32}};
  return t;
}

} // namespace

int main() {
  std::puts("=== CandidateLegalityTest ===");

  // Slice 6 policy: sentinel proving assert() is genuinely active under
  // this Release (-DNDEBUG) build -- see this target's -UNDEBUG option in
  // CMakeLists.txt. If that policy were ever lost, assert() below would
  // compile to nothing and this explicit (non-assert-gated) check would
  // abort with a clear message instead of a false PASS.
  {
    bool assertionsActive = false;
    assert((assertionsActive = true, true));
    if (!assertionsActive) {
      std::fprintf(stderr, "FATAL: assert() is compiled out in this test binary (NDEBUG?)\n");
      std::abort();
    }
  }

  MLIRContext ctx;
  ctx.getOrLoadDialect<mlir::hir::HIRDialect>();

  Block block;
  mlir::hir::Conv2dOp conv = buildConv2dOp(ctx, block);
  Conv2DProblemShape problem = extractConv2DProblemShape(conv);

  // ---- (1) real Conv2dOp now generates exactly 144 candidates (Slice 5:
  // 3 base tiles x 4 reduction-tile choices = 12 TileShapes) ----
  std::vector<KernelCandidate> candidates = generateCandidates(problem);
  assert(candidates.size() == 144);
  assert(problem.outputShapeIsStaticAndSupported);
  assert(problem.outputHeight == 54 && problem.outputWidth == 54);
  std::puts("  [ok] real Conv2dOp now generates exactly 144 candidates (Slice 5 reduction tiling)");

  const NPUTargetConfig &generic = genericNPUv1Target();
  assert(generic.targetId == "generic_npu_v1");
  assert(generic.scratchpadBytes == 256 * 1024);
  assert(generic.weightBufferBytes == 128 * 1024);
  assert(generic.inputBufferBytes == 64 * 1024);
  assert(generic.outputBufferBytes == 64 * 1024);

  // ---- (2)+(4) exactly one result per candidate, in generation order ----
  std::vector<CandidateLegalityResult> results =
      checkCandidateLegality(candidates, generic);
  assert(results.size() == candidates.size());
  for (std::size_t i = 0; i < candidates.size(); ++i) {
    CandidateLegalityResult single = checkCandidateLegality(candidates[i], generic);
    assert(single == results[i]); // batch form matches single-candidate form,
                                   // in the same order.
  }
  std::puts("  [ok] exactly one legality result per candidate, order matches generation order");

  // ---- (3) determinism: repeated calls are field-identical ----
  std::vector<CandidateLegalityResult> resultsAgain =
      checkCandidateLegality(candidates, generic);
  assert(resultsAgain == results);
  std::puts("  [ok] repeated checkCandidateLegality() calls are field-identical");

  // ---- (12) legality checking never mutates the candidate ----
  {
    KernelCandidate before = candidates[0];
    CandidateLegalityResult r = checkCandidateLegality(candidates[0], generic);
    (void)r;
    assert(candidates[0] == before);
    assert(candidates[0].candidateId == before.candidateId);
  }
  std::puts("  [ok] checkCandidateLegality never mutates the candidate");

  // ---- exact byte-footprint math against the documented M/N/K model ----
  // problemK = inputChannels * kernelHeight * kernelWidth = 64*3*3 = 576.
  // These spot-check tiles use tileK=576 (the full-reduction choice) so
  // their byte footprints match exactly what Slices 2-4 originally
  // computed before Slice 5 introduced reduction tiling.
  const int64_t problemK = 64 * 3 * 3;
  assert(problemK == 576);

  const TileShape tile1{16, 8, 8, 576};   // M=64,  N=16, tileK=576 (full)
  const TileShape tile3{64, 32, 32, 576}; // M=1024,N=64, tileK=576 (full)
  const PEArrayShape pe16{16, 16};

  const KernelCandidate &t1i8 =
      findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
  CandidateLegalityResult t1i8Result = checkCandidateLegality(t1i8, generic);
  // M=64,K=576,N=16: input=64*576*1=36864; weight=576*16*1=9216;
  // output=64*16*4=4096; total=50176.
  assert(t1i8Result.inputTileBytes == 36864);
  assert(t1i8Result.weightTileBytes == 9216);
  assert(t1i8Result.outputTileBytes == 4096);
  assert(t1i8Result.totalScratchpadBytes == 50176);
  // ---- (5) at least one candidate is legal for generic_npu_v1 ----
  assert(t1i8Result.isLegal);
  assert(t1i8Result.reasons.empty());
  std::puts("  [ok] Tile1+INT8 candidate is legal for generic_npu_v1 with exact expected byte footprints");

  // ---- (8) FP16 input/weight bytes are exactly 2x INT8; accumulator is
  // precision-independent (documented: INT32 vs FP32, both 4 bytes) ----
  const KernelCandidate &t1fp16 =
      findCandidate(candidates, Precision::FP16, Dataflow::WeightStationary, pe16, tile1);
  CandidateLegalityResult t1fp16Result = checkCandidateLegality(t1fp16, generic);
  assert(t1fp16Result.inputTileBytes == 2 * t1i8Result.inputTileBytes);
  assert(t1fp16Result.weightTileBytes == 2 * t1i8Result.weightTileBytes);
  assert(t1fp16Result.outputTileBytes == t1i8Result.outputTileBytes); // accumulator unchanged
  assert(t1fp16Result.inputTileBytes == 73728);
  assert(t1fp16Result.weightTileBytes == 18432);
  assert(t1fp16Result.outputTileBytes == 4096);
  // Tile1 FP16 exceeds the 64 KiB input buffer (73728 > 65536) -- single reason.
  assert(!t1fp16Result.isLegal);
  assert(t1fp16Result.reasons.size() == 1);
  assert(t1fp16Result.reasons[0] == LegalityReason::InputTileExceedsBuffer);
  std::puts("  [ok] FP16 input/weight bytes are exactly 2x INT8; accumulator bytes unchanged; single-reason illegal case confirmed");

  // ---- (7) multi-reason illegal candidate, deterministic order ----
  const KernelCandidate &t3i8 =
      findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile3);
  CandidateLegalityResult t3i8Result = checkCandidateLegality(t3i8, generic);
  // M=1024,K=576,N=64: input=1024*576=589824; weight=576*64=36864;
  // output=1024*64*4=262144; total=888832.
  assert(t3i8Result.inputTileBytes == 589824);
  assert(t3i8Result.weightTileBytes == 36864);
  assert(t3i8Result.outputTileBytes == 262144);
  assert(t3i8Result.totalScratchpadBytes == 888832);
  assert(!t3i8Result.isLegal);
  assert(t3i8Result.reasons.size() == 3);
  assert(t3i8Result.reasons[0] == LegalityReason::InputTileExceedsBuffer);
  assert(t3i8Result.reasons[1] == LegalityReason::OutputTileExceedsBuffer);
  assert(t3i8Result.reasons[2] == LegalityReason::TotalScratchpadExceeded);
  std::puts("  [ok] multi-reason illegal candidate has all 3 expected reasons in deterministic order");

  // ---- full legal/illegal histogram over all 144 candidates for
  // generic_npu_v1 (Slice 5: hand-derived -- see the Slice 5 final-
  // evidence report for the full per-base-tile/per-K-choice derivation).
  // Reduction choices {16,32,64,576} all evenly divide problemK=576, so
  // there is zero K-padding anywhere in this reference problem; the only
  // things that change per K choice are the byte footprints themselves
  // (smaller tileK -> smaller input/weight tile footprint). Legality is
  // independent of dataflow/PE array (generic_npu_v1 supports all of
  // them), so it reduces to 3 base tiles x 4 K choices x 2 precisions =
  // 24 distinct legality profiles, each replicated 6x (3 dataflows x 2 PE
  // arrays): B1(16,8,8) is legal at all 8 profiles except FP16+K576
  // (illegal, 1 reason); B2(32,16,16) is legal at K in {16,32,64} for
  // both precisions (illegal at K576, both precisions); B3(64,32,32) is
  // illegal at every profile (its output tile alone, 1024*64*4=262144
  // bytes, already exceeds the 64 KiB output buffer regardless of K). ----
  std::size_t legalCount = 0, illegalCount = 0;
  std::size_t histogram[14] = {0};
  for (const auto &r : results) {
    if (r.isLegal)
      ++legalCount;
    else
      ++illegalCount;
    for (LegalityReason reason : r.reasons)
      ++histogram[static_cast<int>(reason)];
  }
  assert(legalCount == 78);
  assert(illegalCount == 66);
  assert(histogram[static_cast<int>(LegalityReason::InputTileExceedsBuffer)] == 36);
  assert(histogram[static_cast<int>(LegalityReason::OutputTileExceedsBuffer)] == 48);
  assert(histogram[static_cast<int>(LegalityReason::TotalScratchpadExceeded)] == 54);
  assert(histogram[static_cast<int>(LegalityReason::WeightTileExceedsBuffer)] == 0);
  assert(histogram[static_cast<int>(LegalityReason::InvalidReductionTile)] == 0);
  assert(histogram[static_cast<int>(LegalityReason::ReductionAlignmentViolation)] == 0);
  assert(histogram[static_cast<int>(LegalityReason::ChannelAlignmentViolation)] == 0);
  std::printf("  [ok] full histogram over 144 candidates: legal=%zu illegal=%zu\n",
              legalCount, illegalCount);

  LegalityPartition partition = partitionByLegality(results);
  assert(partition.legalIndices.size() == 78);
  assert(partition.illegalIndices.size() == 66);
  for (std::size_t i = 1; i < partition.legalIndices.size(); ++i)
    assert(partition.legalIndices[i - 1] < partition.legalIndices[i]);
  for (std::size_t i = 1; i < partition.illegalIndices.size(); ++i)
    assert(partition.illegalIndices[i - 1] < partition.illegalIndices[i]);
  std::puts("  [ok] partitionByLegality preserves original relative order");

  // ---- (6) isolated single-reason tests for each major rejection category ----

  // unsupported precision
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.supportsFp16 = false;
    CandidateLegalityResult r = checkCandidateLegality(t1fp16, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::UnsupportedPrecision);
  }
  // unsupported dataflow
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.supportedDataflows = {Dataflow::WeightStationary, Dataflow::OutputStationary};
    const KernelCandidate &isCandidate =
        findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe16, tile1);
    CandidateLegalityResult r = checkCandidateLegality(isCandidate, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::UnsupportedDataflow);
  }
  // unsupported PE array
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.supportedPEArrays = {PEArrayShape{16, 16}};
    const KernelCandidate &pe32Candidate = findCandidate(
        candidates, Precision::INT8, Dataflow::WeightStationary, PEArrayShape{32, 32}, tile1);
    CandidateLegalityResult r = checkCandidateLegality(pe32Candidate, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::UnsupportedPEArray);
  }
  // input-buffer capacity (isolated)
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.inputBufferBytes = 1000;
    CandidateLegalityResult r = checkCandidateLegality(t1i8, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::InputTileExceedsBuffer);
  }
  // weight-buffer capacity (isolated)
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.weightBufferBytes = 1000; // < 9216
    CandidateLegalityResult r = checkCandidateLegality(t1i8, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::WeightTileExceedsBuffer);
  }
  // output-buffer capacity (isolated)
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.outputBufferBytes = 100; // < 4096
    CandidateLegalityResult r = checkCandidateLegality(t1i8, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::OutputTileExceedsBuffer);
  }
  // total scratchpad capacity (isolated: each individual buffer fits, sum doesn't)
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.inputBufferBytes = 100000;
    t.weightBufferBytes = 100000;
    t.outputBufferBytes = 100000;
    t.scratchpadBytes = 40000; // < 50176 total, but > each individual tile
    CandidateLegalityResult r = checkCandidateLegality(t1i8, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::TotalScratchpadExceeded);
  }
  // channel (N) alignment violation (isolated) -- use tileK=7 (divisible
  // by 7, so ONLY N=16 violates, not tileK) to isolate this from the new
  // ReductionAlignmentViolation reason below.
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.channelAlignmentInt8 = 7; // 16 % 7 != 0
    KernelCandidate withK7 = t1i8;
    withK7.tile.reductionDepth = 7; // 7 % 7 == 0 -- tileK itself stays aligned
    CandidateLegalityResult r = checkCandidateLegality(withK7, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::ChannelAlignmentViolation);
  }
  // reduction (tileK) alignment violation (isolated, Slice 5) -- use
  // channelAlignmentInt8=8 (16 % 8 == 0, N stays aligned) with tileK=7
  // (7 % 8 != 0) to isolate this from ChannelAlignmentViolation.
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.channelAlignmentInt8 = 8;
    KernelCandidate withK7 = t1i8;
    withK7.tile.reductionDepth = 7;
    CandidateLegalityResult r = checkCandidateLegality(withK7, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::ReductionAlignmentViolation);
  }
  // DMA alignment violation (isolated)
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.dmaAlignmentBytes = 1000000; // none of the tile byte sizes are multiples of this
    CandidateLegalityResult r = checkCandidateLegality(t1i8, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::DMAAlignmentViolation);
  }
  // invalid tile shape (isolated, hand-constructed candidate)
  {
    NPUTargetConfig t = makeGenerousTarget();
    KernelCandidate bad = t1i8;
    bad.tile.height = 0;
    CandidateLegalityResult r = checkCandidateLegality(bad, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::InvalidTileShape);
  }
  // invalid reduction tile (isolated, Slice 5): tileK <= 0
  {
    NPUTargetConfig t = makeGenerousTarget();
    KernelCandidate bad = t1i8;
    bad.tile.reductionDepth = 0;
    CandidateLegalityResult r = checkCandidateLegality(bad, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::InvalidReductionTile);
  }
  // invalid reduction tile (isolated, Slice 5): tileK > problemK
  {
    NPUTargetConfig t = makeGenerousTarget();
    KernelCandidate bad = t1i8;
    bad.tile.reductionDepth = problemK + 1;
    CandidateLegalityResult r = checkCandidateLegality(bad, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::InvalidReductionTile);
  }
  std::puts("  [ok] all 11 required rejection categories isolated and verified individually");

  // ---- (9) dataflow-specific legality: constraining "the relevant"
  // buffer for a given dataflow produces that dataflow's expected reason ----
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.inputBufferBytes = 1000;
    const KernelCandidate &isCandidate =
        findCandidate(candidates, Precision::INT8, Dataflow::InputStationary, pe16, tile1);
    CandidateLegalityResult r = checkCandidateLegality(isCandidate, t);
    assert(!r.isLegal && r.reasons.size() == 1 &&
           r.reasons[0] == LegalityReason::InputTileExceedsBuffer);
  }
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.weightBufferBytes = 1000;
    const KernelCandidate &wsCandidate =
        findCandidate(candidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    CandidateLegalityResult r = checkCandidateLegality(wsCandidate, t);
    assert(!r.isLegal && r.reasons.size() == 1 &&
           r.reasons[0] == LegalityReason::WeightTileExceedsBuffer);
  }
  {
    NPUTargetConfig t = makeGenerousTarget();
    t.outputBufferBytes = 100;
    const KernelCandidate &osCandidate =
        findCandidate(candidates, Precision::INT8, Dataflow::OutputStationary, pe16, tile1);
    CandidateLegalityResult r = checkCandidateLegality(osCandidate, t);
    assert(!r.isLegal && r.reasons.size() == 1 &&
           r.reasons[0] == LegalityReason::OutputTileExceedsBuffer);
  }
  std::puts("  [ok] dataflow-specific legality: WS/OS/IS each flag their own relevant buffer constraint");

  // ---- (10) dynamic/unsupported Conv2D output shape fails closed ----
  {
    Block dynBlock;
    mlir::hir::Conv2dOp dynConv = buildConv2dOp(ctx, dynBlock, /*dynamicOutputHeight=*/0);
    Conv2DProblemShape dynProblem = extractConv2DProblemShape(dynConv);
    assert(!dynProblem.outputShapeIsStaticAndSupported);
    assert(dynProblem.outputHeight == -1 && dynProblem.outputWidth == -1);

    std::vector<KernelCandidate> dynCandidates = generateCandidates(dynProblem);
    assert(dynCandidates.size() == 144); // generation itself is unaffected

    const KernelCandidate &dynT1i8 = findCandidate(
        dynCandidates, Precision::INT8, Dataflow::WeightStationary, pe16, tile1);
    CandidateLegalityResult r = checkCandidateLegality(dynT1i8, generic);
    // This exact candidate was fully legal for the static-shape problem;
    // with a dynamic output shape it must fail closed, in isolation.
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::DynamicOrUnsupportedShape);

    // Every candidate for the dynamic-shape problem carries this reason.
    std::vector<CandidateLegalityResult> dynResults =
        checkCandidateLegality(dynCandidates, generic);
    for (const auto &dr : dynResults) {
      bool hasReason = false;
      for (LegalityReason reason : dr.reasons)
        hasReason |= (reason == LegalityReason::DynamicOrUnsupportedShape);
      assert(hasReason);
    }
  }
  std::puts("  [ok] dynamic/unsupported Conv2D output shape fails closed for every candidate");

  // ---- (11) overflow-safe byte arithmetic does not wrap ----
  // (a) problemK itself overflows -> InvalidReductionTile (evaluated
  // before any byte-sum arithmetic is attempted).
  {
    KernelCandidate synth;
    synth.candidateId = "synthetic:problemk_overflow_probe";
    synth.problem.batch = 1;
    synth.problem.inputChannels = int64_t(1) << 40;
    synth.problem.inputHeight = 1;
    synth.problem.inputWidth = 1;
    synth.problem.outputChannels = 1;
    synth.problem.kernelHeight = int64_t(1) << 40; // problemK overflows int64
    synth.problem.kernelWidth = 1;
    synth.problem.outputHeight = 1;
    synth.problem.outputWidth = 1;
    synth.problem.outputShapeIsStaticAndSupported = true;
    synth.precision = Precision::INT8;
    synth.dataflow = Dataflow::WeightStationary;
    synth.peArray = PEArrayShape{16, 16};
    synth.tile = TileShape{16, 8, 8, 576};

    NPUTargetConfig t = makeGenerousTarget();
    CandidateLegalityResult r = checkCandidateLegality(synth, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::InvalidReductionTile);
    assert(r.inputTileBytes == 0 && r.weightTileBytes == 0 && r.outputTileBytes == 0);
  }
  // (b) problemK is fine, but the byte-sum arithmetic itself overflows
  // (huge tileM) -> ByteFootprintOverflow, byte fields explicitly 0.
  {
    KernelCandidate synth;
    synth.candidateId = "synthetic:bytesum_overflow_probe";
    synth.problem.batch = 1;
    synth.problem.inputChannels = 576;
    synth.problem.inputHeight = 1;
    synth.problem.inputWidth = 1;
    synth.problem.outputChannels = 1;
    synth.problem.kernelHeight = 1;
    synth.problem.kernelWidth = 1;
    synth.problem.outputHeight = 1;
    synth.problem.outputWidth = 1;
    synth.problem.outputShapeIsStaticAndSupported = true; // problemK = 576
    synth.precision = Precision::INT8;
    synth.dataflow = Dataflow::WeightStationary;
    synth.peArray = PEArrayShape{16, 16};
    // tileK=576 (valid, <= problemK); tileM = height*width overflows.
    synth.tile = TileShape{16, int64_t(1) << 40, int64_t(1) << 40, 576};

    NPUTargetConfig t = makeGenerousTarget();
    CandidateLegalityResult r = checkCandidateLegality(synth, t);
    assert(!r.isLegal);
    assert(r.reasons.size() == 1);
    assert(r.reasons[0] == LegalityReason::ByteFootprintOverflow);
    // Never a silently wrapped garbage value -- explicit 0 on overflow.
    assert(r.inputTileBytes == 0);
    assert(r.weightTileBytes == 0);
    assert(r.outputTileBytes == 0);
    assert(r.totalScratchpadBytes == 0);
  }
  std::puts("  [ok] overflow-safe arithmetic: problemK overflow -> InvalidReductionTile, "
            "byte-sum overflow -> ByteFootprintOverflow, no wrap in either case");

  // ---- toString() sanity (diagnostics-only stable string form) ----
  assert(toString(LegalityReason::InputTileExceedsBuffer) == "input_tile_exceeds_buffer");
  assert(toString(LegalityReason::ChannelAlignmentViolation) == "channel_alignment_violation");

  std::puts("=== CandidateLegalityTest: PASS ===");
  return 0;
}
