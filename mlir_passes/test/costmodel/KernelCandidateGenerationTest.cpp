// Cost Model Slice 1: Candidate Generation unit tests.
//
// Verifies: (1) a real mlir::hir::Conv2dOp's operand shapes are extracted
// correctly, (2) enumeration produces exactly the full Cartesian product
// of Precision x Dataflow x PEArrayShape x TileShape with no duplicates
// and no missing combinations, (3) repeated generation on the same input
// is fully deterministic (identical candidate lists, identical order,
// identical candidate ids), and (4) no candidate carries a score --
// this slice enumerates only.

#include "costmodel/KernelCandidate.h"
#include "HIR/IR/HIRDialect.h"
#include "HIR/IR/HIROps.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"

#include <cassert>
#include <cstdio>
#include <set>
#include <string>
#include <tuple>

using namespace mlir;
using namespace mlir::costmodel;

static mlir::hir::Conv2dOp buildRealConv2dOp(MLIRContext &ctx, Block &block) {
  OpBuilder builder(&ctx);
  Location loc = builder.getUnknownLoc();

  // input: [N=1, C=64, H=56, W=56], filter: [K=128, C=64, R=3, S=3]
  // (NCHW / KCRS, matching this repository's existing conv conventions).
  auto elementType = builder.getIntegerType(8);
  auto inputType = RankedTensorType::get({1, 64, 56, 56}, elementType);
  auto filterType = RankedTensorType::get({128, 64, 3, 3}, elementType);
  auto outputType = RankedTensorType::get({1, 128, 54, 54}, elementType);

  block.addArgument(inputType, loc);
  block.addArgument(filterType, loc);
  builder.setInsertionPointToStart(&block);

  return builder.create<mlir::hir::Conv2dOp>(loc, outputType,
                                              block.getArgument(0),
                                              block.getArgument(1));
}

int main() {
  std::puts("=== KernelCandidateGenerationTest ===");

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
  mlir::hir::Conv2dOp conv = buildRealConv2dOp(ctx, block);

  // ---- (1) shape extraction from the real HIR conv2d op ----
  Conv2DProblemShape problem = extractConv2DProblemShape(conv);
  assert(problem.batch == 1);
  assert(problem.inputChannels == 64);
  assert(problem.inputHeight == 56);
  assert(problem.inputWidth == 56);
  assert(problem.outputChannels == 128);
  assert(problem.kernelHeight == 3);
  assert(problem.kernelWidth == 3);
  std::puts("  [ok] Conv2DProblemShape extracted from a real HIR Conv2dOp");

  // ---- (2) full Cartesian-product coverage, no duplicates ----
  // Slice 5: the tile axis is now (base spatial/channel tile) x
  // (reduction-tile choice), the latter problem-dependent -- see
  // KernelCandidate.h. For the reference problem, problemK = 64*3*3 =
  // 576, and reductionTileChoicesForProblem(576) == {16, 32, 64, 576}.
  std::vector<KernelCandidate> candidates = generateCandidates(problem);

  int64_t problemK = problem.inputChannels * problem.kernelHeight * problem.kernelWidth;
  assert(problemK == 576);
  std::vector<int64_t> reductionChoices = reductionTileChoicesForProblem(problemK);
  assert(reductionChoices.size() == 4);
  assert((reductionChoices == std::vector<int64_t>{16, 32, 64, 576}));

  size_t expectedCount = allPrecisions().size() * allDataflows().size() *
                         allPEArrayShapes().size() * allBaseSpatialTileShapes().size() *
                         reductionChoices.size();
  assert(expectedCount == 2 * 3 * 2 * 3 * 4); // 144, pinned so the test fails
                                               // loudly if the fixed axes change
  assert(candidates.size() == expectedCount);

  std::set<std::string> seenIds;
  std::set<std::tuple<int, int, int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>>
      seenCombinations;
  for (const KernelCandidate &c : candidates) {
    // No scoring at this slice: nothing resembling a cost/score field
    // exists on KernelCandidate at all (enforced structurally, not by a
    // runtime check -- see KernelCandidate.h).
    assert(c.problem == problem);
    assert(c.tile.reductionDepth > 0);
    assert(c.tile.reductionDepth <= problemK);

    bool inserted = seenIds.insert(c.candidateId).second;
    assert(inserted && "duplicate candidate id");

    auto key = std::make_tuple(static_cast<int>(c.precision), static_cast<int>(c.dataflow),
                                c.peArray.rows, c.peArray.cols, c.tile.outputChannels,
                                c.tile.height, c.tile.width, c.tile.reductionDepth);
    bool combinationInserted = seenCombinations.insert(key).second;
    assert(combinationInserted &&
           "duplicate (precision,dataflow,pe,tile,reductionDepth) combination");
  }
  assert(seenIds.size() == expectedCount);
  assert(seenCombinations.size() == expectedCount);

  // Every declared axis value must appear at least once.
  for (Precision p : allPrecisions()) {
    bool found = false;
    for (const auto &c : candidates)
      found |= (c.precision == p);
    assert(found);
  }
  for (Dataflow d : allDataflows()) {
    bool found = false;
    for (const auto &c : candidates)
      found |= (c.dataflow == d);
    assert(found);
  }
  for (const PEArrayShape &pe : allPEArrayShapes()) {
    bool found = false;
    for (const auto &c : candidates)
      found |= (c.peArray == pe);
    assert(found);
  }
  for (const TileShape &base : allBaseSpatialTileShapes()) {
    bool found = false;
    for (const auto &c : candidates)
      found |= (c.tile.outputChannels == base.outputChannels &&
                c.tile.height == base.height && c.tile.width == base.width);
    assert(found);
  }
  for (int64_t k : reductionChoices) {
    bool found = false;
    for (const auto &c : candidates)
      found |= (c.tile.reductionDepth == k);
    assert(found);
  }
  std::printf("  [ok] full Cartesian product covered exactly once: %zu candidates\n",
              candidates.size());

  // ---- (3) determinism: repeated generation is identical, in order ----
  std::vector<KernelCandidate> candidatesAgain = generateCandidates(problem);
  assert(candidatesAgain.size() == candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i) {
    assert(candidates[i] == candidatesAgain[i]);
    assert(candidates[i].candidateId == candidatesAgain[i].candidateId);
  }
  std::puts("  [ok] repeated generateCandidates() calls are identical and in the same order");

  // A second, independently-built Conv2dOp with the same shape must
  // produce the same candidate ids (determinism is a function of the
  // problem shape, not of object identity / pointer addresses).
  Block block2;
  mlir::hir::Conv2dOp conv2 = buildRealConv2dOp(ctx, block2);
  Conv2DProblemShape problem2 = extractConv2DProblemShape(conv2);
  assert(problem2 == problem);
  std::vector<KernelCandidate> candidatesFromSecondOp =
      generateCandidates(problem2);
  assert(candidatesFromSecondOp.size() == candidates.size());
  for (size_t i = 0; i < candidates.size(); ++i)
    assert(candidatesFromSecondOp[i].candidateId == candidates[i].candidateId);
  std::puts("  [ok] determinism holds across independently-built ops with equal shape");

  // ---- (4) a different conv2d problem shape changes the ids but not the count ----
  Conv2DProblemShape otherProblem = problem;
  otherProblem.outputChannels = 256;
  std::vector<KernelCandidate> otherCandidates = generateCandidates(otherProblem);
  assert(otherCandidates.size() == candidates.size());
  assert(otherCandidates.front().candidateId != candidates.front().candidateId);
  std::puts("  [ok] candidate ids vary with problem shape; candidate count does not");

  std::puts("=== KernelCandidateGenerationTest: PASS ===");
  return 0;
}
