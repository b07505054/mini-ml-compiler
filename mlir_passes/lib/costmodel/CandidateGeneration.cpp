#include "costmodel/KernelCandidate.h"

#include "mlir/IR/BuiltinTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/Support/raw_ostream.h"

using namespace mlir;

namespace mlir::costmodel {

llvm::StringRef toString(Precision precision) {
  switch (precision) {
  case Precision::INT8:
    return "int8";
  case Precision::FP16:
    return "fp16";
  }
  llvm_unreachable("unhandled Precision");
}

llvm::StringRef toString(Dataflow dataflow) {
  switch (dataflow) {
  case Dataflow::WeightStationary:
    return "weight_stationary";
  case Dataflow::OutputStationary:
    return "output_stationary";
  case Dataflow::InputStationary:
    return "input_stationary";
  }
  llvm_unreachable("unhandled Dataflow");
}

const std::vector<Precision> &allPrecisions() {
  static const std::vector<Precision> kPrecisions = {Precision::INT8,
                                                       Precision::FP16};
  return kPrecisions;
}

const std::vector<Dataflow> &allDataflows() {
  static const std::vector<Dataflow> kDataflows = {
      Dataflow::WeightStationary, Dataflow::OutputStationary,
      Dataflow::InputStationary};
  return kDataflows;
}

const std::vector<PEArrayShape> &allPEArrayShapes() {
  static const std::vector<PEArrayShape> kShapes = {
      PEArrayShape{16, 16},
      PEArrayShape{32, 32},
  };
  return kShapes;
}

const std::vector<TileShape> &allBaseSpatialTileShapes() {
  // reductionDepth left at its 0 placeholder -- see the doc comment on
  // this function and on TileShape in KernelCandidate.h.
  static const std::vector<TileShape> kTiles = {
      TileShape{16, 8, 8, 0},
      TileShape{32, 16, 16, 0},
      TileShape{64, 32, 32, 0},
  };
  return kTiles;
}

const std::vector<int64_t> &reductionTileSeeds() {
  static const std::vector<int64_t> kSeeds = {16, 32, 64};
  return kSeeds;
}

std::vector<int64_t> reductionTileChoicesForProblem(int64_t problemK) {
  if (problemK <= 0)
    llvm::report_fatal_error(
        "costmodel::reductionTileChoicesForProblem: problemK must be > 0");
  std::vector<int64_t> choices;
  for (int64_t seed : reductionTileSeeds())
    if (seed <= problemK)
      choices.push_back(seed);
  if (choices.empty() || choices.back() != problemK)
    choices.push_back(problemK);
  return choices;
}

Conv2DProblemShape extractConv2DProblemShape(mlir::hir::Conv2dOp op) {
  auto inputType = llvm::dyn_cast<RankedTensorType>(op.getInput().getType());
  auto filterType = llvm::dyn_cast<RankedTensorType>(op.getFilter().getType());

  if (!inputType || !filterType)
    llvm::report_fatal_error(
        "costmodel::extractConv2DProblemShape: input and filter must both "
        "be ranked tensors (dynamic-rank operands are out of scope for "
        "Cost Model Slice 1)");

  if (inputType.getRank() != 4 || filterType.getRank() != 4)
    llvm::report_fatal_error(
        "costmodel::extractConv2DProblemShape: input and filter must both "
        "be rank-4 (NCHW / KCRS); other ranks are out of scope for Cost "
        "Model Slice 1");

  auto inputShape = inputType.getShape();
  auto filterShape = filterType.getShape();

  for (int64_t dim : inputShape)
    if (ShapedType::isDynamic(dim))
      llvm::report_fatal_error(
          "costmodel::extractConv2DProblemShape: dynamic input dimensions "
          "are out of scope for Cost Model Slice 1");
  for (int64_t dim : filterShape)
    if (ShapedType::isDynamic(dim))
      llvm::report_fatal_error(
          "costmodel::extractConv2DProblemShape: dynamic filter dimensions "
          "are out of scope for Cost Model Slice 1");

  // input: [N, C, H, W], filter: [K, C, R, S].
  Conv2DProblemShape problem;
  problem.batch = inputShape[0];
  problem.inputChannels = inputShape[1];
  problem.inputHeight = inputShape[2];
  problem.inputWidth = inputShape[3];
  problem.outputChannels = filterShape[0];
  problem.kernelHeight = filterShape[2];
  problem.kernelWidth = filterShape[3];

  if (filterShape[1] != problem.inputChannels)
    llvm::report_fatal_error(
        "costmodel::extractConv2DProblemShape: filter input-channel "
        "dimension does not match input tensor's channel dimension");

  // Slice 2 addition: read the output tensor's static shape as the
  // source of truth for output_height/output_width (see KernelCandidate.h
  // doc comment on these fields for why -- HIR_Conv2dOp exposes no typed
  // stride/padding/dilation attributes to derive this arithmetically).
  // Unlike the input/filter checks above (Slice 1, unchanged: abort on a
  // bad shape), a bad/dynamic output shape is recorded gracefully here so
  // Slice 2's legality checker can fail closed with an observable reason
  // instead of crashing.
  if (auto outputType = llvm::dyn_cast<RankedTensorType>(op.getOutput().getType());
      outputType && outputType.getRank() == 4 &&
      !ShapedType::isDynamic(outputType.getShape()[0]) &&
      !ShapedType::isDynamic(outputType.getShape()[2]) &&
      !ShapedType::isDynamic(outputType.getShape()[3])) {
    auto outputShape = outputType.getShape();
    problem.outputHeight = outputShape[2];
    problem.outputWidth = outputShape[3];
    problem.outputShapeIsStaticAndSupported = true;
  }

  return problem;
}

static std::string makeCandidateId(const Conv2DProblemShape &problem,
                                    Precision precision, Dataflow dataflow,
                                    const PEArrayShape &peArray,
                                    const TileShape &tile) {
  // Slice 5: the ID now also encodes tile.reductionDepth (":k=...") so
  // two candidates differing only in reduction-tile depth receive
  // different IDs -- an intentional candidate-ID schema change (old IDs
  // from Slices 1-4 never encoded a K component at all, since every
  // candidate implicitly used the full reduction).
  std::string id;
  llvm::raw_string_ostream os(id);
  os << "conv2d[N=" << problem.batch << ",C=" << problem.inputChannels
     << ",H=" << problem.inputHeight << ",W=" << problem.inputWidth
     << ",K=" << problem.outputChannels << ",R=" << problem.kernelHeight
     << ",S=" << problem.kernelWidth << "]"
     << ":precision=" << toString(precision)
     << ":dataflow=" << toString(dataflow) << ":pe=" << peArray.rows << "x"
     << peArray.cols << ":tile=" << tile.outputChannels << "x" << tile.height
     << "x" << tile.width << ":k=" << tile.reductionDepth;
  os.flush();
  return id;
}

std::vector<KernelCandidate>
generateCandidates(const Conv2DProblemShape &problem) {
  int64_t problemK = 0;
  bool overflow = llvm::MulOverflow(problem.inputChannels, problem.kernelHeight, problemK);
  if (!overflow)
    overflow = llvm::MulOverflow(problemK, problem.kernelWidth, problemK);
  if (overflow)
    llvm::report_fatal_error(
        "costmodel::generateCandidates: inputChannels * kernelHeight * "
        "kernelWidth overflows int64_t");

  std::vector<int64_t> reductionChoices = reductionTileChoicesForProblem(problemK);

  std::vector<KernelCandidate> candidates;
  candidates.reserve(allPrecisions().size() * allDataflows().size() *
                      allPEArrayShapes().size() * allBaseSpatialTileShapes().size() *
                      reductionChoices.size());

  // Fixed nested order: Precision -> Dataflow -> PEArrayShape -> base
  // spatial/channel tile -> reduction-tile choice (Slice 5, innermost).
  // This order is this module's documented determinism contract (see
  // KernelCandidate.h) -- do not reorder without updating that contract.
  for (Precision precision : allPrecisions()) {
    for (Dataflow dataflow : allDataflows()) {
      for (const PEArrayShape &peArray : allPEArrayShapes()) {
        for (const TileShape &baseTile : allBaseSpatialTileShapes()) {
          for (int64_t reductionDepth : reductionChoices) {
            TileShape tile{baseTile.outputChannels, baseTile.height, baseTile.width,
                           reductionDepth};
            KernelCandidate candidate;
            candidate.problem = problem;
            candidate.precision = precision;
            candidate.dataflow = dataflow;
            candidate.peArray = peArray;
            candidate.tile = tile;
            candidate.candidateId =
                makeCandidateId(problem, precision, dataflow, peArray, tile);
            candidates.push_back(std::move(candidate));
          }
        }
      }
    }
  }

  return candidates;
}

} // namespace mlir::costmodel
