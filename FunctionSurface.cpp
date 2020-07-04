// This file defines a function, createSurface, that accepts a
// bounding box, a resolution, and a 3D scalar function, and creates a
// surface approximating where the function is zero.  It can be parallelized,
// but isn't yet.
//
// It uses an algorithm similar to marching cubes, but replacing cubes
// with the "tetragonal disphenoid honeycomb", which is like body-centred cubic,
// split into tetrahedra (tets) by connecting cell centres to all adjacent cell
// centres and to all corners of the cell, in addition to the cubic lattice.
// Using only tets avoids the ambiguous cases of marching cubes, and this honeycomb
// yields the "nicest" tets that can fill all of 3D space.
//
// Since the tetrahedra don't stay within the space of cubic cells, the tets
// "belonging" to a cell are considered to be the 12 tets forming 3 octahedra
// going from the cell centre back to previous cell centres in the x, y, and z
// axis directions.  The edges "belonging" to a cell are the 8 edges from the
// cell centre to the cube corner points, the 3 edges from the cell centre to
// previous cell centres in the x, y, and z axis directions, and the 3 edges
// from the "end" corner of the cell to the previous corners in the axis directions,
// for a total of 14 edges per cell.
//
// To avoid requiring n^3 space or having duplicate points, the algorithm
// alternates computing zero-crossing points along edges and creating polygons
// connecting those points, one z-axis layer at a time.  To avoid producing
// points outside the bounding box or not reaching the edge of the bounding box,
// one more cell layer is added on each side of the box, (i.e. 2 x layers, 2 y layers,
// and 2 z layers), and cell centres that would be in those layers are forced to be
// on the boundary of the box.  Some things are not computed for these outer cells,
// since they're not needed and would result in out of bounds memory accesses.
//
// Note that cells may be rectangular prisms, and not cubes, if the bounding box
// isn't a cube or if the resolution doesn't have the same number of cells along
// each axis.

#include <Array.h>
#include <ArrayDef.h>
#include <Box.h>
#include <Types.h>
#include <Vec.h>

#include <algorithm> // For std::swap
#include <cmath>     // For std::signbit
#include <memory>    // For std::unique_ptr

using namespace OUTER_NAMESPACE;
using namespace OUTER_NAMESPACE :: COMMON_LIBRARY_NAMESPACE;

OUTER_NAMESPACE_BEGIN
namespace FunctionSurface {

// This is used to represent no point existing somewhere.
// For example, a triangle is represented as a quadrilateral
// where the 4th point (i.e. point 3) is noPoint.
constexpr static size_t noPoint = ~size_t(0);

// Things inside this namespace are just for use by createSurface.
namespace internal {

// 8 diagonal edges
// 3 edges from centre to prev centres
// 3 edges from end corner to prev corners
// = 14 edges per cell
constexpr static size_t edgesPerCell = 14;
constexpr static size_t numDiagonals = 8;
constexpr static size_t midBackEdgesStart = numDiagonals;
constexpr static size_t endBackEdgesStart = midBackEdgesStart + 3;

// Negative (sign bit 1) is inside.
// Positive (sign bit 0) is outside.
// Normals go from inside to outside, and polygon winding is right-handed.
//
// Edge 0 is between current centre (0) to previous centre (1)
// Edge 1 is between the two corners (2 and 3)
// Edge 2 is between current centre (0) and first corner (2)
// Edge 3 is between current centre (0) and second corner (3)
// Edge 4 is between previous centre (1) and first corner (2)
// Edge 5 is between previous centre (1) and second corner (3)
constexpr static size_t signsToPolyEdgesMapping[14][4] = {
	// No entry for all 4 points outside
	{0, 3, 2, noPoint}, // 0 inside -> triangle 0, 3, 2
	{0, 4, 5, noPoint}, // 1 inside -> triangle 0, 4, 5
	{2, 4, 5, 3}, // 0 and 1 inside -> quad 2, 4, 5, 3
	{1, 4, 2, noPoint}, // 2 inside -> triangle 1, 4, 2
	{0, 3, 1, 4}, // 0 and 2 inside -> quad 0, 3, 1, 4
	{0, 2, 1, 5}, // 1 and 2 inside -> quad 0, 2, 1, 5
	{1, 5, 3, noPoint}, // 3 outside -> triangle 1, 5, 3

	// Second half is first half reversed and in opposite order.
	{1, 3, 5, noPoint}, // 3 inside -> triangle 1, 3, 5
	{0, 5, 1, 2}, // 1 and 2 outside -> quad 0, 5, 1, 2
	{0, 4, 1, 3}, // 0 and 2 outside -> quad 0, 4, 1, 3
	{1, 2, 4, noPoint}, // 2 outside -> triangle 1, 2, 4
	{2, 3, 5, 4}, // 0 and 1 outside -> quad 2, 3, 5, 4
	{0, 5, 4, noPoint}, // 1 outside -> triangle 0, 5, 4
	{0, 2, 3, noPoint}  // 0 outside -> triangle 0, 2, 3
	// No entry for all 4 points inside
};

double forceCentreInside(
	const size_t i,
	const Spand& bounds,
	const double size,
	const size_t res
) {
	return (i == 0) ? bounds[0] : ((i > res) ? bounds[1] : (bounds[0] + size*((i - 0.5)/res)));
}

void computeSingleOctahedron(
	const size_t centreEdgePoint,
	const size_t planeEdgePoints[4],
	const size_t currentCornerEdgePoints[4],
	const size_t prevCornerEdgePoints[4],
	const bool centreSign,
	const bool prevCentreSign,
	const bool prevCornerSigns[4],
	Array<Vec4<size_t>>& polygons
) {
	for (size_t teti = 0; teti < 4; ++teti) {
		const uint32 signBits =
			uint32(centreSign) |
			(uint32(prevCentreSign)<<1) |
			(uint32(prevCornerSigns[teti])<<2) |
			(uint32(prevCornerSigns[(teti+1)&3])<<3);
		// The most common cases are that all points are inside
		// or all points are outside.
		if (signBits == 0 || signBits == 0xF) {
			continue;
		}

		const size_t tetEdgePoints[6] = {
			centreEdgePoint,
			planeEdgePoints[teti],
			currentCornerEdgePoints[teti],
			currentCornerEdgePoints[(teti+1)&3],
			prevCornerEdgePoints[teti],
			prevCornerEdgePoints[(teti+1)&3]
		};

		const size_t*const polyEdges = signsToPolyEdgesMapping[signBits-1];
		assert(tetEdgePoints[polyEdges[0]] != noPoint);
		assert(tetEdgePoints[polyEdges[1]] != noPoint);
		assert(tetEdgePoints[polyEdges[2]] != noPoint);
		if (polyEdges[3] == noPoint) {
			polygons.append(Vec4<size_t>(
				tetEdgePoints[polyEdges[0]],
				tetEdgePoints[polyEdges[1]],
				tetEdgePoints[polyEdges[2]],
				noPoint
			));
		}
		else {
			assert(tetEdgePoints[polyEdges[3]] != noPoint);
			polygons.append(Vec4<size_t>(
				tetEdgePoints[polyEdges[0]],
				tetEdgePoints[polyEdges[1]],
				tetEdgePoints[polyEdges[2]],
				tetEdgePoints[polyEdges[3]]
			));
		}
	}
}

template<typename FUNCTOR>
void computeCentreValues(
	const Box3d& box,
	const Vec3<size_t> &res,
	const FUNCTOR& functor,
	double* values,
	size_t z
) {
	const Vec3<size_t> centreRes(res[0]+2, res[1]+2, res[2]+2);
	const Vec3d& size = box.size();
	const double zd = forceCentreInside(z, box[2], size[2], res[2]);

	// y == 0 iteration
	size_t i = 0;
	double yd = box[1][0];
	values[i] = functor(Vec3d(box[0][0], yd, zd));
	++i;
	for (size_t x = 1; x < centreRes[0]-1; ++x, ++i) {
		const double xd = box[0][0] + size[0]*((x - 0.5)/res[0]);
		values[i] = functor(Vec3d(xd, yd, zd));
	}
	values[i] = functor(Vec3d(box[0][1], yd, zd));
	++i;

	// Middle y iterations
	for (size_t y = 1; y < centreRes[1]-1; ++y) {
		yd = box[1][0] + size[1]*((y - 0.5)/res[1]);
		values[i] = functor(Vec3d(box[0][0], yd, zd));
		++i;
		for (size_t x = 1; x < centreRes[0]-1; ++x, ++i) {
			const double xd = box[0][0] + size[0]*((x - 0.5)/res[0]);
			values[i] = functor(Vec3d(xd, yd, zd));
		}
		values[i] = functor(Vec3d(box[0][1], yd, zd));
		++i;
	}

	// y == centreRes[1]-1 iteration
	yd = box[1][1];
	values[i] = functor(Vec3d(box[0][0], yd, zd));
	++i;
	for (size_t x = 1; x < centreRes[0]-1; ++x, ++i) {
		const double xd = box[0][0] + size[0]*((x - 0.5)/res[0]);
		values[i] = functor(Vec3d(xd, yd, zd));
	}
	values[i] = functor(Vec3d(box[0][1], yd, zd));
	++i;
}

template<typename FUNCTOR>
void computeCornerValues(
	const Box3d& box,
	const Vec3<size_t> &res,
	const FUNCTOR& functor,
	double* values,
	size_t z
) {
	const Vec3<size_t> cornerRes(res[0]+1, res[1]+1, res[2]+1);
	const Vec3d& size = box.size();
	const double zd = box[2][0] + size[2]*(double(z)/res[2]);
	for (size_t y = 0, i = 0; y < cornerRes[1]; ++y) {
		const double yd = box[1][0] + size[1]*(double(y)/res[1]);
		for (size_t x = 0; x < cornerRes[0]; ++x, ++i) {
			const double xd = box[0][0] + size[0]*(double(x)/res[0]);
			values[i] = functor(Vec3d(xd, yd, zd));
		}
	}
}

void computeEdgePoints(
	const Box3d& box,
	const Vec3<size_t> &res,
	const double* prevCentreValues,
	const double* prevCornerValues,
	const double* currentCentreValues,
	const double* nextCornerValues,
	size_t* nextEdgePoints,
	Array<Vec3d>& positions,
	size_t z
) {
	const Vec3<size_t> cornerRes(res[0]+1, res[1]+1, res[2]+1);
	const Vec3<size_t> centreRes(res[0]+2, res[1]+2, res[2]+2);
	const Vec3<size_t> edgeCellRes(res[0]+2, res[1]+2, res[2]+2);
	const Vec3d& size = box.size();

	// NOTE: Only the centre values need to be forced to be within the bounding box,
	// since when the other values are used, they're already inside the bounding box.
	const double zPrevCentre = forceCentreInside(z-1, box[2], size[2], res[2]);
	const double zStart      = box[2][0] + size[2]*(double(z-1)/res[2]);
	const double zCentre     = forceCentreInside(z, box[2], size[2], res[2]);
	const double zEnd        = box[2][0] + size[2]*(double(z)/res[2]);

	size_t* cellEdgePoints = nextEdgePoints;
	for (size_t y = 0, cell = 0; y < edgeCellRes[1]; ++y) {
		const double yPrevCentre = forceCentreInside(y-1, box[1], size[1], res[1]);
		const double yStart      = box[1][0] + size[1]*(double(y-1)/res[1]);
		const double yCentre     = forceCentreInside(y, box[1], size[1], res[1]);
		const double yEnd        = box[1][0] + size[1]*(double(y)/res[1]);

		for (size_t x = 0; x < edgeCellRes[0]; ++x, ++cell, cellEdgePoints += edgesPerCell) {
			const double xPrevCentre = forceCentreInside(x-1, box[0], size[0], res[0]);
			const double xStart      = box[0][0] + size[0]*(double(x-1)/res[0]);
			const double xCentre     = forceCentreInside(x, box[0], size[0], res[0]);
			const double xEnd        = box[0][0] + size[0]*(double(x)/res[0]);

			const double centreValue = currentCentreValues[cell];
			const bool centreSign = std::signbit(centreValue);

			const size_t cornerCellStart = (y-1)*cornerRes[0] + (x-1);

			// Most edges will not have a crossing, so we might as well
			// clear them all in advance, without negatively impacting performance.
			for (size_t edgei = 0; edgei < edgesPerCell; ++edgei) {
				cellEdgePoints[edgei] = noPoint;
			}

			const bool isFirstOfAxis[3] = {
				x == 0,
				y == 0,
				z == 0
			};

			const bool isLastOfAxis[3] = {
				x == edgeCellRes[0]-1,
				y == edgeCellRes[1]-1,
				z == edgeCellRes[2]-1
			};

			const size_t numFirsts = size_t(isFirstOfAxis[0]) + size_t(isFirstOfAxis[1]) + size_t(isFirstOfAxis[2]);
			const size_t numLasts = size_t(isLastOfAxis[0]) + size_t(isLastOfAxis[1]) + size_t(isLastOfAxis[2]);
			// No edges are needed from cells that are the first in more than 1 axis or last in more than 1 axis,
			// EXCEPT for if numFirsts is 2, the end corner axis for the other axis is still needed.
			if (numLasts > 1 || numFirsts > 2) {
				continue;
			}

			if (numFirsts <= 1) {
				// 8 diagonal edges first
				for (size_t diagi = 0; diagi < numDiagonals; ++diagi) {
					// First cells of an axis exclude all diagonals going backward.
					if ((isFirstOfAxis[0] && !(diagi & 1)) || (isFirstOfAxis[1] && !(diagi & 2)) || (isFirstOfAxis[2] && !(diagi & 4))) {
						continue;
					}
					// Last cells of an axis exclude all diagonals going forward.
					if ((isLastOfAxis[0] && (diagi & 1)) || (isLastOfAxis[1] && (diagi & 2)) || (isLastOfAxis[2] && (diagi & 4))) {
						continue;
					}

					const size_t xOffset = (diagi & 1);
					const size_t yOffset = (diagi & 2) ? cornerRes[0] : 0;
					const double*const cornerValues = (diagi & 4) ? nextCornerValues : prevCornerValues;

					const double cornerValue = cornerValues[cornerCellStart + yOffset + xOffset];

					if (centreSign != std::signbit(cornerValue)) {
						// Zero-crossing on this edge, so find its position.
						const double diff = cornerValue - centreValue;
						// NOTE: diff can be zero if one value is +0.0 and the other is -0.0
						const double t = (diff == 0) ? 0.5 : (-centreValue/diff);
						const Vec3d centrePosition(xCentre, yCentre, zCentre);
						const Vec3d cornerPosition(
							(!(diagi & 1)) ? xStart : xEnd,
							(!(diagi & 2)) ? yStart : yEnd,
							(!(diagi & 4)) ? zStart : zEnd
						);
						const Vec3d crossingPosition = centrePosition + t*(cornerPosition - centrePosition);
						const size_t positionIndex = positions.size();
						positions.append(crossingPosition);
						cellEdgePoints[diagi] = positionIndex;
					}
				}
			}

			// There are no centre-to-previous-centre edges if first in any axis.
			if (numFirsts == 0) {
				// 3 centre-to-previous-centre edges
				const double axisPrevCentreValues[3] = {
					currentCentreValues[cell-1],
					currentCentreValues[cell-centreRes[0]],
					prevCentreValues[cell]
				};

				for (size_t axis = 0; axis < 3; ++axis) {
					// Last cells of an axis only have a centre-to-previous-centre edge for that axis.
					if (numLasts == 1 && !isLastOfAxis[axis]) {
						continue;
					}

					if (centreSign != std::signbit(axisPrevCentreValues[axis])) {
						// Zero-crossing on this edge, so find its position.
						const double diff = axisPrevCentreValues[axis] - centreValue;
						// NOTE: diff can be zero if one value is +0.0 and the other is -0.0
						const double t = (diff == 0) ? 0.5 : (-centreValue/diff);
						const Vec3d centrePosition(xCentre, yCentre, zCentre);
						const Vec3d prevCentrePosition(
							(axis == 0) ? xPrevCentre : xCentre,
							(axis == 1) ? yPrevCentre : yCentre,
							(axis == 2) ? zPrevCentre : zCentre
						);
						const Vec3d crossingPosition = centrePosition + t*(prevCentrePosition - centrePosition);
						const size_t positionIndex = positions.size();
						positions.append(crossingPosition);
						cellEdgePoints[midBackEdgesStart+axis] = positionIndex;
					}
				}
			}

			// There are no end-to-previous-corner edges if last in any axis.
			if (numLasts == 0) {
				// 3 end-to-previous-corner edges
				const double endCornerValue = nextCornerValues[cornerCellStart + cornerRes[0] + 1];
				const bool endCornerSign = std::signbit(endCornerValue);
				const double axisPrevCornerValues[3] = {
					nextCornerValues[cornerCellStart + cornerRes[0]],
					nextCornerValues[cornerCellStart + 1],
					isFirstOfAxis[2] ? 0.0 : prevCornerValues[cornerCellStart + cornerRes[0] + 1]
				};

				for (size_t axis = 0; axis < 3; ++axis) {
					// First cells of one or two axes don't have an end-to-previous-corner edge for that axis.
					if (isFirstOfAxis[axis]) {
						continue;
					}

					if (endCornerSign != std::signbit(axisPrevCornerValues[axis])) {
						// Zero-crossing on this edge, so find its position.
						const double diff = axisPrevCornerValues[axis] - endCornerValue;
						// NOTE: diff can be zero if one value is +0.0 and the other is -0.0
						const double t = (diff == 0) ? 0.5 : (-endCornerValue/diff);
						const Vec3d endCornerPosition(xEnd, yEnd, zEnd);
						const Vec3d prevCornerPosition(
							(axis == 0) ? xStart : xEnd,
							(axis == 1) ? yStart : yEnd,
							(axis == 2) ? zStart : zEnd
						);
						const Vec3d crossingPosition = endCornerPosition + t*(prevCornerPosition - endCornerPosition);
						const size_t positionIndex = positions.size();
						positions.append(crossingPosition);
						cellEdgePoints[endBackEdgesStart+axis] = positionIndex;
					}
				}
			}
		}
	}
};

void computePolygons(
	const Vec3<size_t> &res,
	const double* prevCentreValues,
	const double* prevCornerValues,
	const double* currentCentreValues,
	const double* nextCornerValues,
	const size_t* prevEdgePoints,
	const size_t* nextEdgePoints,
	size_t z,
	Array<Vec4<size_t>>& polygons
) {
	const Vec3<size_t> cornerRes(res[0]+1, res[1]+1, res[2]+1);
	const Vec3<size_t> centreRes(res[0]+2, res[1]+2, res[2]+2);
	const Vec3<size_t> edgeCellRes(res[0]+2, res[1]+2, res[2]+2);

	const size_t* cellPrevZEdgePoints = prevEdgePoints;
	const size_t* cellEdgePoints = nextEdgePoints;
	// Skip the first cell in each axis.
	const size_t edgesPerRow = edgesPerCell*edgeCellRes[0];
	cellEdgePoints += edgesPerRow;
	cellPrevZEdgePoints += edgesPerRow;
	for (size_t y = 1, cell = edgeCellRes[0]; y < edgeCellRes[1]; ++y) {
		cellEdgePoints += edgesPerCell;
		cellPrevZEdgePoints += edgesPerCell;
		++cell;
		for (size_t x = 1; x < edgeCellRes[0]; ++x, ++cell, cellEdgePoints += edgesPerCell, cellPrevZEdgePoints += edgesPerCell) {

			const size_t* cellPrevXEdgePoints = cellEdgePoints - edgesPerCell;
			const size_t* cellPrevZXEdgePoints = cellPrevZEdgePoints - edgesPerCell;
			const size_t* cellPrevYEdgePoints = cellEdgePoints - edgesPerRow;
			const size_t* cellPrevYZEdgePoints = cellPrevZEdgePoints - edgesPerRow;
			const size_t* cellPrevXYEdgePoints = cellEdgePoints - edgesPerCell - edgesPerRow;

			const bool isLastOfAxis[3] = {
				x == edgeCellRes[0]-1,
				y == edgeCellRes[1]-1,
				z == edgeCellRes[2]-1
			};
			const size_t numLasts = size_t(isLastOfAxis[0]) + size_t(isLastOfAxis[1]) + size_t(isLastOfAxis[2]);
			if (numLasts > 1) {
				// No octahedra if last in more than one axis.
				continue;
			}

			// 12 tetrahedra per cell:
			// 4 from the centre back along z
			// 4 from the centre back along y
			// 4 from the centre back along x
			const bool centreSign = std::signbit(currentCentreValues[cell]);

			const size_t cornerCellStart = (y-1)*cornerRes[0] + (x-1);

			bool axisPrevCentreSigns[3];
			bool axisPrevCornerSigns[3][4];

			if (numLasts == 0 || isLastOfAxis[2]) {
				axisPrevCentreSigns[2] = std::signbit(prevCentreValues[cell]);
				// xy plane
				axisPrevCornerSigns[2][0] = std::signbit(prevCornerValues[cornerCellStart]);
				axisPrevCornerSigns[2][1] = std::signbit(prevCornerValues[cornerCellStart + 1]);
				axisPrevCornerSigns[2][2] = std::signbit(prevCornerValues[cornerCellStart + cornerRes[0] + 1]);
				axisPrevCornerSigns[2][3] = std::signbit(prevCornerValues[cornerCellStart + cornerRes[0]]);
			}
			else {
				// Exclude octahedron by treating sign as equal to centreSign.
				axisPrevCentreSigns[2] = centreSign;
				for (size_t i = 0; i < 4; ++i) {
					axisPrevCornerSigns[2][i] = centreSign;
				}
			}

			if (numLasts == 0 || isLastOfAxis[1]) {
				axisPrevCentreSigns[1] = std::signbit(currentCentreValues[cell-centreRes[0]]);
				// zx plane
				axisPrevCornerSigns[1][0] = std::signbit(prevCornerValues[cornerCellStart]);
				axisPrevCornerSigns[1][1] = std::signbit(nextCornerValues[cornerCellStart]);
				axisPrevCornerSigns[1][2] = std::signbit(nextCornerValues[cornerCellStart + 1]);
				axisPrevCornerSigns[1][3] = std::signbit(prevCornerValues[cornerCellStart + 1]);
			}
			else {
				// Exclude octahedron by treating sign as equal to centreSign.
				axisPrevCentreSigns[1] = centreSign;
				for (size_t i = 0; i < 4; ++i) {
					axisPrevCornerSigns[1][i] = centreSign;
				}
			}

			if (numLasts == 0 || isLastOfAxis[0]) {
				axisPrevCentreSigns[0] = std::signbit(currentCentreValues[cell-1]);
				// yz plane
				axisPrevCornerSigns[0][0] = std::signbit(prevCornerValues[cornerCellStart]);
				axisPrevCornerSigns[0][1] = std::signbit(prevCornerValues[cornerCellStart + cornerRes[0]]);
				axisPrevCornerSigns[0][2] = std::signbit(nextCornerValues[cornerCellStart + cornerRes[0]]);
				axisPrevCornerSigns[0][3] = std::signbit(nextCornerValues[cornerCellStart]);
			}
			else {
				// Exclude octahedron by treating sign as equal to centreSign.
				axisPrevCentreSigns[0] = centreSign;
				for (size_t i = 0; i < 4; ++i) {
					axisPrevCornerSigns[0][i] = centreSign;
				}
			}

			bool anyInAxisOctahedron[3];
			for (size_t axis = 0; axis < 3; ++axis) {
				anyInAxisOctahedron[axis] = !(
					(centreSign == axisPrevCentreSigns[axis]) &&
					(centreSign == axisPrevCornerSigns[axis][0]) &&
					(centreSign == axisPrevCornerSigns[axis][1]) &&
					(centreSign == axisPrevCornerSigns[axis][2]) &&
					(centreSign == axisPrevCornerSigns[axis][3])
				);
			}

			// z axis tets
			if (anyInAxisOctahedron[2]) {
				const size_t centreEdgePoint = cellEdgePoints[midBackEdgesStart+2];
				const size_t xyEdgePoints[4] = {
					cellPrevYZEdgePoints[endBackEdgesStart+0],
					cellPrevZEdgePoints[endBackEdgesStart+1],
					cellPrevZEdgePoints[endBackEdgesStart+0],
					cellPrevZXEdgePoints[endBackEdgesStart+1]
				};
				// NOTE: 0, 1, 3, 2, for cycle, instead of zig-zag
				const size_t currentCornerEdgePoints[4] = {
					cellEdgePoints[0],
					cellEdgePoints[1],
					cellEdgePoints[3],
					cellEdgePoints[2]
				};
				// NOTE: 4, 5, 7, 6, for cycle, instead of zig-zag
				const size_t prevCornerEdgePoints[4] = {
					cellPrevZEdgePoints[4],
					cellPrevZEdgePoints[5],
					cellPrevZEdgePoints[7],
					cellPrevZEdgePoints[6]
				};

				computeSingleOctahedron(
					centreEdgePoint,
					xyEdgePoints,
					currentCornerEdgePoints,
					prevCornerEdgePoints,
					centreSign,
					axisPrevCentreSigns[2],
					axisPrevCornerSigns[2],
					polygons
				);
			}

			// y axis tets
			if (anyInAxisOctahedron[1]) {
				const size_t centreEdgePoint = cellEdgePoints[midBackEdgesStart+1];
				const size_t zxEdgePoints[4] = {
					cellPrevXYEdgePoints[endBackEdgesStart+2],
					cellPrevYEdgePoints[endBackEdgesStart+0],
					cellPrevYEdgePoints[endBackEdgesStart+2],
					cellPrevYZEdgePoints[endBackEdgesStart+0]
				};
				const size_t currentCornerEdgePoints[4] = {
					cellEdgePoints[0],
					cellEdgePoints[4],
					cellEdgePoints[5],
					cellEdgePoints[1]
				};
				const size_t prevCornerEdgePoints[4] = {
					cellPrevYEdgePoints[2],
					cellPrevYEdgePoints[6],
					cellPrevYEdgePoints[7],
					cellPrevYEdgePoints[3]
				};

				computeSingleOctahedron(
					centreEdgePoint,
					zxEdgePoints,
					currentCornerEdgePoints,
					prevCornerEdgePoints,
					centreSign,
					axisPrevCentreSigns[1],
					axisPrevCornerSigns[1],
					polygons
				);
			}

			// x axis tets
			if (anyInAxisOctahedron[0]) {
				const size_t centreEdgePoint = cellEdgePoints[midBackEdgesStart+0];
				const size_t xyEdgePoints[4] = {
					cellPrevZXEdgePoints[endBackEdgesStart+1],
					cellPrevXEdgePoints[endBackEdgesStart+2],
					cellPrevXEdgePoints[endBackEdgesStart+1],
					cellPrevXYEdgePoints[endBackEdgesStart+2]
				};
				const size_t currentCornerEdgePoints[4] = {
					cellEdgePoints[0],
					cellEdgePoints[2],
					cellEdgePoints[6],
					cellEdgePoints[4]
				};
				const size_t prevCornerEdgePoints[4] = {
					cellPrevXEdgePoints[1],
					cellPrevXEdgePoints[3],
					cellPrevXEdgePoints[7],
					cellPrevXEdgePoints[5]
				};

				computeSingleOctahedron(
					centreEdgePoint,
					xyEdgePoints,
					currentCornerEdgePoints,
					prevCornerEdgePoints,
					centreSign,
					axisPrevCentreSigns[0],
					axisPrevCornerSigns[0],
					polygons
				);
			}
		}
	}
};

} // namespace internal

// This function fills positions and polygons with a representation of a
// surface approximating where functor is zero, inside box, sampling the box
// using res[0] by res[1] by res[2] cells.  polygons are each either a triangle
// or a quadrilateral, containing indices referring to the positions array,
// and triangles are represented by having their last index be "noPoint" (above,
// namely ~size_t(0), or the unsigned equivalent of -1).
template<typename FUNCTOR>
void createSurface(
	const Box3d& box,
	const Vec3<size_t>& res,
	const FUNCTOR& functor,
	Array<Vec3d>& positions,
	Array<Vec4<size_t>>& polygons
) {
	using namespace internal;

	// March along z axis direction, so sweep plane is XY.
	const Vec3<size_t> cornerRes(res[0]+1, res[1]+1, res[2]+1);
	const Vec3<size_t> centreRes(res[0]+2, res[1]+2, res[2]+2);
	const Vec3<size_t> edgeCellRes(res[0]+2, res[1]+2, res[2]+2);
	const size_t xyCornerPointsTotal = cornerRes[0]*cornerRes[1];
	const size_t xyCentrePointsTotal = centreRes[0]*centreRes[1];
	std::unique_ptr<double[]> prevValues(new double[xyCentrePointsTotal + xyCornerPointsTotal]);
	std::unique_ptr<double[]> nextValues(new double[xyCentrePointsTotal + xyCornerPointsTotal]);
	double* prevCentreValues = prevValues.get();
	double* prevCornerValues = prevValues.get() + xyCentrePointsTotal;
	double* nextCentreValues = nextValues.get();
	double* nextCornerValues = nextValues.get() + xyCentrePointsTotal;

	const size_t xyEdgeCellsTotal = edgeCellRes[0]*edgeCellRes[1];
	std::unique_ptr<size_t[]> prevEdgePoints(new size_t[edgesPerCell*xyEdgeCellsTotal]);
	std::unique_ptr<size_t[]> nextEdgePoints(new size_t[edgesPerCell*xyEdgeCellsTotal]);

	// The main loop over z layers.

	// TODO: Parallelize this, being careful to only call computeEdgePoints
	// once for each z layer.
	// computeCentreValues, computeCornerValues, computeEdgePoints, and computePolygons
	// can also be individually parallelized, if res[0] or res[1] are large enough.

	// The first z layer doesn't compute polygons, only edge points.
	// The polygons are created by the next layer.
	computeCentreValues(box, res, functor, prevCentreValues, 0);
	computeCornerValues(box, res, functor, prevCornerValues, 0);
	computeEdgePoints(
		box, res,
		nullptr, nullptr,
		prevCentreValues, prevCornerValues,
		prevEdgePoints.get(), positions, 0
	);

	for (size_t z = 0; z < res[2]; ++z) {
		computeCentreValues(box, res, functor, nextCentreValues, z+1);
		computeCornerValues(box, res, functor, nextCornerValues, z+1);

		computeEdgePoints(
			box, res,
			prevCentreValues, prevCornerValues,
			nextCentreValues, nextCornerValues,
			nextEdgePoints.get(), positions, z+1
		);

		computePolygons(
			res,
			prevCentreValues, prevCornerValues,
			nextCentreValues, nextCornerValues,
			prevEdgePoints.get(), nextEdgePoints.get(),
			z+1, polygons
		);

		// Avoid allocating new memory by just swapping the pointers.
		std::swap(prevCentreValues, nextCentreValues);
		std::swap(prevCornerValues, nextCornerValues);
		std::swap(prevEdgePoints, nextEdgePoints);
	}

	// The last z layer doesn't compute corner points, since they'd never be used.
	computeCentreValues(box, res, functor, nextCentreValues, res[2]+1);

	computeEdgePoints(
		box, res,
		prevCentreValues, prevCornerValues,
		nextCentreValues, nullptr,
		nextEdgePoints.get(), positions, res[2]+1
	);

	computePolygons(
		res,
		prevCentreValues, prevCornerValues,
		nextCentreValues, nullptr,
		prevEdgePoints.get(), nextEdgePoints.get(),
		res[2]+1, polygons
	);
}

} // namespace FunctionSurface
OUTER_NAMESPACE_END
