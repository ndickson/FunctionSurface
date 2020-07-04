# FunctionSurface
This is a C++ implementation of an algorithm for creating a surface approximating where a 3D scalar function is zero.  It's similar to marching cubes, but replacing cubes with the "tetragonal disphenoid honeycomb".  Using only tets avoids the ambiguous cases of marching cubes.  It can be parallelized, but isn't yet.

This code relies on my Common library, but just the headers.
