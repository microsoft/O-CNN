#ifndef _OCTREE_TYPES_
#define _OCTREE_TYPES_

#include <cstdint>

// typedef uint8_t uint8;
// typedef uint16_t uint16;
// typedef uint32_t uint32;
// typedef uint64_t uint64;

// these definations are from tensorflow
typedef unsigned char uint8;
typedef unsigned short uint16;
typedef unsigned int uint32;
typedef unsigned long long uint64;

template<typename Dtype> struct KeyTrait {};
template<> struct KeyTrait<uint32> { typedef uint8 uints; };
template<> struct KeyTrait<uint64> { typedef uint16 uints; };

#ifdef KEY64
typedef uint64 uintk;
#else
typedef uint32 uintk;
#endif // KEY64

#endif // _OCTREE_TYPES_
