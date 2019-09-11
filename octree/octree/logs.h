#ifndef _OCTREE_LOGS_H_
#define _OCTREE_LOGS_H_

#ifdef USE_GLOG

#include <glog/logging.h>

#else

#include <iostream>

// do nothing except for evaluating the #expression#, use std::cout to eat the 
// following expressions like "<<...<<...;"
#define CHECK(expression) ((void)0), (expression), (true) ? std::cout : std::cout 

#endif // USE_GLOG

#endif // _OCTREE_LOGS_H_
