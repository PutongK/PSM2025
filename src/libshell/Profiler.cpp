//
//  Profiler.cpp
//  Elasticity
//
//  Created by Wim van Rees on 3/28/16.
//  Copyright © 2016 Wim van Rees. All rights reserved.
//

#include "Profiler.hpp"

#ifdef USETBB

#include "tbb/tick_count.h"

void ProfileAgent::_getTime(tbb::tick_count& time)
{
    time = tbb::tick_count::now();
}

float ProfileAgent::_getElapsedTime(const tbb::tick_count& tS, const tbb::tick_count& tE)
{
    return (tE - tS).seconds();
}

#else
#include <time.h>
void ProfileAgent::_getTime(clock_t& time)
{
    time = clock();
}

float ProfileAgent::_getElapsedTime(const clock_t& tS, const clock_t& tE)
{
    return (tE - tS)/(double)CLOCKS_PER_SEC;
}

#endif


