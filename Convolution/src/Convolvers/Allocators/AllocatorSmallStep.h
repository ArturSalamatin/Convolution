/*****************************************************************//**
 * \file   AllocatorSmallStep.h
 * \brief  The file contains allocator definitions for
 * the SmallStep calculations within the VarStep regime
 * 
 * It specifies data access (struct SmallStep)
 * on read (OnGet...) and write (OnPush...)
 * for kernels (struct KernelSmallStep)
 * and fluxes (struct FluxSmallStep)
 * 
 * \author artur.salamatin
 * \date   June 2023
 *********************************************************************/

#pragma once
#include "AllocatorConstStep.h"

namespace Convolution
{
	// does not require any special Allocator definitions
	// just regime is defined in ../Regimes/SmallStep.h
}

