
#ifndef __MODEL_H__
#define __MODEL_H__

#define __PREFIX(x) model ## x

#include "Gap.h"

#ifdef __EMUL__
#include <sys/types.h>
#include <unistd.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <sys/param.h>
#include <string.h>
extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE model_L3_Flash;
#endif

#if defined(USE_HYPER)
// model_L3_Flash 应该和 BUILD——MODEL——SQ8BIT/model_L3_Flash_Const.dat 有关，在 modelLernels.c 中有引用这个变量，但不知道是在哪里定义的
extern AT_HYPERFLASH_FS_EXT_ADDR_TYPE model_L3_Flash;
#elif defined(USE_SPI)
extern AT_QSPIFLASH_FS_EXT_ADDR_TYPE model_L3_Flash;
#endif


#endif
