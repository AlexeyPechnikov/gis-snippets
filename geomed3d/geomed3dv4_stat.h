#ifndef GEOMED_H_STAT
#define GEOMED_H_STAT

#ifdef __cplusplus
extern "C" {
#endif

// focal stats
enum {
    GEOMED_STAT_ORIG = 0,
    GEOMED_STAT_ROTMEAN,
    GEOMED_STATS
};
const char* GEOMED_STAT_NAMES[] = 
{
    [GEOMED_STAT_ORIG] = "orig",
    [GEOMED_STAT_ROTMEAN] = "mean"
};

#ifdef __cplusplus
}
#endif

#endif  // GEOMED_H_STAT
