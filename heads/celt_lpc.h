#ifndef PLC_H
#define PLC_H

#include "arch.h"
#include "common.h"

#if defined(OPUS_X86_MAY_HAVE_SSE4_1)
#include "x86/celt_lpc_sse.h"
#endif

#define LPC_ORDER 24

void rnn_lpc(opus_val16* _lpc, const opus_val32* ac, int p);

int rnn_autocorr(const opus_val16* x, opus_val32* ac,
    const opus_val16* window, int overlap, int lag, int n);

#endif /* PLC_H */
#pragma once
