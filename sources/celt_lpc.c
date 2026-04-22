#include "celt_lpc.h"
#include "arch.h"
#include "common.h"
#include "pitch.h"

void rnn_lpc(
    opus_val16* _lpc, /* out: [0...p-1] LPC coefficients      */
    const opus_val32* ac,  /* in:  [0...p] autocorrelation values  */
    int          p
)
{
    int i, j;
    opus_val32 r;
    opus_val32 error = ac[0];

    float* lpc = _lpc;

    RNN_CLEAR(lpc, p);
    if (ac[0] != 0)
    {
        for (i = 0; i < p; i++) {
            /* Sum up this iteration's reflection coefficient */
            opus_val32 rr = 0;
            for (j = 0; j < i; j++)
                rr += MULT32_32_Q31(lpc[j], ac[i - j]);
            rr += SHR32(ac[i + 1], 3);
            r = -SHL32(rr, 3) / error;
            /*  Update LPC coefficients and total error */
            lpc[i] = SHR32(r, 3);
            for (j = 0; j < (i + 1) >> 1; j++)
            {
                opus_val32 tmp1, tmp2;
                tmp1 = lpc[j];
                tmp2 = lpc[i - 1 - j];
                lpc[j] = tmp1 + MULT32_32_Q31(r, tmp2);
                lpc[i - 1 - j] = tmp2 + MULT32_32_Q31(r, tmp1);
            }

            error = error - MULT32_32_Q31(MULT32_32_Q31(r, r), error);
            /* Bail out once we get 30 dB gain */

            if (error < .001f * ac[0])
                break;
        }
    }
}

float xx[864];
int rnn_autocorr(
    const opus_val16* x,   /*  in: [0...n-1] samples x   */
    opus_val32* ac,  /* out: [0...lag-1] ac values */
    const opus_val16* window,
    int          overlap,
    int          lag,
    int          n)
{
    opus_val32 d;
    int i, k;
    int fastN = n - lag;
    int shift;
    const opus_val16* xptr;
    // opus_val16 xx[n];
    // opus_val16* xx = (opus_val16*)malloc(sizeof(opus_val16) * n);

    celt_assert(n > 0);
    celt_assert(overlap >= 0);
    if (overlap == 0)
    {
        xptr = x;
    }
    else {
        for (i = 0; i < n; i++)
            xx[i] = x[i];
        for (i = 0; i < overlap; i++)
        {
            xx[i] = MULT16_16_Q15(x[i], window[i]);
            xx[n - i - 1] = MULT16_16_Q15(x[n - i - 1], window[i]);
        }
        xptr = xx;
    }
    shift = 0;

    rnn_pitch_xcorr(xptr, xptr, ac, fastN, lag + 1);
    for (k = 0; k <= lag; k++)
    {
        for (i = k + fastN, d = 0; i < n; i++)
            d = MAC16_16(d, xptr[i], xptr[i - k]);
        ac[k] += d;
    }

    // free(xx);
    return shift;
}
