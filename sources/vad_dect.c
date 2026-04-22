#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include<stdint.h>
#include "kiss_fft.h"
#include "common.h"
#include <math.h>
#include "pitch.h"
#include "arch.h"
#include "vad_dect.h"
#include "rnn_vad.h"

#define M_PI 3.141592653589793f

#define FRAME_SIZE_SHIFT 2
#define FRAME_SIZE (120<<FRAME_SIZE_SHIFT)
#define WINDOW_SIZE (2*FRAME_SIZE)
#define FREQ_SIZE (FRAME_SIZE + 1)

#define PITCH_MIN_PERIOD 60
#define PITCH_MAX_PERIOD 768
#define PITCH_FRAME_SIZE 960
#define PITCH_BUF_SIZE (PITCH_MAX_PERIOD+PITCH_FRAME_SIZE)

#define SQUARE(x) ((x)*(x))

#define NB_BANDS 22

#define CEPS_MEM 8
#define NB_DELTA_CEPS 6

#define NB_FEATURES (NB_BANDS+3*NB_DELTA_CEPS+2)

static opus_int16 eband5ms[] = {
    /*0  200 400 600 800  1k 1.2 1.4 1.6  2k 2.4 2.8 3.2  4k 4.8 5.6 6.8  8k 9.6 12k 15.6 20k*/
      0,  1,  2,  3,  4,  5,  6,  7,  8, 10, 12, 14, 16, 20, 24, 28, 34, 40, 48, 60, 78, 100
};

typedef struct {
    int init;
    kiss_fft_state* kfft;
    float half_window[FRAME_SIZE];
    float dct_table[NB_BANDS * NB_BANDS];
} CommonState;

typedef struct {
    float analysis_mem[FRAME_SIZE];
    float cepstral_mem[CEPS_MEM][NB_BANDS];
    int memid;
    float synthesis_mem[FRAME_SIZE];
    float pitch_buf[PITCH_BUF_SIZE];
    float pitch_enh_buf[PITCH_BUF_SIZE];
    float last_gain;
    int last_period;
    float mem_hp_x[2];
    float lastg[NB_BANDS];
    float state[24];
}VadState;

static inline float fast_log10f(float x)
{
    union { float f; uint32_t i; } vx = { x };

    int exponent = ((vx.i >> 23) & 255) - 127;

    vx.i = (vx.i & 0x7FFFFF) | 0x3F800000;
    float m = vx.f;

    float log2 = exponent + (m - 1.0f);

    return log2 * 0.30102999566f;
}

static void compute_band_energy(float* bandE, const kiss_fft_cpx* X) {
    int i;
    float sum[NB_BANDS] = { 0 };
    float inv_band_size = 1.0f;
    for (i = 0; i < NB_BANDS - 1; i++)
    {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        inv_band_size = 1.0f / band_size;
        for (j = 0; j < band_size; j++) {
            float tmp;
            // float frac = (float)j / band_size;

            float frac = (float)j * inv_band_size;

            tmp = SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r);
            tmp += SQUARE(X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i);
            sum[i] += (1.0f - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0f;
    sum[NB_BANDS - 1] *= 2.0f;
    for (i = 0; i < NB_BANDS; i++)
    {
        bandE[i] = sum[i];
    }
}

static void compute_band_corr(float* bandE, const kiss_fft_cpx* X, const kiss_fft_cpx* P) {
    int i;
    float sum[NB_BANDS] = { 0 };
    float inv_band_size = 1.0f;
    for (i = 0; i < NB_BANDS - 1; i++)
    {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        inv_band_size = 1.0f / band_size;
        for (j = 0; j < band_size; j++) {
            float tmp;
            // float frac = (float)j / band_size;
            float frac = (float)j * inv_band_size;
            tmp = X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].r;
            tmp += X[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i * P[(eband5ms[i] << FRAME_SIZE_SHIFT) + j].i;
            sum[i] += (1.0f - frac) * tmp;
            sum[i + 1] += frac * tmp;
        }
    }
    sum[0] *= 2.0f;
    sum[NB_BANDS - 1] *= 2.0f;
    for (i = 0; i < NB_BANDS; i++)
    {
        bandE[i] = sum[i];
    }
}

static void interp_band_gain(float* g, const float* bandE) {
    int i;
    float inv_band_size = 1.0f;
    memset(g, 0, FREQ_SIZE);

    for (i = 0; i < NB_BANDS - 1; i++)
    {
        int j;
        int band_size;
        band_size = (eband5ms[i + 1] - eband5ms[i]) << FRAME_SIZE_SHIFT;
        inv_band_size = 1.0f / band_size;
        for (j = 0; j < band_size; j++) {
            // float frac = (float)j / band_size;
            float frac = (float)j * inv_band_size;
            g[(eband5ms[i] << FRAME_SIZE_SHIFT) + j] = (1 - frac) * bandE[i] + frac * bandE[i + 1];
        }
    }
}

CommonState common;
VadState vad_s;

static void check_init()
{
    int i;
    if (common.init) return;
    common.kfft = rnn_fft_alloc_twiddles(2 * FRAME_SIZE, NULL, NULL, NULL, 0);
    
    for (i = 0; i < FRAME_SIZE; i++)
    {
        common.half_window[i] = sinf(.5f * M_PI * sinf(.5f * M_PI * (i + .5f) / FRAME_SIZE) * sinf(.5f * M_PI * (i + .5f) / FRAME_SIZE));
    }

    for (i = 0; i < NB_BANDS; i++)
    {
        int j;
        for (j = 0; j < NB_BANDS; j++)
        {
            common.dct_table[i * NB_BANDS + j] = cosf((i + .5f) * j * M_PI / NB_BANDS);

            // if (j == 0) common.dct_table[i * NB_BANDS + j] *= sqrtf(.5);
            if (j == 0) common.dct_table[i * NB_BANDS + j] *= 0.7071068f;
        }
    }
    common.init = 1;
}

static void dct(float* out, const float* in)
{
    int i;
    for (i = 0; i < NB_BANDS; i++)
    {
        int j;
        float sum = 0;
        for (j = 0; j < NB_BANDS; j++)
        {
            sum += in[j] * common.dct_table[j * NB_BANDS + i];
        }
        // out[i] = sum * sqrt(2. / 22);
        out[i] = sum * 0.3015113f;
    }
}

static void forward_transform(kiss_fft_cpx* out, const float* in) {
    int i;
    kiss_fft_cpx x[WINDOW_SIZE];
    kiss_fft_cpx y[WINDOW_SIZE];
    // check_init();
    for (i = 0; i < WINDOW_SIZE; i++) {
        x[i].r = in[i];
        x[i].i = 0;
    }
    rnn_fft(common.kfft, x, y, 0);
    for (i = 0; i < FREQ_SIZE; i++) {
        out[i] = y[i];
    }
}

static void inverse_transform(float* out, const kiss_fft_cpx* in) {
    int i;
    kiss_fft_cpx x[WINDOW_SIZE];
    kiss_fft_cpx y[WINDOW_SIZE];
    // check_init();
    for (i = 0; i < FREQ_SIZE; i++) {
        x[i] = in[i];
    }
    for (; i < WINDOW_SIZE; i++) {
        x[i].r = x[WINDOW_SIZE - i].r;
        x[i].i = -x[WINDOW_SIZE - i].i;
    }
    rnn_fft(common.kfft, x, y, 0);
    /* output in reverse order for IFFT. */
    out[0] = WINDOW_SIZE * y[0].r;
    for (i = 1; i < WINDOW_SIZE; i++) {
        out[i] = WINDOW_SIZE * y[WINDOW_SIZE - i].r;
    }
}

static void apply_window(float* x)
{
    int i;
    for (i = 0; i < FRAME_SIZE; i++) 
    {
        x[i] *= common.half_window[i];
        x[WINDOW_SIZE - 1 - i] *= common.half_window[i];
    }
}

void vad_dect_init()
{
    // memset(vad_s, 0, sizeof(vad_s));
    /*float analysis_mem[FRAME_SIZE];
    float cepstral_mem[CEPS_MEM][NB_BANDS];
    int memid;
    float synthesis_mem[FRAME_SIZE];
    float pitch_buf[PITCH_BUF_SIZE];
    float pitch_enh_buf[PITCH_BUF_SIZE];
    float last_gain;
    int last_period;
    float mem_hp_x[2];
    float lastg[NB_BANDS];
    float state[24];*/
    memset(vad_s.analysis_mem, 0, sizeof(vad_s.analysis_mem));
    memset(vad_s.cepstral_mem, 0, sizeof(vad_s.cepstral_mem));
    vad_s.memid = 0;
    memset(vad_s.pitch_buf, 0, sizeof(vad_s.pitch_buf));
    memset(vad_s.pitch_enh_buf, 0, sizeof(vad_s.pitch_enh_buf));
    memset(vad_s.mem_hp_x, 0, sizeof(vad_s.mem_hp_x));
    memset(vad_s.lastg, 0, sizeof(vad_s.lastg));
    memset(vad_s.state, 0, sizeof(vad_s.state));

    check_init();
    return 0;
}

static void frame_analysis(kiss_fft_cpx* X, float* Ex, const float* in)
{
    int i;
    float x[WINDOW_SIZE];
    RNN_COPY(x, vad_s.analysis_mem, FRAME_SIZE);
    for (i = 0; i < FRAME_SIZE; i++) x[FRAME_SIZE + i] = in[i];
    RNN_COPY(vad_s.analysis_mem, in, FRAME_SIZE);
    apply_window(x);
    forward_transform(X, x, 1);

    compute_band_energy(Ex, X);
}

float Ly[NB_BANDS];
float p[WINDOW_SIZE];
float pitch_buf[PITCH_BUF_SIZE >> 1];
float tmp[NB_BANDS];

static int compute_frame_features(kiss_fft_cpx* X, kiss_fft_cpx* P,
    float* Ex, float* Ep, float* Exp, float* features, const float* in)
{
    int i;
    float E = 0.0f;
    float* ceps_0, * ceps_1, * ceps_2;
    float spec_variability = 0.0f;

    int pitch_index;
    float gain;
    float* (pre[1]);
    float follow, logMax;
    frame_analysis(X, Ex, in);

    RNN_MOVE(vad_s.pitch_buf, &vad_s.pitch_buf[FRAME_SIZE], PITCH_BUF_SIZE - FRAME_SIZE);
    RNN_COPY(&vad_s.pitch_buf[PITCH_BUF_SIZE - FRAME_SIZE], in, FRAME_SIZE);


    pre[0] = &vad_s.pitch_buf[0];
    rnn_pitch_downsample(pre, pitch_buf, PITCH_BUF_SIZE, 1);
    rnn_pitch_search(pitch_buf + (PITCH_MAX_PERIOD >> 1), pitch_buf, PITCH_FRAME_SIZE,
        PITCH_MAX_PERIOD - 3 * PITCH_MIN_PERIOD, &pitch_index);
    pitch_index = PITCH_MAX_PERIOD - pitch_index;

    gain = rnn_remove_doubling(pitch_buf, PITCH_MAX_PERIOD, PITCH_MIN_PERIOD,
        PITCH_FRAME_SIZE, &pitch_index, vad_s.last_period, vad_s.last_gain);
    vad_s.last_period = pitch_index;
    vad_s.last_gain = gain;
    for (i = 0; i < WINDOW_SIZE; i++)
        p[i] = vad_s.pitch_buf[PITCH_BUF_SIZE - WINDOW_SIZE - pitch_index + i];
    apply_window(p);
    forward_transform(P, p, 0);
    compute_band_energy(Ep, P);
    compute_band_corr(Exp, X, P);
    for (i = 0; i < NB_BANDS; i++) Exp[i] = Exp[i] / sqrtf(.001f + Ex[i] * Ep[i]);
    dct(tmp, Exp);
    for (i = 0; i < NB_DELTA_CEPS; i++) features[NB_BANDS + 2 * NB_DELTA_CEPS + i] = tmp[i];
    features[NB_BANDS + 2 * NB_DELTA_CEPS] -= 1.3f;
    features[NB_BANDS + 2 * NB_DELTA_CEPS + 1] -= 0.9f;
    features[NB_BANDS + 3 * NB_DELTA_CEPS] = .01f * (pitch_index - 300);
    logMax = -2.0f;
    follow = -2.0f;
    for (i = 0; i < NB_BANDS; i++) {
        Ly[i] = fast_log10f(1e-2f + Ex[i]);
        Ly[i] = MAX16(logMax - 7.0f, MAX16(follow - 1.5f, Ly[i]));
        logMax = MAX16(logMax, Ly[i]);
        follow = MAX16(follow - 1.5f, Ly[i]);
        E += Ex[i];
    }
    if (E < 0.04f) {
        /* If there's no audio, avoid messing up the state. */
        RNN_CLEAR(features, NB_FEATURES);
        return 1;
    }
    dct(features, Ly);
    features[0] -= 12.0f;
    features[1] -= 4.0f;
    ceps_0 = vad_s.cepstral_mem[vad_s.memid];
    ceps_1 = (vad_s.memid < 1) ? vad_s.cepstral_mem[CEPS_MEM + vad_s.memid - 1] : vad_s.cepstral_mem[vad_s.memid - 1];
    ceps_2 = (vad_s.memid < 2) ? vad_s.cepstral_mem[CEPS_MEM + vad_s.memid - 2] : vad_s.cepstral_mem[vad_s.memid - 2];
    for (i = 0; i < NB_BANDS; i++) ceps_0[i] = features[i];
    vad_s.memid++;
    for (i = 0; i < NB_DELTA_CEPS; i++) {
        features[i] = ceps_0[i] + ceps_1[i] + ceps_2[i];
        features[NB_BANDS + i] = ceps_0[i] - ceps_2[i];
        features[NB_BANDS + NB_DELTA_CEPS + i] = ceps_0[i] - 2.0f * ceps_1[i] + ceps_2[i];
    }
    /* Spectral variability features. */
    if (vad_s.memid == CEPS_MEM) vad_s.memid = 0;
    for (i = 0; i < CEPS_MEM; i++)
    {
        int j;
        float mindist = 1e15f;
        for (j = 0; j < CEPS_MEM; j++)
        {
            int k;
            float dist = 0;
            for (k = 0; k < NB_BANDS; k++)
            {
                float tmp;
                tmp = vad_s.cepstral_mem[i][k] - vad_s.cepstral_mem[j][k];
                dist += tmp * tmp;
            }
            if (j != i)
                mindist = MIN32(mindist, dist);
        }
        spec_variability += mindist;
    }
    features[NB_BANDS + 3 * NB_DELTA_CEPS + 1] = spec_variability / CEPS_MEM - 2.1f;
    return 0;
}

static void biquad(float* y, float mem[2], const float* x, const float* b, const float* a, int N) {
    int i;
    for (i = 0; i < N; i++) {
        float xi, yi;
        xi = x[i];
        yi = x[i] + mem[0];
        mem[0] = mem[1] + (b[0] * xi - a[0] * yi);
        mem[1] = (b[1] * xi - a[1] * yi);
        y[i] = yi;
    }
}

kiss_fft_cpx X[FREQ_SIZE];
kiss_fft_cpx P[WINDOW_SIZE];

float x[FRAME_SIZE];
float Ex[NB_BANDS], Ep[NB_BANDS];
float Exp[NB_BANDS];
float features[NB_FEATURES];
float g[NB_BANDS];
float gf[FREQ_SIZE] = { 1 };

float rnn_get_vad(const float* in)
{
    int i;
    kiss_fft_cpx X[FREQ_SIZE];
    kiss_fft_cpx P[WINDOW_SIZE];
    float x[FRAME_SIZE];
    float Ex[NB_BANDS], Ep[NB_BANDS];
    float Exp[NB_BANDS];
    float features[NB_FEATURES];
    float vad_prob = 0;
    int silence;
    static const float a_hp[2] = { -1.99599f, 0.99600f };
    static const float b_hp[2] = { -2.0f, 1.0f };
    biquad(x, vad_s.mem_hp_x, in, b_hp, a_hp, FRAME_SIZE);
    silence = compute_frame_features(X, P, Ex, Ep, Exp, features, x);

    if (!silence)
    {
        compute_vad(features, vad_s.state, &vad_prob);
    }

    return vad_prob;
}
