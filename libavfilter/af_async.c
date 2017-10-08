/*
 * Audio Sync Filter
 * Copyright (c) 2017 Sebastien Roy <roys@iro.umontreal.ca>
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Audio Sync Filter
 *
 * Matches audio from multiple sources and compute the time shift between them.
 * Relies on dynamic programming to match the envelope of the signal, so
 * regular audio is suffficient to figure the shifts.
 */

#include "libavutil/attributes.h"
#include "libavutil/audio_fifo.h"
#include "libavutil/avassert.h"
#include "libavutil/avstring.h"
#include "libavutil/channel_layout.h"
#include "libavutil/common.h"
#include "libavutil/float_dsp.h"
#include "libavutil/mathematics.h"
#include "libavutil/opt.h"
#include "libavutil/samplefmt.h"

#include "audio.h"
#include "avfilter.h"
#include "filters.h"
#include "formats.h"
#include "internal.h"

#define INPUT_ON       1    /**< input is active */
#define INPUT_EOF      2    /**< input has reached EOF (may still be active) */

#define DURATION_LONGEST  0
#define DURATION_SHORTEST 1
#define DURATION_FIRST    2


typedef struct FrameInfo {
    int nb_samples;
    int64_t pts;
    struct FrameInfo *next;
} FrameInfo;

/**
 * Linked list used to store timestamps and frame sizes of all frames in the
 * FIFO for the first input.
 *
 * This is needed to keep timestamps synchronized for the case where multiple
 * input frames are pushed to the filter for processing before a frame is
 * requested by the output link.
 */
typedef struct FrameList {
    int nb_frames;
    int nb_samples;
    FrameInfo *list;
    FrameInfo *end;
} FrameList;

static void frame_list_clear(FrameList *frame_list)
{
    if (frame_list) {
        while (frame_list->list) {
            FrameInfo *info = frame_list->list;
            frame_list->list = info->next;
            av_free(info);
        }
        frame_list->nb_frames  = 0;
        frame_list->nb_samples = 0;
        frame_list->end        = NULL;
    }
}

static int frame_list_next_frame_size(FrameList *frame_list)
{
    if (!frame_list->list)
        return 0;
    return frame_list->list->nb_samples;
}

static int64_t frame_list_next_pts(FrameList *frame_list)
{
    if (!frame_list->list)
        return AV_NOPTS_VALUE;
    return frame_list->list->pts;
}

static void frame_list_remove_samples(FrameList *frame_list, int nb_samples)
{
    if (nb_samples >= frame_list->nb_samples) {
        frame_list_clear(frame_list);
    } else {
        int samples = nb_samples;
        while (samples > 0) {
            FrameInfo *info = frame_list->list;
            av_assert0(info);
            if (info->nb_samples <= samples) {
                samples -= info->nb_samples;
                frame_list->list = info->next;
                if (!frame_list->list)
                    frame_list->end = NULL;
                frame_list->nb_frames--;
                frame_list->nb_samples -= info->nb_samples;
                av_free(info);
            } else {
                info->nb_samples       -= samples;
                info->pts              += samples;
                frame_list->nb_samples -= samples;
                samples = 0;
            }
        }
    }
}

static int frame_list_add_frame(FrameList *frame_list, int nb_samples, int64_t pts)
{
    FrameInfo *info = av_malloc(sizeof(*info));
    if (!info)
        return AVERROR(ENOMEM);
    info->nb_samples = nb_samples;
    info->pts        = pts;
    info->next       = NULL;

    if (!frame_list->list) {
        frame_list->list = info;
        frame_list->end  = info;
    } else {
        av_assert0(frame_list->end);
        frame_list->end->next = info;
        frame_list->end       = info;
    }
    frame_list->nb_frames++;
    frame_list->nb_samples += nb_samples;

    return 0;
}

/* FIXME: use directly links fifo */

typedef struct MixContext {
    const AVClass *class;       /**< class for AVOptions */
    AVFloatDSPContext *fdsp;

    int nb_inputs;              /**< number of inputs */
    int active_inputs;          /**< number of input currently active */
    int duration_mode;          /**< mode for determining duration */
    float dropout_transition;   /**< transition time when an input drops out */
    int observe_acc;            /**< observation for offset computation */
    int observe_n;               /** nb of max values to match, for 4800sam, 0.1sec res */
    int observe_done;            /**< done with accumulating samples */
    int observe_reference;      /* input de reference pour le DP */

    float *observe_max;          /** [nbs] current maximum **/
    int *observe_samples;        /** [nbs] samples currently accounted in max **/
    int *observe_cur;            /** [nbs] current value for this input, in dp table */
    float *observe_vals;           /** [n*nbs] values stored, [sample*nbs+in] */
    float *observe_dp;           /** [_n * _n] contain costs */
    int *observe_from;           /** [_n * _n] contain from */

    int nb_channels;            /**< number of channels */
    int sample_rate;            /**< sample rate */
    int planar;
    AVAudioFifo **fifos;        /**< audio fifo for each input */
    uint8_t *input_state;       /**< current state of each input */
    float *input_scale;         /**< mixing scale factor for each input */
    float scale_norm;           /**< normalization factor for all inputs */
    int64_t next_pts;           /**< calculated pts for next output frame */
    FrameList *frame_list;      /**< list of frame info for the first input */
} MixContext;

#define OFFSET(x) offsetof(MixContext, x)
#define A AV_OPT_FLAG_AUDIO_PARAM
#define F AV_OPT_FLAG_FILTERING_PARAM
static const AVOption async_options[] = {
    { "inputs", "Number of inputs.",
            OFFSET(nb_inputs), AV_OPT_TYPE_INT, { .i64 = 2 }, 1, 1024, A|F },
    { "duration", "How to determine the end-of-stream.",
            OFFSET(duration_mode), AV_OPT_TYPE_INT, { .i64 = DURATION_LONGEST }, 0,  2, A|F, "duration" },
        { "longest",  "Duration of longest input.",  0, AV_OPT_TYPE_CONST, { .i64 = DURATION_LONGEST  }, 0, 0, A|F, "duration" },
        { "shortest", "Duration of shortest input.", 0, AV_OPT_TYPE_CONST, { .i64 = DURATION_SHORTEST }, 0, 0, A|F, "duration" },
        { "first",    "Duration of first input.",    0, AV_OPT_TYPE_CONST, { .i64 = DURATION_FIRST    }, 0, 0, A|F, "duration" },
    { "dropout_transition", "Transition time, in seconds, for volume "
                            "renormalization when an input stream ends.",
            OFFSET(dropout_transition), AV_OPT_TYPE_FLOAT, { .dbl = 2.0 }, 0, INT_MAX, A|F },
    { "samples", "Number of samples of observation for offset computation",
            OFFSET(observe_n), AV_OPT_TYPE_INT, { .i64 = 6000 }, 100, 100000, A|F },
    { "precision", "Precision of offset computation, measured in audio samples (4800=0.1sec @ 48khz)",
            OFFSET(observe_acc), AV_OPT_TYPE_INT, { .i64 = 480 }, 1, 100000, A|F },
    { "ref", "Reference input stream for matching",
            OFFSET(observe_reference), AV_OPT_TYPE_INT, { .i64 = 0 }, 0, 1000, A|F },
    { NULL }
};

AVFILTER_DEFINE_CLASS(async);

/**
 * Update the scaling factors to apply to each input during mixing.
 *
 * This balances the full volume range between active inputs and handles
 * volume transitions when EOF is encountered on an input but mixing continues
 * with the remaining inputs.
 */
static void calculate_scales(MixContext *s, int nb_samples)
{
    int i;

    if (s->scale_norm > s->active_inputs) {
        s->scale_norm -= nb_samples / (s->dropout_transition * s->sample_rate);
        s->scale_norm = FFMAX(s->scale_norm, s->active_inputs);
    }

    for (i = 0; i < s->nb_inputs; i++) {
        if (s->input_state[i] & INPUT_ON)
            s->input_scale[i] = 1.0f / s->scale_norm;
        else
            s->input_scale[i] = 0.0f;
    }
}

/**
 * Accumulate info from the streams
 *
 */
static void observe_init(MixContext *s) {
    int i;
    printf("observe inputs=%d duration=%dx%d=%d audio samples, precision=%d audio samples rate=%d\n",s->nb_inputs,
            s->observe_n,s->observe_acc,
            s->observe_n*s->observe_acc,s->observe_acc,
            s->sample_rate);
    printf("observe inputs=%d duration=%.2f s, precision=%d ms rate=%d\n",s->nb_inputs,
            (double)s->observe_n*s->observe_acc/s->sample_rate,
            s->observe_acc*1000/s->sample_rate,
            s->sample_rate);

    s->observe_max=av_mallocz_array(s->nb_inputs, sizeof(float));
    s->observe_samples=av_mallocz_array(s->nb_inputs, sizeof(int));
    s->observe_cur=av_mallocz_array(s->nb_inputs, sizeof(int));

    for(i=0;i<s->nb_inputs;i++) {
        s->observe_max[i]=0.0;
        s->observe_samples[i]=0;
        s->observe_cur[i]=0;
    }

    //s->observe_n=600;  // avec 4800 par max @48kz, 0.1sec par sample donc 600=60sec
    s->observe_dp=av_mallocz_array(s->observe_n*s->observe_n, sizeof(float));
    s->observe_from=av_mallocz_array(s->observe_n*s->observe_n, sizeof(int));

    s->observe_vals=av_mallocz_array(s->observe_n*s->nb_inputs, sizeof(float));
    /** [n*nbs] values stored, [sample*nbs+in] */
    /** les valeurs sont definie jusqua cur[in] **/

    s->observe_done=0;

    printf("allocated %d kbytes\n",(s->observe_n*s->observe_n*8)/1024);
    if( s->observe_dp==NULL || s->observe_from==NULL ) {
        printf("**** UNABLE TO ALLOCATE MEM\n");
    }
}

//
// _dp[b*_n+a] avec a le curr[0] et b le curr[1]
//
// in=0: add value max to dp[(0..cur[1]-1)*_n+(cur[0]-1)]
// in=1: add value max to dp[(cur[1]-1)*_n+(0..cur[0]-1)]
//
// if cur[0] and cur[1] = _n, we are done!
//
//
static void observe_compute_dp(MixContext *s,int a,int b)
{
    int i,j,n,p;
    float *dp=s->observe_dp;
    float occlusion=0.01;
    int *h;
    n=s->observe_n;
    p=s->nb_inputs;
    
    if( s->observe_cur[a]<n || s->observe_cur[b]<n ) {
        printf("... not enough samples for dp!!\n");
        return;
    }

    printf("COMPUTING DP for input %d and %d\n",a,b);
    /*
    for(i=0;i<n;i++) {
        printf("@x %4d %12.6f %12.6f\n",i,
                s->observe_vals[i*p+a],
                s->observe_vals[i*p+b]);
    }
    */

    // rempli le tableau!!
    // cost(i,j) = vals[i,0]-vals[i,1]
    // dp[i,j] = cost(i,j) + min ( dp[i-1,j]+k dp[i,j-1]+k dp[i-1,j-1] )

    for(j=0;j<n;j++) {
        for(i=0;i<n;i++) {
            float cost,k,t;
            int fr;
            cost=fabs(s->observe_vals[i*p+a]-s->observe_vals[j*p+b]);
            if( i==0 && j==0 ) { k=cost;fr=-1; } else { k=99999999999.9;fr=-1; }
            if( i>0 ) {
                t=cost+dp[j*n+i-1]+occlusion; if( t<k ) {k=t;fr=j*n+i-1;}
            }
            if( j>0 ) {
                t=cost+dp[(j-1)*n+i]+occlusion; if( t<k ) {k=t;fr=(j-1)*n+i;}
            }
            if( i>0 && j>0 ) {
                t=cost+dp[(j-1)*n+i-1]+occlusion; if( t<k ) {k=t;fr=(j-1)*n+i-1;}
            }
            dp[j*n+i]=k;
            s->observe_from[j*n+i]=fr;
        }
    }

    /*
    for(j=0;j<n;j++) { for(i=0;i<n;i++) { printf("@@ %f\n", s->observe_dp[j*n+i]); } }
    */

    // le deplacement maximal est +-n, donc 2n+1 [(-n..n)+n]-> 0..2n
    h=(int *)malloc((2*n+1)*sizeof(int));
    for(i=-n;i<=n;i++) h[i+n]=0;

    /* find solution */
    for(i=n-1,j=n-1;;) {
        int v;
        v=s->observe_from[j*n+i];
        //printf("@@@ i=%4d j=%4d (i-j)=%d cost=%12.6f from=%d (%d,%d)\n",i,j,i-j,dp[j*n+i], v,v%n,v/n); fflush(stdout);
        h[(i-j)+n]++;
        if( v<0 ) break;
        i=v%n;
        j=v/n;
    }

    // histo
    //for(i=-n;i<=n;i++) { printf("histo %4d : %6d\n",i,h[i+n]); }

    // le gagnant!!
    {
        int ibest=0;
        for(i=-n;i<=n;i++) {
            if( h[i+n]>h[ibest+n] ) ibest=i;
        }
        printf("solution for input %2d and %2d : delta=%5d = %8d +/- %8d samples\n",a,b,ibest,ibest*s->observe_acc,s->observe_acc/2);
        for(i=-5;i<=5;i++) { printf("histo %4d (%8dms +/- %5dms) : %6d %s\n",i+ibest,(((i+ibest)*s->observe_acc)*1000)/s->sample_rate,(s->observe_acc*1000/2)/s->sample_rate,h[i+ibest+n],i==0?"<====":""); }
    }


    free(h);

    /***
    int from0,to0,from1,to1,i0,i1;
    printf("dp [%d] add value %12.6f, cur=(%d,%d)\n",in,val,s->observe_cur[0],s->observe_cur[1]);
    from0=s->observe_cur[0]-1;
    from1=s->observe_cur[1]-1;
    if( in==0 ) { to0=from0;to1=from1;from1=0; }
    else if( in==1 ) { to1=from1;to0=from0;from0=0; }

    for(i0=from0,i1=from1;i0<=to0 && i1<=to1;i0++,i1++) {
        printf("updating dp(%d,%d) with val=%f\n",i0,i1,val);
    }
    ***/
}


// ajoute des samples de la source in, calcule max et genere calcule dp
static void observe_accumulate(MixContext *s,int in,float *a,int len) {
    int i;
    if( s->observe_done==s->nb_inputs ) return;
    //printf("accumulate in=%d len=%d current=%d samples=%d\n",in,len,s->observe_cur[in],s->observe_samples[in]);
    // pour l' instant seulement in0 et in1
    //if( in>1 ) return;

    // on garde le canal 0 si on est multichannel
    for(i=0;i<len;i+=s->nb_channels) {
        float k=fabs(a[i]); // a[i+channel_number] possible si on veut
        //if( in==0 && s->observe_cur[in]==0 ) { printf("sample %4d = %12.6f\n",i,a[i]); }
        if( s->observe_samples[in]==0 || k>s->observe_max[in] ) s->observe_max[in]=k;
        // one more sample counted!
        s->observe_samples[in]+=1;
        //printf("samp %6d count=%5d\n",i,s->observe_samples[in]);
        if( s->observe_samples[in]==s->observe_acc ) {
            // enough samples?
            if( s->observe_cur[in]<=s->observe_n ) {

                //printf("i=%5d cur=%3d adding in=%d max=%12.6f\n",i,s->observe_cur[in],in,s->observe_max[in]);
                s->observe_samples[in]=0; // reset
                // add row or column (depending on in=0 or 1) at position cur
                // until _cur==_n
                s->observe_vals[s->observe_cur[in]*s->nb_inputs+in]=s->observe_max[in];
                s->observe_cur[in]+=1;

                // on a fini? si oui on va calculer tout
                {
                    int j;
                    s->observe_done=0;
                    for(j=0;j<s->nb_inputs;j++) if( s->observe_cur[j]>=s->observe_n ) s->observe_done++;
                    //printf("NNNNNNNNNb input=%d done=%d cur0=%d cur1=%d\n",s->nb_inputs,s->observe_done,s->observe_cur[0],s->observe_cur[1]);
                    if( s->observe_done==s->nb_inputs ) {
                        // match toutes les sources ensemble
                        /**
                        int a,b;
                        for(a=0;a<s->nb_inputs;a++)
                        for(b=0;b<s->nb_inputs;b++) {
                            if( a!=b ) observe_compute_dp(s,a,b);
                        }
                        **/
                        // match tout avec ref
                        int a;
                        for(a=0;a<s->nb_inputs;a++) {
                            if( a!=s->observe_reference ) observe_compute_dp(s,s->observe_reference,a);
                        }
                        return;
                    }
                }
            }
        }
    }
}


/*******/


static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    MixContext *s      = ctx->priv;
    int i;
    char buf[64];

    s->planar          = av_sample_fmt_is_planar(outlink->format);
    s->sample_rate     = outlink->sample_rate;
    outlink->time_base = (AVRational){ 1, outlink->sample_rate };
    s->next_pts        = AV_NOPTS_VALUE;

    s->frame_list = av_mallocz(sizeof(*s->frame_list));
    if (!s->frame_list)
        return AVERROR(ENOMEM);

    s->fifos = av_mallocz_array(s->nb_inputs, sizeof(*s->fifos));
    if (!s->fifos)
        return AVERROR(ENOMEM);

    s->nb_channels = outlink->channels;
    for (i = 0; i < s->nb_inputs; i++) {
        s->fifos[i] = av_audio_fifo_alloc(outlink->format, s->nb_channels, 1024);
        if (!s->fifos[i])
            return AVERROR(ENOMEM);
    }

    s->input_state = av_malloc(s->nb_inputs);
    if (!s->input_state)
        return AVERROR(ENOMEM);
    memset(s->input_state, INPUT_ON, s->nb_inputs);
    s->active_inputs = s->nb_inputs;

    s->input_scale = av_mallocz_array(s->nb_inputs, sizeof(*s->input_scale));
    if (!s->input_scale)
        return AVERROR(ENOMEM);
    s->scale_norm = s->active_inputs;
    calculate_scales(s, 0);

    av_get_channel_layout_string(buf, sizeof(buf), -1, outlink->channel_layout);

    av_log(ctx, AV_LOG_VERBOSE,
           "inputs:%d fmt:%s srate:%d cl:%s\n", s->nb_inputs,
           av_get_sample_fmt_name(outlink->format), outlink->sample_rate, buf);

    observe_init(s);

    return 0;
}



/**
 * Read samples from the input FIFOs, mix, and write to the output link.
 */
static int output_frame(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    MixContext      *s = ctx->priv;
    AVFrame *out_buf, *in_buf;
    int nb_samples, ns, i;

    if (s->input_state[0] & INPUT_ON) {
        /* first input live: use the corresponding frame size */
        nb_samples = frame_list_next_frame_size(s->frame_list);
        for (i = 1; i < s->nb_inputs; i++) {
            if (s->input_state[i] & INPUT_ON) {
                ns = av_audio_fifo_size(s->fifos[i]);
                if (ns < nb_samples) {
                    if (!(s->input_state[i] & INPUT_EOF))
                        /* unclosed input with not enough samples */
                        return 0;
                    /* closed input to drain */
                    nb_samples = ns;
                }
            }
        }
    } else {
        /* first input closed: use the available samples */
        nb_samples = INT_MAX;
        for (i = 1; i < s->nb_inputs; i++) {
            if (s->input_state[i] & INPUT_ON) {
                ns = av_audio_fifo_size(s->fifos[i]);
                nb_samples = FFMIN(nb_samples, ns);
            }
        }
        if (nb_samples == INT_MAX) {
            ff_outlink_set_status(outlink, AVERROR_EOF, s->next_pts);
            return 0;
        }
    }

    s->next_pts = frame_list_next_pts(s->frame_list);
    frame_list_remove_samples(s->frame_list, nb_samples);

    calculate_scales(s, nb_samples);

    if (nb_samples == 0)
        return 0;

    out_buf = ff_get_audio_buffer(outlink, nb_samples);
    if (!out_buf)
        return AVERROR(ENOMEM);

    in_buf = ff_get_audio_buffer(outlink, nb_samples);
    if (!in_buf) {
        av_frame_free(&out_buf);
        return AVERROR(ENOMEM);
    }

    for (i = 0; i < s->nb_inputs; i++) {
        //printf("input %d\n",i);
        if (s->input_state[i] & INPUT_ON) {
            int planes, plane_size, p;

            av_audio_fifo_read(s->fifos[i], (void **)in_buf->extended_data,
                               nb_samples);

            planes     = s->planar ? s->nb_channels : 1;
            plane_size = nb_samples * (s->planar ? 1 : s->nb_channels);
            plane_size = FFALIGN(plane_size, 16);

            //printf("  in=%d planes=%d size=%d nb_channels=%d planar=%d\n",i,planes,plane_size,s->nb_channels,s->planar);

            if (out_buf->format == AV_SAMPLE_FMT_FLT ||
                out_buf->format == AV_SAMPLE_FMT_FLTP) {
                for (p = 0; p < planes; p++) {
                    observe_accumulate(s,i,(float *) in_buf->extended_data[p], plane_size);
                    s->fdsp->vector_fmac_scalar((float *)out_buf->extended_data[p],
                                                (float *) in_buf->extended_data[p],
                                                s->input_scale[i], plane_size);
                }
            } else {
                for (p = 0; p < planes; p++) {
                    s->fdsp->vector_dmac_scalar((double *)out_buf->extended_data[p],
                                                (double *) in_buf->extended_data[p],
                                                s->input_scale[i], plane_size);
                }
            }
        }
    }
    av_frame_free(&in_buf);

    out_buf->pts = s->next_pts;
    if (s->next_pts != AV_NOPTS_VALUE)
        s->next_pts += nb_samples;

    return ff_filter_frame(outlink, out_buf);
}

/**
 * Requests a frame, if needed, from each input link other than the first.
 */
static int request_samples(AVFilterContext *ctx, int min_samples)
{
    MixContext *s = ctx->priv;
    int i;

    av_assert0(s->nb_inputs > 1);

    for (i = 1; i < s->nb_inputs; i++) {
        if (!(s->input_state[i] & INPUT_ON) ||
             (s->input_state[i] & INPUT_EOF))
            continue;
        if (av_audio_fifo_size(s->fifos[i]) >= min_samples)
            continue;
        ff_inlink_request_frame(ctx->inputs[i]);
    }
    return output_frame(ctx->outputs[0]);
}

/**
 * Calculates the number of active inputs and determines EOF based on the
 * duration option.
 *
 * @return 0 if mixing should continue, or AVERROR_EOF if mixing should stop.
 */
static int calc_active_inputs(MixContext *s)
{
    int i;
    int active_inputs = 0;
    for (i = 0; i < s->nb_inputs; i++)
        active_inputs += !!(s->input_state[i] & INPUT_ON);
    s->active_inputs = active_inputs;

    if (!active_inputs ||
        (s->duration_mode == DURATION_FIRST && !(s->input_state[0] & INPUT_ON)) ||
        (s->duration_mode == DURATION_SHORTEST && active_inputs != s->nb_inputs))
        return AVERROR_EOF;
    return 0;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *outlink = ctx->outputs[0];
    MixContext *s = ctx->priv;
    AVFrame *buf = NULL;
    int i, ret;

    for (i = 0; i < s->nb_inputs; i++) {
        AVFilterLink *inlink = ctx->inputs[i];

        if ((ret = ff_inlink_consume_frame(ctx->inputs[i], &buf)) > 0) {
            if (i == 0) {
                int64_t pts = av_rescale_q(buf->pts, inlink->time_base,
                                           outlink->time_base);
                ret = frame_list_add_frame(s->frame_list, buf->nb_samples, pts);
                if (ret < 0) {
                    av_frame_free(&buf);
                    return ret;
                }
            }

            ret = av_audio_fifo_write(s->fifos[i], (void **)buf->extended_data,
                                      buf->nb_samples);
            if (ret < 0) {
                av_frame_free(&buf);
                return ret;
            }

            av_frame_free(&buf);

            ret = output_frame(outlink);
            if (ret < 0)
                return ret;
        }
    }

    for (i = 0; i < s->nb_inputs; i++) {
        int64_t pts;
        int status;

        if (ff_inlink_acknowledge_status(ctx->inputs[i], &status, &pts)) {
            if (status == AVERROR_EOF) {
                if (i == 0) {
                    s->input_state[i] = 0;
                    if (s->nb_inputs == 1) {
                        ff_outlink_set_status(outlink, status, pts);
                        return 0;
                    }
                } else {
                    s->input_state[i] |= INPUT_EOF;
                    if (av_audio_fifo_size(s->fifos[i]) == 0) {
                        s->input_state[i] = 0;
                    }
                }
            }
        }
    }

    if (calc_active_inputs(s)) {
        ff_outlink_set_status(outlink, AVERROR_EOF, s->next_pts);
        return 0;
    }

    if (ff_outlink_frame_wanted(outlink)) {
        int wanted_samples;

        if (!(s->input_state[0] & INPUT_ON))
            return request_samples(ctx, 1);

        if (s->frame_list->nb_frames == 0) {
            ff_inlink_request_frame(ctx->inputs[0]);
            return 0;
        }
        av_assert0(s->frame_list->nb_frames > 0);

        wanted_samples = frame_list_next_frame_size(s->frame_list);

        return request_samples(ctx, wanted_samples);
    }


    return 0;
}

static av_cold int init(AVFilterContext *ctx)
{
    MixContext *s = ctx->priv;
    int i, ret;

    printf("INIT nbin=%d\n",ctx->nb_inputs);

    for (i = 0; i < s->nb_inputs; i++) {
        char name[32];
        AVFilterPad pad = { 0 };

        snprintf(name, sizeof(name), "input%d", i);
        pad.type           = AVMEDIA_TYPE_AUDIO;
        pad.name           = av_strdup(name);
        if (!pad.name)
            return AVERROR(ENOMEM);

        if ((ret = ff_insert_inpad(ctx, i, &pad)) < 0) {
            av_freep(&pad.name);
            return ret;
        }
    }

    s->fdsp = avpriv_float_dsp_alloc(0);
    if (!s->fdsp)
        return AVERROR(ENOMEM);


    return 0;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    int i;
    MixContext *s = ctx->priv;

    if (s->fifos) {
        for (i = 0; i < s->nb_inputs; i++)
            av_audio_fifo_free(s->fifos[i]);
        av_freep(&s->fifos);
    }
    frame_list_clear(s->frame_list);
    av_freep(&s->frame_list);
    av_freep(&s->input_state);
    av_freep(&s->input_scale);
    av_freep(&s->fdsp);

    for (i = 0; i < ctx->nb_inputs; i++)
        av_freep(&ctx->input_pads[i].name);
}

static int query_formats(AVFilterContext *ctx)
{
    AVFilterFormats *formats = NULL;
    AVFilterChannelLayouts *layouts;
    int ret;

    layouts = ff_all_channel_counts();
    if (!layouts) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }

    if ((ret = ff_add_format(&formats, AV_SAMPLE_FMT_FLT ))          < 0 ||
        (ret = ff_add_format(&formats, AV_SAMPLE_FMT_FLTP))          < 0 ||
        (ret = ff_add_format(&formats, AV_SAMPLE_FMT_DBL ))          < 0 ||
        (ret = ff_add_format(&formats, AV_SAMPLE_FMT_DBLP))          < 0 ||
        (ret = ff_set_common_formats        (ctx, formats))          < 0 ||
        (ret = ff_set_common_channel_layouts(ctx, layouts))          < 0 ||
        (ret = ff_set_common_samplerates(ctx, ff_all_samplerates())) < 0)
        goto fail;
    return 0;
fail:
    if (layouts)
        av_freep(&layouts->channel_layouts);
    av_freep(&layouts);
    return ret;
}

static const AVFilterPad avfilter_af_async_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_AUDIO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_af_async = {
    .name           = "async",
    .description    = NULL_IF_CONFIG_SMALL("Audio mixing."),
    .priv_size      = sizeof(MixContext),
    .priv_class     = &async_class,
    .init           = init,
    .uninit         = uninit,
    .activate       = activate,
    .query_formats  = query_formats,
    .inputs         = NULL,
    .outputs        = avfilter_af_async_outputs,
    .flags          = AVFILTER_FLAG_DYNAMIC_INPUTS,
};
