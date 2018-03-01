/*
 * Copyright (c) 2016 Floris Sluiter
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
 * Pixel remap2 filter
 * This filter copies pixel by pixel a source frame to a target frame.
 * It remaps the pixels to a new x,y destination based on two files ymap/xmap.
 * Map files are passed as a parameter and are in PGM format (P2 or P5),
 * where the values are y(rows)/x(cols) coordinates of the source_frame.
 * The *target* frame dimension is based on mapfile dimensions: specified in the
 * header of the mapfile and reflected in the number of datavalues.
 * Dimensions of ymap and xmap must be equal. Datavalues must be positive or zero.
 * Any datavalue in the ymap or xmap which value is higher
 * then the *source* frame height or width is silently ignored, leaving a
 * blank/chromakey pixel. This can safely be used as a feature to create overlays.
 *
 * Algorithm digest:
 * Target_frame[y][x] = Source_frame[ map[y][x][red] ][ [map[y][x][green] ] * map[y][x][blue];
 */

#include "libavutil/imgutils.h"
#include "libavutil/pixdesc.h"
#include "libavutil/opt.h"
#include "avfilter.h"
#include "formats.h"
#include "framesync.h"
#include "internal.h"
#include "video.h"

typedef struct Remap2Context {
    const AVClass *class;
    int nb_planes;
    int nb_components;
    int step;
    FFFrameSync fs;
    float cx;
    float cy;
    float radius;

    void (*remap)(struct Remap2Context *s, const AVFrame *in, const AVFrame *map, AVFrame *out);
} Remap2Context;


#define OFFSET(x) offsetof(Remap2Context, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM|AV_OPT_FLAG_VIDEO_PARAM

static const AVOption remap2_options[] = {
    { "cx", "center x offset", OFFSET(cx), AV_OPT_TYPE_FLOAT, {.dbl=-1}, -1, 10000, FLAGS},
    { "cy", "center y offset", OFFSET(cy), AV_OPT_TYPE_FLOAT, {.dbl=-1}, -1, 10000, FLAGS},
    { "r", "radius", OFFSET(radius), AV_OPT_TYPE_FLOAT, {.dbl=-1}, -1, 10000, FLAGS},
    { NULL }
};
AVFILTER_DEFINE_CLASS(remap2);


static int query_formats(AVFilterContext *ctx)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_YUVA444P,
        AV_PIX_FMT_YUV444P,
        AV_PIX_FMT_YUVJ444P,
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
        AV_PIX_FMT_ARGB, AV_PIX_FMT_ABGR, AV_PIX_FMT_RGBA, AV_PIX_FMT_BGRA,
        AV_PIX_FMT_GBRP, AV_PIX_FMT_GBRAP,
        AV_PIX_FMT_YUV444P9, AV_PIX_FMT_YUV444P10, AV_PIX_FMT_YUV444P12,
        AV_PIX_FMT_YUV444P14, AV_PIX_FMT_YUV444P16,
        AV_PIX_FMT_YUVA444P9, AV_PIX_FMT_YUVA444P10, AV_PIX_FMT_YUVA444P16,
        AV_PIX_FMT_GBRP9, AV_PIX_FMT_GBRP10, AV_PIX_FMT_GBRP12,
        AV_PIX_FMT_GBRP14, AV_PIX_FMT_GBRP16,
        AV_PIX_FMT_GBRAP10, AV_PIX_FMT_GBRAP12, AV_PIX_FMT_GBRAP16,
        AV_PIX_FMT_RGB48, AV_PIX_FMT_BGR48,
        AV_PIX_FMT_RGBA64, AV_PIX_FMT_BGRA64,
        AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAY9,
        AV_PIX_FMT_GRAY10, AV_PIX_FMT_GRAY12,
        AV_PIX_FMT_GRAY16,
        AV_PIX_FMT_NONE
    };
    static const enum AVPixelFormat map_fmts[] = {
        AV_PIX_FMT_RGB48LE,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *pix_formats = NULL, *map_formats = NULL;
    int ret;

    if (!(pix_formats = ff_make_format_list(pix_fmts)) ||
        !(map_formats = ff_make_format_list(map_fmts))) {
        ret = AVERROR(ENOMEM);
        goto fail;
    }
    if ((ret = ff_formats_ref(pix_formats, &ctx->inputs[0]->out_formats)) < 0 ||
        (ret = ff_formats_ref(map_formats, &ctx->inputs[1]->out_formats)) < 0 ||
        (ret = ff_formats_ref(pix_formats, &ctx->outputs[0]->in_formats)) < 0)
        goto fail;
    return 0;
fail:
    printf("FAIL!\n");
    if (pix_formats)
        av_freep(&pix_formats->formats);
    av_freep(&pix_formats);
    if (map_formats)
        av_freep(&map_formats->formats);
    av_freep(&map_formats);
    return ret;
}

/**
 * remap_planar algorithm expects planes of same size
 * pixels are copied from source to target using :
 * Target_frame[y][x] = Source_frame[ map[y][x] ][ [map[y][x] ];
 */
static void remap2_planar_interpolate(Remap2Context *s, const AVFrame *in,
                         const AVFrame *map, AVFrame *out)
{
    const int linesize = map->linesize[0]/2;
    int x , y, plane;

    int def=1;
    if( s->cx<0 && s->cy<0 && s->radius<0 ) def=0;
    if( def ) {
        def=0;
        if( s->cx<0 ) {def=1;s->cx=in->width/2.0;}
        if( s->cy<0 ) {def=1;s->cy=in->height/2.0;}
        if( s->radius<0 ) {def=1;s->radius=(in->width<in->height?in->width:in->height)/2.0;}
        if( def ) printf("*** Parametres par defaut: cx=%5f  cy=%5f  r=%5f\n\n",s->cx,s->cy,s->radius);
    }


    //printf("nbplane=%d linesize=%d xmap=0x%08lx\n",s->nb_planes,linesize,xmap);

    //printf("** map nbplanes=0x%08lx\n",map->format);

    for (plane = 0; plane < s->nb_planes ; plane++) {
        uint8_t *dst         = out->data[plane];
        const int dlinesize  = out->linesize[plane];
        const uint8_t *src   = in->data[plane];
        const int slinesize  = in->linesize[plane];

        //printf("plane %d has linesize s=%d d=%d\n", plane,slinesize,dlinesize);

        const uint16_t *xmap = (const uint16_t *)map->data[0];
        // ymap=xmap+1, bmap=xmap+2, format packed

        int a00,a10,a01,a11;

        for (y = 0; y < out->height; y++) {
            const uint16_t *q=xmap;
            for (x = 0; x < out->width; x++,q+=3) {
                int ipx,ipy;
                float fpx,fpy;
                float px,py;
                //if( xmap[x]==0 || ymap[x]==0 ) { dst[x]=0;continue; }
                // subpixel position
                if( def ) {
                    // on utilise les parametres fisheye cx,cy,radius
                    px=(float)q[0]/65536.0*(2.0*s->radius)+s->cx-s->radius;
                    py=(float)q[1]/65536.0*(2.0*s->radius)+s->cy-s->radius;
                }else{
                    // remap normal 0..1 pour toute la largeur ou la hauteur
                    px=(float)q[0]/65536.0*in->width;
                    py=(float)q[1]/65536.0*in->height;
                }
                float pm=(float)q[2]/65536.0; // mask: 0 a 1
                //printf("xy=%4d,%4d -> (%12.6f, %12.6f) inw=%4d inh=%4d\n",x,y,px,py,in->width,in->height);
                //printf("xy=%4d,%4d -> (%6d,%6d,%6d) inw=%4d inh=%4d ls=%d\n",x,y,q[0],q[1],q[2],in->width,in->height,linesize);
                // a cause de cx,cy, on peut avoir des points en dehors de l'image
                if( px<0 ) px=0; else if( px>=in->width ) px=in->width-1;
                if( py<0 ) py=0; else if( py>=in->height ) py=in->height-1;
               
                // test si le masque est trop haut, alors, on donne du noir
                if( pm>0.15 ) {
                    dst[x]=(plane==0?0:128);
                    continue;
                }
               
                  
                //
                //float pb=(float)q[2]; // mask
                // integer part
                ipx=(int)px;
                ipy=(int)py;
                // fractional part
                fpx=px-ipx;
                fpy=py-ipy;
                // interpolation bilineaire
                // AUCUN CHECK de depassement!!!
                a00=a01=a10=a11=src[ipy*slinesize+ipx];
                if( ipx+1<in->width ) {
                    a10=src[ipy*slinesize+ipx+1];
                    if( ipy+1<in->height ) {
                        a11=src[(ipy+1)*slinesize+ipx+1];
                    }
                }
                if( ipy+1<in->height ) {
                    a01=src[(ipy+1)*slinesize+ipx];
                }
                //printf("%4d %4d %4d %4d : %4d %4d %4.3f %4.3f\n",a00,a10,a01,a11,ipx,ipy,fpx,fpy);
                dst[x] = a00*(1-fpx)*(1-fpy)+a10*fpx*(1-fpy)+a01*(1-fpx)*fpy+a11*fpx*fpy;
            }
            dst  += dlinesize;
            xmap += linesize;
        }
    }
}




static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    Remap2Context *s = ctx->priv;
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    printf("***** config input!!\n");

    s->nb_planes = av_pix_fmt_count_planes(inlink->format);
    s->nb_components = desc->nb_components;


    s->remap=NULL;
    if (desc->comp[0].depth == 8) {
        if (s->nb_planes > 1 || s->nb_components == 1) {
                s->remap=remap2_planar_interpolate;
        }
    }

    s->step = av_get_padded_bits_per_pixel(desc) >> 3;
    printf("*** depth=%d nbplanes=%d nbcomponents=%d padded_bpp=%d ***\n",
            desc->comp[0].depth,
            s->nb_planes,
            s->nb_components,
            av_get_padded_bits_per_pixel(desc));
    return 0;
}

static int process_frame(FFFrameSync *fs)
{
    AVFilterContext *ctx = fs->parent;
    Remap2Context *s = fs->opaque;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out, *in, *map;
    int ret;
    int k;


    if ((ret = ff_framesync_get_frame(&s->fs, 0, &in,   0)) < 0 ) {
        printf("ERR1 %d\n",ret);
        return ret;
    }
    if ((ret = ff_framesync_get_frame(&s->fs, 1, &map, 0)) < 0 ) {
        printf("ERR2 %d\n",ret);
        return ret;
    }

    if (ctx->is_disabled) {
        out = av_frame_clone(in);
        if (!out)
            return AVERROR(ENOMEM);
    } else {
        out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
        if (!out)
            return AVERROR(ENOMEM);
        av_frame_copy_props(out, in);

        s->remap(s, in, map, out);
    }
    out->pts = av_rescale_q(in->pts, s->fs.time_base, outlink->time_base);

    k=ff_filter_frame(outlink, out);
    return k;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    Remap2Context *s = ctx->priv;
    AVFilterLink *srclink = ctx->inputs[0];
    AVFilterLink *maplink = ctx->inputs[1];
    FFFrameSyncIn *in;
    int ret;
    int k;

    printf("configuring output...\n");

    outlink->w = maplink->w;
    outlink->h = maplink->h;
    outlink->time_base = srclink->time_base;
    outlink->sample_aspect_ratio = srclink->sample_aspect_ratio;
    outlink->frame_rate = srclink->frame_rate;

    ret = ff_framesync_init(&s->fs, ctx, 2);
    printf("config output init %d\n",ret);
    if (ret < 0)
        return ret;

    in = s->fs.in;
    in[0].time_base = srclink->time_base;
    in[1].time_base = maplink->time_base;
    in[0].sync   = 2;
    in[0].before = EXT_STOP;
    in[0].after  = EXT_STOP;
    in[1].sync   = 1;
    in[1].before = EXT_NULL;
    in[1].after  = EXT_INFINITY;
    s->fs.opaque   = s;
    s->fs.on_event = process_frame;

    k=ff_framesync_configure(&s->fs);
    printf("done k=%d\n",k);
    return k;
}

static int activate(AVFilterContext *ctx)
{
    int k;
    Remap2Context *s = ctx->priv;
    k=ff_framesync_activate(&s->fs);
    return k;
}


static av_cold void uninit(AVFilterContext *ctx)
{
    Remap2Context *s = ctx->priv;
    printf("uninit...\n");

    ff_framesync_uninit(&s->fs);
    printf("uninit done\n");
}

static const AVFilterPad remap2_inputs[] = {
    {
        .name         = "source",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
    {
        .name         = "map",
        .type         = AVMEDIA_TYPE_VIDEO,
    },
    { NULL }
};

static const AVFilterPad remap2_outputs[] = {
    {
        .name          = "default",
        .type          = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_remap2 = {
    .name          = "remap2",
    .description   = NULL_IF_CONFIG_SMALL("Remap pixels using RGB LUT."),
    .priv_size     = sizeof(Remap2Context),
    .uninit        = uninit,
    .query_formats = query_formats,
    .activate      = activate,
    .inputs        = remap2_inputs,
    .outputs       = remap2_outputs,
    .priv_class    = &remap2_class,
    .flags         = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
