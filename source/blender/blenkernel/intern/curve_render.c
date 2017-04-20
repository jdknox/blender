/*
 * ***** BEGIN GPL LICENSE BLOCK *****
 *
 * This program is free software; you can redistribute it and/or
 * modify it under the terms of the GNU General Public License
 * as published by the Free Software Foundation; either version 2
 * of the License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
 *
 * The Original Code is Copyright (C) 2017 by Blender Foundation.
 * All rights reserved.
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/blenkernel/intern/curve_render.c
 *  \ingroup bke
 *
 * \brief Curve API for render engines
 */

#include "MEM_guardedalloc.h"

#include "BLI_utildefines.h"
#include "BLI_math_vector.h"

#include "DNA_curve_types.h"

#include "BKE_curve.h"
#include "BKE_curve_render.h"

#include "GPU_batch.h"

#define SELECT   1

/**
 * TODO
 * - Ensure `CurveCache`, `SEQUENCER_DAG_WORKAROUND`.
 * - Check number of verts/edges to see if cache is valid.
 * - Check if 'overlay.edges' can use single attribyte per edge, not 2 (for selection drawing).
 */

/* ---------------------------------------------------------------------- */
/* Curve Interface, direct access to basic data. */

static void curve_render_overlay_verts_edges_len_get(
        ListBase *lb, bool hide_handles,
        int *r_vert_len, int *r_edge_len)
{
	BLI_assert(r_vert_len || r_edge_len);
	int vert_len = 0;
	int edge_len = 0;
	for (Nurb *nu = lb->first; nu; nu = nu->next) {
		if (nu->bezt) {
			vert_len += hide_handles ? nu->pntsu : (nu->pntsu * 3);
			/* 2x handles per point*/
			edge_len += 2 * nu->pntsu;
		}
		else if (nu->bp) {
			vert_len += nu->pntsu;
			/* segments between points */
			edge_len += nu->pntsu - 1;
		}
	}
	if (r_vert_len) {
		*r_vert_len = vert_len;
	}
	if (r_edge_len) {
		*r_edge_len = edge_len;
	}
}

static void curve_render_wire_verts_edges_len_get(
        const CurveCache *ob_curve_cache,
        int *r_vert_len, int *r_edge_len)
{
	BLI_assert(r_vert_len || r_edge_len);
	int vert_len = 0;
	int edge_len = 0;
	for (const BevList *bl = ob_curve_cache->bev.first; bl; bl = bl->next) {
		if (bl->nr > 0) {
			const bool is_cyclic = bl->poly != -1;

			/* verts */
			vert_len += bl->nr;

			/* edges */
			edge_len += bl->nr;
			if (!is_cyclic) {
				edge_len -= 1;
			}
		}
	}
	if (r_vert_len) {
		*r_vert_len = vert_len;
	}
	if (r_edge_len) {
		*r_edge_len = edge_len;
	}
}

static int curve_render_normal_len_get(const ListBase *lb, const CurveCache *ob_curve_cache)
{
	int normal_len = 0;
	const BevList *bl;
	const Nurb *nu;
	for (bl = ob_curve_cache->bev.first, nu = lb->first; nu && bl; bl = bl->next, nu = nu->next) {
		int nr = bl->nr;
		int skip = nu->resolu / 16;
#if 0
		while (nr-- > 0) { /* accounts for empty bevel lists */
			normal_len += 1;
			nr -= skip;
		}
#else
		normal_len += max_ii((nr + max_ii(skip - 1, 0)) / (skip + 1), 0);
#endif
	}
	return normal_len;
}

/* ---------------------------------------------------------------------- */
/* Curve Interface, indirect, partially cached access to complex data. */

typedef struct CurveRenderData {
	int types;

	struct {
		int vert_len;
		int edge_len;
	} overlay;

	struct {
		int vert_len;
		int edge_len;
	} wire;

	/* edit mode normal's */
	struct {
		/* 'edge_len == len * 2'
		 * 'vert_len == len * 3' */
		int len;
	} normal;

	bool hide_handles;
	bool hide_normals;

	/* borrow from 'Object' */
	CurveCache *ob_curve_cache;

	/* borrow from 'Curve' */
	struct EditNurb *edit_latt;
	ListBase *nurbs;

	/* edit, index in nurb list */
	int actnu;
	/* edit, index in active nurb (BPoint or BezTriple) */
	int actvert;
} CurveRenderData;

enum {
	/* Wire center-line */
	CU_DATATYPE_WIRE        = 1 << 0,
	/* Edit-mode verts and optionally handles */
	CU_DATATYPE_OVERLAY     = 1 << 1,
	/* Edit-mode normals */
	CU_DATATYPE_NORMAL      = 1 << 2,
};

/*
 * ob_curve_cache can be NULL, only needed for CU_DATATYPE_WIRE
 */
static CurveRenderData *curve_render_data_create(Curve *cu, CurveCache *ob_curve_cache, const int types)
{
	CurveRenderData *rdata = MEM_callocN(sizeof(*rdata), __func__);
	rdata->types = types;
	ListBase *nurbs;

	rdata->hide_handles = (cu->drawflag & CU_HIDE_HANDLES) != 0;
	rdata->hide_normals = (cu->drawflag & CU_HIDE_NORMALS) != 0;

	rdata->actnu = cu->actnu;
	rdata->actvert = cu->actvert;

	rdata->ob_curve_cache = ob_curve_cache;

	if (types & CU_DATATYPE_WIRE) {
		curve_render_wire_verts_edges_len_get(
		        rdata->ob_curve_cache,
		        &rdata->wire.vert_len, &rdata->wire.edge_len);
	}

	if (cu->editnurb) {
		EditNurb *editnurb = cu->editnurb;
		nurbs = &editnurb->nurbs;

		rdata->edit_latt = editnurb;

		if (types & CU_DATATYPE_OVERLAY) {
			curve_render_overlay_verts_edges_len_get(
			        nurbs, rdata->hide_handles,
			        &rdata->overlay.vert_len,
			        rdata->hide_handles ? NULL : &rdata->overlay.edge_len);

			rdata->actnu = cu->actnu;
			rdata->actvert = cu->actvert;
		}
		if (types & CU_DATATYPE_NORMAL) {
			rdata->normal.len = curve_render_normal_len_get(nurbs, rdata->ob_curve_cache);
		}
	}
	else {
		nurbs = &cu->nurb;
	}

	rdata->nurbs = nurbs;

	return rdata;
}

static void curve_render_data_free(CurveRenderData *rdata)
{
#if 0
	if (rdata->loose_verts) {
		MEM_freeN(rdata->loose_verts);
	}
#endif
	MEM_freeN(rdata);
}

static int curve_render_data_overlay_verts_len_get(const CurveRenderData *rdata)
{
	BLI_assert(rdata->types & CU_DATATYPE_OVERLAY);
	return rdata->overlay.vert_len;
}

static int curve_render_data_overlay_edges_len_get(const CurveRenderData *rdata)
{
	BLI_assert(rdata->types & CU_DATATYPE_OVERLAY);
	return rdata->overlay.edge_len;
}

static int curve_render_data_wire_verts_len_get(const CurveRenderData *rdata)
{
	BLI_assert(rdata->types & CU_DATATYPE_WIRE);
	return rdata->wire.vert_len;
}

static int curve_render_data_wire_edges_len_get(const CurveRenderData *rdata)
{
	BLI_assert(rdata->types & CU_DATATYPE_WIRE);
	return rdata->wire.edge_len;
}

static int curve_render_data_normal_len_get(const CurveRenderData *rdata)
{
	BLI_assert(rdata->types & CU_DATATYPE_NORMAL);
	return rdata->normal.len;
}

enum {
	VFLAG_VERTEX_SELECTED = 1 << 0,
	VFLAG_VERTEX_ACTIVE   = 1 << 1,
};

/* ---------------------------------------------------------------------- */
/* Curve Batch Cache */

typedef struct CurveBatchCache {
	/* center-line */
	struct {
		VertexBuffer *verts;
		VertexBuffer *edges;
		Batch *batch;
		ElementList *elem;
	} wire;

	/* normals */
	struct {
		VertexBuffer *verts;
		VertexBuffer *edges;
		Batch *batch;
		ElementList *elem;
	} normal;

	/* control handles and vertices */
	struct {
		Batch *edges;
		Batch *verts;
	} overlay;

	/* settings to determine if cache is invalid */
	bool is_dirty;

	bool hide_handles;
	bool hide_normals;

	float normal_size;

	bool is_editmode;
} CurveBatchCache;

/* Batch cache management. */

static bool curve_batch_cache_valid(Curve *cu)
{
	CurveBatchCache *cache = cu->batch_cache;

	if (cache == NULL) {
		return false;
	}

	if (cache->is_editmode != (cu->editnurb != NULL)) {
		return false;
	}

	if (cache->is_editmode) {
		if ((cache->hide_handles != ((cu->drawflag & CU_HIDE_HANDLES) != 0))) {
			return false;
		}
		else if ((cache->hide_normals != ((cu->drawflag & CU_HIDE_NORMALS) != 0))) {
			return false;
		}
	}

	if (cache->is_dirty == false) {
		return true;
	}
	else {
		/* TODO: check number of vertices/edges? */
		if (cache->is_editmode) {
			return false;
		}
	}

	return true;
}

static void curve_batch_cache_init(Curve *cu)
{
	CurveBatchCache *cache = cu->batch_cache;

	if (!cache) {
		cache = cu->batch_cache = MEM_callocN(sizeof(*cache), __func__);
	}
	else {
		memset(cache, 0, sizeof(*cache));
	}

	cache->hide_handles = (cu->drawflag & CU_HIDE_HANDLES) != 0;
	cache->hide_normals = (cu->drawflag & CU_HIDE_NORMALS) != 0;

#if 0
	ListBase *nurbs;
	if (cu->editnurb) {
		EditNurb *editnurb = cu->editnurb;
		nurbs = &editnurb->nurbs;
	}
	else {
		nurbs = &cu->nurb;
	}
#endif

	cache->is_editmode = cu->editnurb != NULL;

	cache->is_dirty = false;
}

static CurveBatchCache *curve_batch_cache_get(Curve *cu)
{
	if (!curve_batch_cache_valid(cu)) {
		BKE_curve_batch_cache_clear(cu);
		curve_batch_cache_init(cu);
	}
	return cu->batch_cache;
}

void BKE_curve_batch_cache_dirty(Curve *cu)
{
	CurveBatchCache *cache = cu->batch_cache;
	if (cache) {
		cache->is_dirty = true;
	}
}

void BKE_curve_batch_selection_dirty(Curve *cu)
{
	CurveBatchCache *cache = cu->batch_cache;
	if (cache) {
		BATCH_DISCARD_ALL_SAFE(cache->overlay.verts);
		BATCH_DISCARD_ALL_SAFE(cache->overlay.edges);
	}
}

void BKE_curve_batch_cache_clear(Curve *cu)
{
	CurveBatchCache *cache = cu->batch_cache;
	if (!cache) {
		return;
	}

	BATCH_DISCARD_ALL_SAFE(cache->overlay.verts);
	BATCH_DISCARD_ALL_SAFE(cache->overlay.edges);

	if (cache->wire.batch) {
		BATCH_DISCARD_ALL_SAFE(cache->wire.batch);
		cache->wire.verts = NULL;
		cache->wire.edges = NULL;
		cache->wire.elem = NULL;
	}
	else {
		VERTEXBUFFER_DISCARD_SAFE(cache->wire.verts);
		VERTEXBUFFER_DISCARD_SAFE(cache->wire.edges);
		ELEMENTLIST_DISCARD_SAFE(cache->wire.elem);
	}

	if (cache->normal.batch) {
		BATCH_DISCARD_ALL_SAFE(cache->normal.batch);
		cache->normal.verts = NULL;
		cache->normal.edges = NULL;
		cache->normal.elem = NULL;
	}
	else {
		VERTEXBUFFER_DISCARD_SAFE(cache->normal.verts);
		VERTEXBUFFER_DISCARD_SAFE(cache->normal.edges);
		ELEMENTLIST_DISCARD_SAFE(cache->normal.elem);
	}
}

void BKE_curve_batch_cache_free(Curve *cu)
{
	BKE_curve_batch_cache_clear(cu);
	MEM_SAFE_FREE(cu->batch_cache);
}

/* Batch cache usage. */
static VertexBuffer *curve_batch_cache_get_wire_verts(CurveRenderData *rdata, CurveBatchCache *cache)
{
	BLI_assert(rdata->types & CU_DATATYPE_WIRE);
	BLI_assert(rdata->ob_curve_cache != NULL);

	if (cache->wire.verts == NULL) {
		static VertexFormat format = { 0 };
		static unsigned pos_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
		}

		const int vert_len = curve_render_data_wire_verts_len_get(rdata);

		VertexBuffer *vbo = cache->wire.verts = VertexBuffer_create_with_format(&format);
		VertexBuffer_allocate_data(vbo, vert_len);
		int vbo_len_used = 0;
		for (const BevList *bl = rdata->ob_curve_cache->bev.first; bl; bl = bl->next) {
			if (bl->nr > 0) {
				const int i_end = vbo_len_used + bl->nr;
				for (const BevPoint *bevp = bl->bevpoints; vbo_len_used < i_end; vbo_len_used++, bevp++) {
					VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bevp->vec);
				}
			}
		}
		BLI_assert(vbo_len_used == vert_len);
	}

	return cache->wire.verts;
}

static ElementList *curve_batch_cache_get_wire_edges(CurveRenderData *rdata, CurveBatchCache *cache)
{
	BLI_assert(rdata->types & CU_DATATYPE_WIRE);
	BLI_assert(rdata->ob_curve_cache != NULL);

	if (cache->wire.edges == NULL) {
		const int vert_len = curve_render_data_wire_verts_len_get(rdata);
		const int edge_len = curve_render_data_wire_edges_len_get(rdata);
		int edge_len_used = 0;

		ElementListBuilder elb;
		ElementListBuilder_init(&elb, PRIM_LINES, edge_len, vert_len);

		int i = 0;
		for (const BevList *bl = rdata->ob_curve_cache->bev.first; bl; bl = bl->next) {
			if (bl->nr > 0) {
				const bool is_cyclic = bl->poly != -1;
				const int i_end = i + (bl->nr);
				int i_prev;
				if (is_cyclic) {
					i_prev = i + (bl->nr - 1);
				}
				else {
					i_prev = i;
					i += 1;
				}
				for (; i < i_end; i_prev = i++) {
					add_line_vertices(&elb, i_prev, i);
					edge_len_used += 1;
				}
			}
		}

		if (rdata->hide_handles) {
			BLI_assert(edge_len_used <= edge_len);
		}
		else {
			BLI_assert(edge_len_used == edge_len);
		}

		cache->wire.elem = ElementList_build(&elb);
	}

	return cache->wire.elem;
}

static VertexBuffer *curve_batch_cache_get_normal_verts(CurveRenderData *rdata, CurveBatchCache *cache)
{
	BLI_assert(rdata->types & CU_DATATYPE_NORMAL);
	BLI_assert(rdata->ob_curve_cache != NULL);

	if (cache->normal.verts == NULL) {
		static VertexFormat format = { 0 };
		static unsigned pos_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
		}

		const int normal_len = curve_render_data_normal_len_get(rdata);
		const int vert_len = normal_len * 3;

		VertexBuffer *vbo = cache->normal.verts = VertexBuffer_create_with_format(&format);
		VertexBuffer_allocate_data(vbo, vert_len);
		int vbo_len_used = 0;

		const BevList *bl;
		const Nurb *nu;

		for (bl = rdata->ob_curve_cache->bev.first, nu = rdata->nurbs->first;
		     nu && bl;
		     bl = bl->next, nu = nu->next)
		{
			const BevPoint *bevp = bl->bevpoints;
			int nr = bl->nr;
			int skip = nu->resolu / 16;

			while (nr-- > 0) { /* accounts for empty bevel lists */
				const float fac = bevp->radius * cache->normal_size;
				float vec_a[3]; /* Offset perpendicular to the curve */
				float vec_b[3]; /* Delta along the curve */

				vec_a[0] = fac;
				vec_a[1] = 0.0f;
				vec_a[2] = 0.0f;

				mul_qt_v3(bevp->quat, vec_a);
				madd_v3_v3fl(vec_a, bevp->dir, -fac);

				reflect_v3_v3v3(vec_b, vec_a, bevp->dir);
				negate_v3(vec_b);

				add_v3_v3(vec_a, bevp->vec);
				add_v3_v3(vec_b, bevp->vec);

				VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used++, vec_a);
				VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used++, bevp->vec);
				VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used++, vec_b);

				bevp += skip + 1;
				nr -= skip;
			}
		}
		BLI_assert(vbo_len_used == vert_len);
	}

	return cache->normal.verts;
}

static ElementList *curve_batch_cache_get_normal_edges(CurveRenderData *rdata, CurveBatchCache *cache)
{
	BLI_assert(rdata->types & CU_DATATYPE_NORMAL);
	BLI_assert(rdata->ob_curve_cache != NULL);

	if (cache->normal.edges == NULL) {
		const int normal_len = curve_render_data_normal_len_get(rdata);
		const int edge_len = normal_len * 2;

		ElementListBuilder elb;
		ElementListBuilder_init(&elb, PRIM_LINES, edge_len, normal_len * 2);

		int vbo_len_used = 0;
		for (int i = 0; i < normal_len; i++) {
			add_line_vertices(&elb, vbo_len_used + 0, vbo_len_used + 1);
			add_line_vertices(&elb, vbo_len_used + 1, vbo_len_used + 2);
			vbo_len_used += 3;
		}

		BLI_assert(vbo_len_used == normal_len * 3);

		cache->normal.elem = ElementList_build(&elb);
	}

	return cache->normal.elem;
}

static void curve_batch_cache_create_overlay_batches(Curve *cu)
{
	/* Since CU_DATATYPE_OVERLAY is slow to generate, generate them all at once */
	int options = CU_DATATYPE_OVERLAY;

	CurveBatchCache *cache = curve_batch_cache_get(cu);
	CurveRenderData *rdata = curve_render_data_create(cu, NULL, options);

	if (cache->overlay.verts == NULL) {
		static VertexFormat format = { 0 };
		static unsigned pos_id, data_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
			data_id = VertexFormat_add_attrib(&format, "data", COMP_U8, 1, KEEP_INT);
		}

		VertexBuffer *vbo = VertexBuffer_create_with_format(&format);
		const int vbo_len_capacity = curve_render_data_overlay_verts_len_get(rdata);
		int vbo_len_used = 0;
		VertexBuffer_allocate_data(vbo, vbo_len_capacity);
		int i = 0;
		for (Nurb *nu = rdata->nurbs->first; nu; nu = nu->next) {
			if (nu->bezt) {
				int a = 0;
				for (const BezTriple *bezt = nu->bezt; a < nu->pntsu; a++, bezt++) {
					if (bezt->hide == false) {
						const bool is_active = (i == rdata->actvert);
						char vflag;

						if (rdata->hide_handles) {
							vflag = (bezt->f2 & SELECT) ?
							        (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
							VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bezt->vec[1]);
							VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
							vbo_len_used += 1;
						}
						else {
							for (int j = 0; j < 3; j++) {
								vflag = ((&bezt->f1)[j] & SELECT) ?
								        (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
								VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bezt->vec[j]);
								VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
								vbo_len_used += 1;
							}
						}
					}
					i += 1;
				}
			}
			else if (nu->bp) {
				int a = 0;
				for (const BPoint *bp = nu->bp; a < nu->pntsu; a++, bp++) {
					if (bp->hide == false) {
						const bool is_active = (i == rdata->actvert);
						char vflag;
						vflag = (bp->f1 & SELECT) ? (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
						VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bp->vec);
						VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
						vbo_len_used += 1;
					}
					i += 1;
				}
			}
			i += nu->pntsu;
		}
		if (vbo_len_capacity != vbo_len_used) {
			VertexBuffer_resize_data(vbo, vbo_len_used);
		}

		cache->overlay.verts = Batch_create(PRIM_POINTS, vbo, NULL);
	}


	if ((cache->overlay.edges == NULL) && (rdata->hide_handles == false)) {
		/* Note: we could reference indices to vertices (above) */

		static VertexFormat format = { 0 };
		static unsigned pos_id, data_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
			data_id = VertexFormat_add_attrib(&format, "data", COMP_U8, 1, KEEP_INT);
		}

		VertexBuffer *vbo = VertexBuffer_create_with_format(&format);
		const int edge_len =  curve_render_data_overlay_edges_len_get(rdata);
		const int vbo_len_capacity = edge_len * 2;
		int vbo_len_used = 0;
		VertexBuffer_allocate_data(vbo, vbo_len_capacity);
		int i = 0;
		for (Nurb *nu = rdata->nurbs->first; nu; nu = nu->next) {
			if (nu->bezt) {
				int a = 0;
				for (const BezTriple *bezt = nu->bezt; a < nu->pntsu; a++, bezt++) {
					if (bezt->hide == false) {
						const bool is_active = (i == rdata->actvert);
						char vflag;

						vflag = (bezt->f1 & SELECT) ? (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
						VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bezt->vec[0]);
						VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
						vbo_len_used += 1;

						/* same vertex twice, only check different selection */
						for (int j = 0; j < 2; j++) {
							vflag = ((j ? bezt->f3 : bezt->f1) & SELECT) ?
							        (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
							VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bezt->vec[1]);
							VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
							vbo_len_used += 1;
						}

						vflag = (bezt->f3 & SELECT) ? (is_active ? VFLAG_VERTEX_ACTIVE : VFLAG_VERTEX_SELECTED) : 0;
						VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bezt->vec[2]);
						VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
						vbo_len_used += 1;
					}
					i += 1;
				}
			}
			else if (nu->bp) {
				int a = 1;
				for (const BPoint *bp_prev = nu->bp, *bp_curr = &nu->bp[1]; a < nu->pntsu; a++, bp_prev = bp_curr++) {
					if ((bp_prev->hide == false) && (bp_curr->hide == false)) {
						char vflag;
						vflag = ((bp_prev->f1 & SELECT) && (bp_curr->f1 & SELECT)) ? VFLAG_VERTEX_SELECTED : 0;
						VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bp_prev->vec);
						VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
						vbo_len_used += 1;
						VertexBuffer_set_attrib(vbo, pos_id, vbo_len_used, bp_curr->vec);
						VertexBuffer_set_attrib(vbo, data_id, vbo_len_used, &vflag);
						vbo_len_used += 1;

					}
				}
			}
		}
		if (vbo_len_capacity != vbo_len_used) {
			VertexBuffer_resize_data(vbo, vbo_len_used);
		}

		cache->overlay.edges = Batch_create(PRIM_LINES, vbo, NULL);
	}

	curve_render_data_free(rdata);
}

Batch *BKE_curve_batch_cache_get_wire_edge(Curve *cu, CurveCache *ob_curve_cache)
{
	CurveBatchCache *cache = curve_batch_cache_get(cu);

	if (cache->wire.batch == NULL) {
		/* create batch from Curve */
		CurveRenderData *rdata = curve_render_data_create(cu, ob_curve_cache, CU_DATATYPE_WIRE);

		cache->wire.batch = Batch_create(
		        PRIM_LINES,
		        curve_batch_cache_get_wire_verts(rdata, cache),
		        curve_batch_cache_get_wire_edges(rdata, cache));

		curve_render_data_free(rdata);
	}
	return cache->wire.batch;
}

Batch *BKE_curve_batch_cache_get_normal_edge(Curve *cu, CurveCache *ob_curve_cache, float normal_size)
{
	CurveBatchCache *cache = curve_batch_cache_get(cu);

	if (cache->normal.batch != NULL) {
		cache->normal_size = normal_size;
		if (cache->normal_size != normal_size) {
			BATCH_DISCARD_ALL_SAFE(cache->normal.batch);
		}
	}
	cache->normal_size = normal_size;

	if (cache->normal.batch == NULL) {
		/* create batch from Curve */
		CurveRenderData *rdata = curve_render_data_create(cu, ob_curve_cache, CU_DATATYPE_NORMAL);

		cache->normal.batch = Batch_create(
		        PRIM_LINES,
		        curve_batch_cache_get_normal_verts(rdata, cache),
		        curve_batch_cache_get_normal_edges(rdata, cache));

		curve_render_data_free(rdata);
		cache->normal_size = normal_size;
	}
	return cache->normal.batch;
}

Batch *BKE_curve_batch_cache_get_overlay_edges(Curve *cu)
{
	CurveBatchCache *cache = curve_batch_cache_get(cu);

	if (cache->overlay.edges == NULL) {
		curve_batch_cache_create_overlay_batches(cu);
	}

	return cache->overlay.edges;
}

Batch *BKE_curve_batch_cache_get_overlay_verts(Curve *cu)
{
	CurveBatchCache *cache = curve_batch_cache_get(cu);

	if (cache->overlay.verts == NULL) {
		curve_batch_cache_create_overlay_batches(cu);
	}

	return cache->overlay.verts;
}