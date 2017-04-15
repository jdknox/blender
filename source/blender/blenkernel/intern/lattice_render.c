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
 * Contributor(s): Blender Foundation, Mike Erwin, Dalai Felinto
 *
 * ***** END GPL LICENSE BLOCK *****
 */

/** \file blender/blenkernel/intern/lattice_render.c
 *  \ingroup bke
 *
 * \brief Lattice API for render engines
 */

#include "MEM_guardedalloc.h"

#include "BLI_utildefines.h"
#include "BLI_math_vector.h"

#include "DNA_curve_types.h"
#include "DNA_lattice_types.h"

#include "BKE_lattice_render.h"

#include "GPU_batch.h"

#define SELECT   1

/**
 * TODO
 * - 'DispList' is currently not used
 *   (we could avoid using since it will be removed)
 */

/* ---------------------------------------------------------------------- */
/* Lattice Interface, direct access to basic data. */

static int vert_tot(int u, int v, int w)
{
	if (u <= 0 || v <= 0 || w <= 0) {
		return 0;
	}
	return u * v * w;
}

static int edge_tot(int u, int v, int w)
{
	if (u <= 0 || v <= 0 || w <= 0) {
		return 0;
	}
	return (((((u - 1) * v) +
	          ((v - 1) * u)) * w) +
	        ((w - 1) * (u * v)));
}

static int lattice_render_verts_num_get(Lattice *lt)
{
	if (lt->editlatt) {
		lt = lt->editlatt->latt;
	}

	const int u = lt->pntsu;
	const int v = lt->pntsv;
	const int w = lt->pntsw;

	if ((lt->flag & LT_OUTSIDE) == 0) {
		return vert_tot(u, v, w);
	}
	else {
		/* TODO remove internal coords */
		return vert_tot(u, v, w);
	}
}

static int lattice_render_edges_num_get(Lattice *lt)
{
	if (lt->editlatt) {
		lt = lt->editlatt->latt;
	}

	const int u = lt->pntsu;
	const int v = lt->pntsv;
	const int w = lt->pntsw;

	if ((lt->flag & LT_OUTSIDE) == 0) {
		return edge_tot(u, v, w);
	}
	else {
		/* TODO remove internal coords */
		return edge_tot(u, v, w);
	}
}

/* ---------------------------------------------------------------------- */
/* Lattice Interface, indirect, partially cached access to complex data. */

typedef struct LatticeRenderData {
	int types;

	int totvert;
	int totedge;

	struct {
		int u_len, v_len, w_len;
	} dims;
	bool show_only_outside;

	struct EditLatt *edit_latt;
	BPoint *bp;

	int actbp;
} LatticeRenderData;

enum {
	LR_DATATYPE_VERT       = 1 << 0,
	LR_DATATYPE_EDGE       = 1 << 1,
	LR_DATATYPE_OVERLAY    = 1 << 2,
};

static LatticeRenderData *lattice_render_data_create(Lattice *lt, const int types)
{
	LatticeRenderData *lrdata = MEM_callocN(sizeof(*lrdata), __func__);
	lrdata->types = types;

	if (lt->editlatt) {
		EditLatt *editlatt = lt->editlatt;
		lt = editlatt->latt;

		lrdata->edit_latt = editlatt;

		if (types & (LR_DATATYPE_VERT)) {
			lrdata->totvert = lattice_render_verts_num_get(lt);
		}
		if (types & (LR_DATATYPE_EDGE)) {
			lrdata->totedge = lattice_render_edges_num_get(lt);
		}
		if (types & LR_DATATYPE_OVERLAY) {
			lrdata->actbp = lt->actbp;
		}
	}
	else {
		if (types & (LR_DATATYPE_VERT)) {
			lrdata->totvert = lattice_render_verts_num_get(lt);
		}
		if (types & (LR_DATATYPE_EDGE)) {
			lrdata->totedge = lattice_render_edges_num_get(lt);
			/*no edge data */
		}
	}

	lrdata->bp = lt->def;

	lrdata->dims.u_len = lt->pntsu;
	lrdata->dims.v_len = lt->pntsv;
	lrdata->dims.w_len = lt->pntsw;

	lrdata->show_only_outside = (lt->flag & LT_OUTSIDE) != 0;
	lrdata->actbp = lt->actbp;

	return lrdata;
}

static void lattice_render_data_free(LatticeRenderData *lrdata)
{
#if 0
	if (lrdata->loose_verts) {
		MEM_freeN(lrdata->loose_verts);
	}
#endif
	MEM_freeN(lrdata);
}

static int lattice_render_data_verts_num_get(const LatticeRenderData *lrdata)
{
	BLI_assert(lrdata->types & LR_DATATYPE_VERT);
	return lrdata->totvert;
}

static int lattice_render_data_edges_num_get(const LatticeRenderData *lrdata)
{
	BLI_assert(lrdata->types & LR_DATATYPE_EDGE);
	return lrdata->totedge;
}

static const BPoint *lattice_render_data_vert_bpoint(const LatticeRenderData *lrdata, const int vert_idx)
{
	BLI_assert(lrdata->types & LR_DATATYPE_VERT);
	return &lrdata->bp[vert_idx];
}

enum {
	VFLAG_VERTEX_SELECTED = 1 << 0,
	VFLAG_VERTEX_ACTIVE   = 1 << 1,
};

/* ---------------------------------------------------------------------- */
/* Lattice Batch Cache */

typedef struct LatticeBatchCache {
	VertexBuffer *pos;
	ElementList *edges;

	Batch *all_verts;
	Batch *all_edges;

	Batch *overlay_verts;

	/* settings to determine if cache is invalid */
	bool is_dirty;

	struct {
		int u_len, v_len, w_len;
	} dims;
	bool show_only_outside;

	bool is_editmode;
} LatticeBatchCache;

/* Batch cache management. */

static bool lattice_batch_cache_valid(Lattice *lt)
{
	LatticeBatchCache *cache = lt->batch_cache;

	if (cache == NULL) {
		return false;
	}

	if (cache->is_editmode != (lt->editlatt != NULL)) {
		return false;
	}

	if (cache->is_dirty == false) {
		return true;
	}
	else {
		if (cache->is_editmode) {
			return false;
		}
		else if ((cache->dims.u_len != lt->pntsu) ||
		         (cache->dims.v_len != lt->pntsv) ||
		         (cache->dims.w_len != lt->pntsw) ||
		         ((cache->show_only_outside != ((lt->flag & LT_OUTSIDE) != 0))))
		{
			return false;
		}
	}

	return true;
}

static void lattice_batch_cache_init(Lattice *lt)
{
	LatticeBatchCache *cache = lt->batch_cache;

	if (!cache) {
		cache = lt->batch_cache = MEM_callocN(sizeof(*cache), __func__);
	}
	else {
		memset(cache, 0, sizeof(*cache));
	}

	cache->dims.u_len = lt->pntsu;
	cache->dims.v_len = lt->pntsv;
	cache->dims.w_len = lt->pntsw;
	cache->show_only_outside = (lt->flag & LT_OUTSIDE) != 0;

	cache->is_editmode = lt->editlatt != NULL;

	cache->is_dirty = false;
}

static LatticeBatchCache *lattice_batch_cache_get(Lattice *lt)
{
	if (!lattice_batch_cache_valid(lt)) {
		BKE_lattice_batch_cache_clear(lt);
		lattice_batch_cache_init(lt);
	}
	return lt->batch_cache;
}

void BKE_lattice_batch_cache_dirty(Lattice *lt)
{
	LatticeBatchCache *cache = lt->batch_cache;
	if (cache) {
		cache->is_dirty = true;
	}
}

void BKE_lattice_batch_selection_dirty(Lattice *lt)
{
	LatticeBatchCache *cache = lt->batch_cache;
	if (cache) {
		/* TODO Separate Flag vbo */
		BATCH_DISCARD_ALL_SAFE(cache->overlay_verts);
	}
}

void BKE_lattice_batch_cache_clear(Lattice *lt)
{
	LatticeBatchCache *cache = lt->batch_cache;
	if (!cache) {
		return;
	}

	BATCH_DISCARD_SAFE(cache->all_verts);
	BATCH_DISCARD_SAFE(cache->all_edges);
	BATCH_DISCARD_ALL_SAFE(cache->overlay_verts);

	VERTEXBUFFER_DISCARD_SAFE(cache->pos);
	ELEMENTLIST_DISCARD_SAFE(cache->edges);
}

void BKE_lattice_batch_cache_free(Lattice *lt)
{
	BKE_lattice_batch_cache_clear(lt);
	MEM_SAFE_FREE(lt->batch_cache);
}

/* Batch cache usage. */
static VertexBuffer *lattice_batch_cache_get_pos(LatticeRenderData *lrdata, LatticeBatchCache *cache)
{
	BLI_assert(lrdata->types & LR_DATATYPE_VERT);

	if (cache->pos == NULL) {
		static VertexFormat format = { 0 };
		static unsigned pos_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
		}

		const int vertex_ct = lattice_render_data_verts_num_get(lrdata);

		cache->pos = VertexBuffer_create_with_format(&format);
		VertexBuffer_allocate_data(cache->pos, vertex_ct);
		for (int i = 0; i < vertex_ct; ++i) {
			const BPoint *bp = lattice_render_data_vert_bpoint(lrdata, i);
			VertexBuffer_set_attrib(cache->pos, pos_id, i, bp->vec);
		}
	}

	return cache->pos;
}

static ElementList *lattice_batch_cache_get_edges(LatticeRenderData *lrdata, LatticeBatchCache *cache)
{
	BLI_assert(lrdata->types & (LR_DATATYPE_VERT | LR_DATATYPE_EDGE));

	if (cache->edges == NULL) {
		const int vertex_ct = lattice_render_data_verts_num_get(lrdata);
		const int edge_len = lattice_render_data_edges_num_get(lrdata);
		int edge_len_real = 0;

		ElementListBuilder elb;
		ElementListBuilder_init(&elb, PRIM_LINES, edge_len, vertex_ct);

#define LATT_INDEX(u, v, w) \
	((((w) * lrdata->dims.v_len + (v)) * lrdata->dims.u_len) + (u))

		for (int w = 0; w < lrdata->dims.w_len; w++) {
			int wxt = (w == 0 || w == lrdata->dims.w_len - 1);
			for (int v = 0; v < lrdata->dims.v_len; v++) {
				int vxt = (v == 0 || v == lrdata->dims.v_len - 1);
				for (int u = 0; u < lrdata->dims.u_len; u++) {
					int uxt = (u == 0 || u == lrdata->dims.u_len - 1);

					if (w && ((uxt || vxt) || !lrdata->show_only_outside)) {
						add_line_vertices(&elb, LATT_INDEX(u, v, w - 1), LATT_INDEX(u, v, w));
						BLI_assert(edge_len_real <= edge_len);
						edge_len_real++;
					}
					if (v && ((uxt || wxt) || !lrdata->show_only_outside)) {
						add_line_vertices(&elb, LATT_INDEX(u, v - 1, w), LATT_INDEX(u, v, w));
						BLI_assert(edge_len_real <= edge_len);
						edge_len_real++;
					}
					if (u && ((vxt || wxt) || !lrdata->show_only_outside)) {
						add_line_vertices(&elb, LATT_INDEX(u - 1, v, w), LATT_INDEX(u, v, w));
						BLI_assert(edge_len_real <= edge_len);
						edge_len_real++;
					}
				}
			}
		}

#undef LATT_INDEX

		if (lrdata->show_only_outside) {
			BLI_assert(edge_len_real <= edge_len);
		}
		else {
			BLI_assert(edge_len_real == edge_len);
		}

		cache->edges = ElementList_build(&elb);
	}

	return cache->edges;
}

static void lattice_batch_cache_create_overlay_batches(Lattice *lt)
{
	/* Since LR_DATATYPE_OVERLAY is slow to generate, generate them all at once */
	int options = LR_DATATYPE_VERT | LR_DATATYPE_OVERLAY;

	LatticeBatchCache *cache = lattice_batch_cache_get(lt);
	LatticeRenderData *lrdata = lattice_render_data_create(lt, options);

	if (cache->overlay_verts == NULL) {
		static VertexFormat format = { 0 };
		static unsigned pos_id, data_id;
		if (format.attrib_ct == 0) {
			/* initialize vertex format */
			pos_id = VertexFormat_add_attrib(&format, "pos", COMP_F32, 3, KEEP_FLOAT);
			data_id = VertexFormat_add_attrib(&format, "data", COMP_U8, 1, KEEP_INT);
		}

		const int vertex_ct = lattice_render_data_verts_num_get(lrdata);

		VertexBuffer *vbo = VertexBuffer_create_with_format(&format);
		VertexBuffer_allocate_data(vbo, vertex_ct);
		for (int i = 0; i < vertex_ct; ++i) {
			const BPoint *bp = lattice_render_data_vert_bpoint(lrdata, i);

			char vflag = 0;
			if (bp->f1 & SELECT) {
				if (i == lrdata->actbp) {
					vflag |= VFLAG_VERTEX_ACTIVE;
				}
				else {
					vflag |= VFLAG_VERTEX_SELECTED;
				}
			}

			VertexBuffer_set_attrib(vbo, pos_id, i, bp->vec);
			VertexBuffer_set_attrib(vbo, data_id, i, &vflag);
		}

		cache->overlay_verts = Batch_create(PRIM_POINTS, vbo, NULL);
	}	

	lattice_render_data_free(lrdata);
}

Batch *BKE_lattice_batch_cache_get_all_edges(Lattice *lt)
{
	LatticeBatchCache *cache = lattice_batch_cache_get(lt);

	if (cache->all_edges == NULL) {
		/* create batch from Lattice */
		LatticeRenderData *lrdata = lattice_render_data_create(lt, LR_DATATYPE_VERT | LR_DATATYPE_EDGE);

		cache->all_edges = Batch_create(PRIM_LINES, lattice_batch_cache_get_pos(lrdata, cache),
		                                lattice_batch_cache_get_edges(lrdata, cache));

		lattice_render_data_free(lrdata);
	}

	return cache->all_edges;
}

Batch *BKE_lattice_batch_cache_get_all_verts(Lattice *lt)
{
	LatticeBatchCache *cache = lattice_batch_cache_get(lt);

	if (cache->all_verts == NULL) {
		LatticeRenderData *lrdata = lattice_render_data_create(lt, LR_DATATYPE_VERT);

		cache->all_verts = Batch_create(PRIM_POINTS, lattice_batch_cache_get_pos(lrdata, cache), NULL);

		lattice_render_data_free(lrdata);
	}

	return cache->all_verts;
}

Batch *BKE_lattice_batch_cache_get_overlay_verts(Lattice *lt)
{
	LatticeBatchCache *cache = lattice_batch_cache_get(lt);

	if (cache->overlay_verts == NULL) {
		lattice_batch_cache_create_overlay_batches(lt);
	}

	return cache->overlay_verts;
}