# ##### BEGIN GPL LICENSE BLOCK #####
#
#  This program is free software; you can redistribute it and/or
#  modify it under the terms of the GNU General Public License
#  as published by the Free Software Foundation; either version 2
#  of the License, or (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program; if not, write to the Free Software Foundation,
#  Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ##### END GPL LICENSE BLOCK #####

# <pep8 compliant>

"""
./blender.bin --background -noaudio --factory-startup --python tests/python/bl_mesh_uv_select_similar_test.py -- --verbose
"""

import pathlib
import sys
import unittest
import numpy as np

import bpy
import bmesh
import mathutils as mu


NUM_FACES = 8           # create a square grid NUM_FACES wide
WIDTH = NUM_FACES       # width of grid
INRADIUS = WIDTH / 2    # inradius of grid

args = None


def create_test_mesh():
   # create active mesh
    bpy.ops.object.add(type="MESH")
    bpy.ops.mesh.uv_texture_add()
    return bpy.context.active_object.data


def create_test_bmesh():
    bm = bmesh.new()

    # create multiple, split faces
    bmesh.ops.create_grid(bm, x_segments=NUM_FACES + 1, y_segments=NUM_FACES + 1, size=INRADIUS, calc_uvs=True)
    bmesh.ops.translate(bm, vec=(INRADIUS, INRADIUS, 0), verts=bm.verts)
    bmesh.ops.split_edges(bm, edges=bm.edges)
    uvmap = bm.loops.layers.uv.new()

    # calculate simple one-place decimals
    end_point = max(1 - NUM_FACES / 10, 0.1)    # don't scale below 10%
    scale_row = np.linspace(1, end_point, NUM_FACES, endpoint=False)
    scales = np.repeat(scale_row, NUM_FACES)    # same scale for uv faces in each row
    
    # scale uv faces to match mesh faces for easy visual inspection
    i = 0
    for face in bm.faces:
        scale = scales[i]
        scale_vector = (1, scale, 1)
        face_translate = face.calc_center_median()
        face_matrix = mu.Matrix.Translation(face_translate)

        bmesh.ops.scale(bm, vec=scale_vector, space=face_matrix.inverted(), verts=face.verts)
        for face_loop in face.loops:
            face_loop[uvmap].uv = face_loop.vert.co.xy
        i += 1
    return bm


class TestHelper:
    @classmethod
    def setUpClass(cls):
        cls.mesh = create_test_mesh()
        cls.bmesh = create_test_bmesh()
        cls.bmesh.to_mesh(cls.mesh)
        bpy.ops.object.editmode_toggle()

        cls.bmesh = bmesh.from_edit_mesh(cls.mesh)
        cls.bmesh.faces.ensure_lookup_table()
        cls.uv_layer = cls.bmesh.loops.layers.uv.active

    @classmethod
    def tearDownClass(cls):
        if cls.bmesh.is_valid:
            cls.bmesh.free()

    def select_all_mesh_faces(self):
        bpy.ops.mesh.select_all(action='SELECT')

    def deselect_all_uv_faces(self):
        for face in self.bmesh.faces:
            for loop in face.loops:
                loop[self.uv_layer].select = False

    def select_uv_faces(self, indices):
        # rough check for non-sequence
        if not hasattr(indices, '__len__'):
            indices = [indices]
            
        for index in indices:
            for loop in self.bmesh.faces[index].loops:
                loop[self.uv_layer].select = True
        bmesh.update_edit_mesh(self.mesh)

    def get_selected_uv_faces(self):
        selected_uv_faces = []
        for face in self.bmesh.faces:
            selected = True
            for loop in face.loops:
                selected &= loop[self.uv_layer].select
            if selected:
                selected_uv_faces.append(face.index)
        return selected_uv_faces


class SelectSimilarUVFaceTest(TestHelper, unittest.TestCase):
    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'
        self.optype = 'AREA'

    def reset_all(self):
        self.select_all_mesh_faces()
        self.deselect_all_uv_faces()

    def choose_index_given_all_uvs_visible(self, index, compare, threshold):
        self.reset_all()
        self.select_uv_faces(index)
        return bpy.ops.uv.select_similar(type=self.optype, compare=compare, threshold=threshold)

    def check_by_index(self, index, should_be, compare='EQUAL', threshold=0.099):
        res = self.choose_index_given_all_uvs_visible(index, compare=compare, threshold=threshold)
        self.assertEqual({'FINISHED'}, res)
        selected_uv_faces = set(self.get_selected_uv_faces())
        self.assertSetEqual(selected_uv_faces, should_be)

    def test_single_equal(self):
        self.check_by_index(0, set(range(NUM_FACES)))
        self.check_by_index(0, set(range(NUM_FACES * 2)), threshold=0.101)

    def test_single_less(self):
        self.check_by_index(0, set(range(NUM_FACES * NUM_FACES)), compare='LESS')
        index = NUM_FACES * (NUM_FACES - 1)
        self.check_by_index(index, set(range(index, NUM_FACES * NUM_FACES)), compare='LESS')
        self.check_by_index(0, set(range(NUM_FACES * NUM_FACES)), compare='LESS', threshold=0.101)

    def test_single_greater(self):
        self.check_by_index(8, set(range(NUM_FACES * 2)), compare='GREATER')
        index = NUM_FACES * NUM_FACES - 1
        self.check_by_index(index, set(range(NUM_FACES * NUM_FACES)), compare='GREATER')
        self.check_by_index(8, set(range(NUM_FACES * 3)), compare='GREATER', threshold=0.101)

    def test_multiple(self):
        to_select = range(0, NUM_FACES * NUM_FACES, NUM_FACES)
        should_be = set(range(NUM_FACES * NUM_FACES))
        self.check_by_index(to_select, should_be)
        self.check_by_index(to_select, should_be, compare='LESS')
        self.check_by_index(to_select, should_be, compare='GREATER')
        to_select = range(0, NUM_FACES * NUM_FACES)
        should_be = set(to_select)
        self.check_by_index(to_select, should_be)


def main():
    global args
    import argparse

    if '--' in sys.argv:
        argv = [sys.argv[0]] + sys.argv[sys.argv.index('--') + 1:]
    else:
        argv = sys.argv

    parser = argparse.ArgumentParser()
    args, remaining = parser.parse_known_args(argv)

    try:
        unittest.main(argv=remaining)
    except SystemExit as e:
        # keep blender open for debugging tests
        if '--background' in sys.argv:
            raise(SystemExit(e.code))


if __name__ == "__main__":
    import traceback
    # So a python error exits Blender itself too
    try:
        main()
    except SystemExit:
        raise
    except:
        traceback.print_exc()
        sys.exit(1)
