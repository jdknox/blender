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
import math
import numpy as np

import bpy
import bmesh
import mathutils as mu


COLUMNS = 8                  # create a square grid with COLUMNS x COLUMNS faces
ROWS = COLUMNS               # for eaiser visualization of code
TOTAL_FACES = COLUMNS * COLUMNS # width of grid
INRADIUS = COLUMNS / 2       # inradius of grid

args = None


def create_test_mesh(name):
   # create active mesh
    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.ops.object.add(type="MESH")
    bpy.ops.mesh.uv_texture_add()
    obj = bpy.context.active_object
    if name:
        obj.name = 'test_%s_select_similar' % name
    return obj.data


def create_area_bmesh():
    bm = bmesh.new()

    # create multiple, split faces
    bmesh.ops.create_grid(bm, x_segments=COLUMNS + 1, y_segments=COLUMNS + 1, size=INRADIUS, calc_uvs=True)
    bmesh.ops.translate(bm, vec=(INRADIUS, INRADIUS, 0), verts=bm.verts)
    bmesh.ops.split_edges(bm, edges=bm.edges)
    uvmap = bm.loops.layers.uv.new()

    # calculate simple one-place decimals
    end_point = max(1 - COLUMNS / 10, 0.1)    # don't scale below 10%
    scale_row = np.linspace(1, end_point, COLUMNS, endpoint=False)
    scales = np.repeat(scale_row, COLUMNS)    # same scale for uv faces in each row

    i = 0
    for face in bm.faces:
        scale = scales[i]
        scale_vector = (1, scale, 1)
        face_translate = face.calc_center_median()
        face_matrix = mu.Matrix.Translation(face_translate)
        bmesh.ops.scale(bm, vec=scale_vector, space=face_matrix.inverted(), verts=face.verts)
        i += 1
    return bm


def create_sides_bmesh():
    bm = bmesh.new()

    # create multiple "circles"
    s = 1.1
    for y in range(COLUMNS):
        for x in range(COLUMNS):
            n = y + 3   # start with triangle...
            d = bmesh.ops.create_circle(bm, cap_ends=True, segments=n, diameter=0.5)
            bmesh.ops.translate(bm, vec=(s * x, s * y, 0), verts=d['verts'])
    
    uvmap = bm.loops.layers.uv.new()
    return bm


def create_coplanar_bmesh():
    bm = bmesh.new()

    # create 2x2x2 cube of cubes
    s = 2
    w = 2
    rotationMatrix = mu.Matrix.Rotation(math.radians(-45.0), 3, 'Y')
    for z in range(w):
        for y in range(w):
            for x in range(w):
                d = bmesh.ops.create_cube(bm, size=1.0)
                if x == 1 and z == 1:
                    bmesh.ops.rotate(bm, cent=(0, 0, 0), matrix=rotationMatrix, verts=d['verts'])
                bmesh.ops.translate(bm, vec=(s * x, s * y, s * z), verts=d['verts'])
    
    uvmap = bm.loops.layers.uv.new()
    return bm


class TestHelper:
    @classmethod
    def setUpClass(cls, bmesh_function, name=None):
        cls.mesh = create_test_mesh(name)
        cls.bmesh = bmesh_function()
        cls.bmesh.to_mesh(cls.mesh)
        bpy.ops.object.editmode_toggle()

        cls.bmesh = bmesh.from_edit_mesh(cls.mesh)
        cls.bmesh.faces.ensure_lookup_table()
        cls.uv_layer = cls.bmesh.loops.layers.uv.active
        cls.update_uvs()

    @classmethod
    def tearDownClass(cls):
        if cls.bmesh.is_valid:
            cls.bmesh.free()

    @classmethod
    def update_uvs(cls):
        # scale uv faces to match mesh faces for easy visual inspection
        for face in cls.bmesh.faces:
            for face_loop in face.loops:
                face_loop[cls.uv_layer].uv = face_loop.vert.co.xy

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


class SelectSimilarUVFaceNormalTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'NORMAL'
        TestHelper.setUpClass(create_coplanar_bmesh, cls.optype)

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_normal_single(self):
        faces = np.array([0, 6, 12, 18, 24, 36])
        should_be = set(faces)              # -X facing
        self.check_by_index(0, should_be)
        should_be = set(faces + 5)          # +Z facing
        self.check_by_index(5, should_be)
        should_be = set(range(3, 8 * 6, 6)) # -Y facing
        self.check_by_index(3, should_be)
        should_be = set([32, 44])           # top right cubes, normal: <0.7, 0, 0.7>
        self.check_by_index(32, should_be)

    def test_normal_multiple(self):
        all_faces = range(0, 8 * 6)
        should_be = set(all_faces)
        self.check_by_index(all_faces, should_be)

        exclude = np.array([30, 32, 34, 35])        # select by all faces of first cube, but the tool
        should_be = should_be.difference(exclude)   # should skip the ones without (coordinate) unit vecgor normals
        should_be = should_be.difference(exclude + 12)
        self.check_by_index(range(0, 6), should_be)
        


class SelectSimilarUVFaceCoplanarTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'COPLANAR'
        TestHelper.setUpClass(create_coplanar_bmesh, cls.optype)

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_coplanar_single(self):
        should_be = set(range(0, 37, 12))   # plane: YZ, X = -0.5 (left of left cubes)
        self.check_by_index(0, should_be)
        should_be = set(range(5, 24, 6))    # plane: XY, Z = 0.5 (top of bottom cubes)
        self.check_by_index(5, should_be)
        should_be = set([32, 44])           # plane XY, but rotated -45 degrees Y (top right cubes)
        self.check_by_index(32, should_be)
        should_be = set([3, 9, 27, 33])     # plane: XZ, Y = -0.5 (front faces)
        self.check_by_index(3, should_be)

    def test_coplanar_multiple(self):
        all_faces = range(0, 8 * 6)
        should_be = set(all_faces)
        self.check_by_index(all_faces, should_be)


class SelectSimilarUVFaceSmoothTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'SMOOTH'
        TestHelper.setUpClass(create_area_bmesh, cls.optype)
        for face in cls.bmesh.faces:
            face.smooth = True if (face.index % COLUMNS) else False

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_smooth_single(self):
        should_be = set(range(0, TOTAL_FACES, COLUMNS))
        self.check_by_index(0, should_be)
        should_be = set(range(TOTAL_FACES)).difference(should_be)
        self.check_by_index(1, should_be)

    def test_smooth_multiple(self):
        should_be = set(range(TOTAL_FACES))
        self.check_by_index([0,1], should_be)
        self.check_by_index(range(TOTAL_FACES), should_be)


class SelectSimilarUVFaceMaterialTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'MATERIAL'
        TestHelper.setUpClass(create_area_bmesh, cls.optype)
        for i in range(2):
            material = bpy.data.materials.new('material_%02i' % i)
            gb = 0.2 + 0.6 * i
            material.diffuse_color = [0.8, gb, gb]
            cls.mesh.materials.append(material)
        for face in cls.bmesh.faces:
            face.material_index = 1 if (face.index % COLUMNS) else 0

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_material_single(self):
        should_be = set(range(0, TOTAL_FACES, COLUMNS))
        self.check_by_index(0, should_be)
        should_be = set(range(TOTAL_FACES)).difference(should_be)
        self.check_by_index(1, should_be)

    def test_material_multiple(self):
        should_be = set(range(TOTAL_FACES))
        self.check_by_index([0,1], should_be)
        self.check_by_index(range(TOTAL_FACES), should_be)


class SelectSimilarUVFaceImageTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'IMAGE'
        TestHelper.setUpClass(create_area_bmesh, cls.optype)

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_image_single(self):
        should_be = set(range(TOTAL_FACES))
        self.check_by_index(0, should_be)


class SelectSimilarUVFaceSidesTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'SIDES'
        TestHelper.setUpClass(create_sides_bmesh, cls.optype)

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_sides_single(self):
        for i in range(ROWS):
            index = i * COLUMNS
            should_be = set(np.arange(COLUMNS) + index)
            self.check_by_index(index, should_be)

    def test_sides_single_less(self):
        for i in range(ROWS):
            index = i * COLUMNS
            should_be = set(np.arange(index + 1))
            self.check_by_index(index, should_be, compare='LESS')

    def test_sides_single_greater(self):
        for i in range(ROWS):
            index = i * COLUMNS
            should_be = set(np.arange(index + COLUMNS, TOTAL_FACES))
            should_be.add(index)
            self.check_by_index(index, should_be, compare='GREATER')

    def test_sides_multiple(self):
        indices = range(0, TOTAL_FACES, COLUMNS)
        should_be = set(range(TOTAL_FACES))
        self.check_by_index(indices, should_be)
        indices = range(TOTAL_FACES)
        self.check_by_index(indices, should_be)


class SelectSimilarUVFacePerimeterTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'PERIMETER'
        TestHelper.setUpClass(create_area_bmesh, cls.optype)
        # scale the last face in each row so that the next row up has the same perimeter
        faces = list(cls.bmesh.faces)[COLUMNS - 1::COLUMNS]
        for face in faces:
            face_translate = face.calc_center_median()
            face_matrix = mu.Matrix.Translation(face_translate)
            verts = list(face.verts)[1:3]
            bmesh.ops.translate(cls.bmesh, vec=[-0.1, 0, 0], verts=verts)
        # last face in top row can match the first row
        bmesh.ops.translate(cls.bmesh, vec=[0.8, 0, 0], verts=verts)
        cls.update_uvs()

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_perimeter_single(self):
        for i in range(COLUMNS - 1, TOTAL_FACES, COLUMNS):
            should_be = set((np.arange(0, COLUMNS) + i) % TOTAL_FACES)
            self.check_by_index(i, should_be)

    def test_perimeter_single_less(self):
        # start checking from the top right face, then "wrap around" to bottom right
        for i in range(-1, TOTAL_FACES - 1, COLUMNS):
            should_be = set(np.arange(i, TOTAL_FACES - 1) % TOTAL_FACES)
            self.check_by_index(i, should_be, compare='LESS')

    def test_perimeter_single_greater(self):
        should_be = set()
        for i in range(-1, TOTAL_FACES - 1, COLUMNS):
            should_be = should_be.union(set((np.arange(0, COLUMNS) + i) % TOTAL_FACES))
            self.check_by_index(i, should_be, compare='GREATER')

    def test_perimeter_multiple(self):
        indices = range(COLUMNS - 1, TOTAL_FACES, COLUMNS)
        should_be = set(range(TOTAL_FACES))
        self.check_by_index(indices, should_be)
        indices = range(TOTAL_FACES)
        self.check_by_index(indices, should_be)


class SelectSimilarUVFaceAreaTest(TestHelper, unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.optype = 'AREA'
        TestHelper.setUpClass(create_area_bmesh, cls.optype)

    def setUp(self):
        bpy.context.scene.tool_settings.uv_select_mode = 'FACE'

    def test_area_single(self):
        self.check_by_index(0, set(range(COLUMNS)))
        self.check_by_index(0, set(range(COLUMNS * 2)), threshold=0.101)

    def test_area_single_less(self):
        self.check_by_index(0, set(range(TOTAL_FACES)), compare='LESS')
        index = COLUMNS * (COLUMNS - 1)
        self.check_by_index(index, set(range(index, TOTAL_FACES)), compare='LESS')
        self.check_by_index(0, set(range(TOTAL_FACES)), compare='LESS', threshold=0.101)

    def test_area_single_greater(self):
        self.check_by_index(8, set(range(COLUMNS * 2)), compare='GREATER')
        index = TOTAL_FACES - 1
        self.check_by_index(index, set(range(TOTAL_FACES)), compare='GREATER')
        self.check_by_index(8, set(range(COLUMNS * 3)), compare='GREATER', threshold=0.101)

    def test_area_multiple(self):
        indices = range(0, TOTAL_FACES, COLUMNS)
        should_be = set(range(TOTAL_FACES))
        self.check_by_index(indices, should_be)
        self.check_by_index(indices, should_be, compare='LESS')
        self.check_by_index(indices, should_be, compare='GREATER')
        indices = range(TOTAL_FACES)
        should_be = set(indices)
        self.check_by_index(indices, should_be)


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
