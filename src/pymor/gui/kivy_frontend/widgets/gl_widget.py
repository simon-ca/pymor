from pymor.grids.referenceelements import triangle, square
import time
import math

VS = """
#ifdef GL_ES
    precision highp float;
#endif

attribute vec2  v_pos;
attribute float  v_color;

uniform mat4 modelview_mat;
uniform mat4 projection_mat;
uniform vec2 shift;
uniform vec2 scale;

varying vec4 frag_color;
varying vec2 tex_coord0;

void main (void) {
    vec2 pos_moved = v_pos * scale + shift;
    vec4 pos = modelview_mat * vec4(pos_moved,1.0,1.0);
    gl_Position = projection_mat * pos;
    frag_color = vec4(v_color, 0.0, 0.0, 0.0);
    tex_coord0 = vec2(0.0, 0.0);
}
"""

FS = """
#ifdef GL_ES
    precision highp float;
#endif

varying vec4 frag_color;

uniform sampler2D tex;

vec3 getJetColor(float value) {
         float fourValue = 4.0 * value;
         float red   = min(fourValue - 1.5, -fourValue + 4.5);
         float green = min(fourValue - 0.5, -fourValue + 3.5);
         float blue  = min(fourValue + 0.5, -fourValue + 2.5);

         return clamp( vec3(red, green, blue), 0.0, 1.0 );
    }

vec3 getAntiJetColor(float value) {
         //R = -0.5*sin( L*(1.37*pi)+0.13*pi )+0.5;
         //G = -0.4*cos( L*(1.5*pi) )+0.4;
         //B =  0.3*sin( L*(2.11*pi) )+0.3;

         float pi = 3.1415926;

         float red   = -0.5*sin(value*1.37*pi+0.13*pi)+0.5;
         float green = -0.4*cos(value*1.50*pi)+0.4;
         float blue  = 0.3*sin(value*2.11*pi)+0.3;

         return clamp( vec3(red, green, blue), 0.0, 1.0 );
    }

void main (void){
    float value = frag_color.x;
    gl_FragColor = vec4(getAntiJetColor(value), 1.0);
}
"""

import numpy as np

try:
    from kivy.graphics import RenderContext
    from kivy.uix.widget import Widget
    from kivy.uix.label import Label
    HAVE_KIVY = True
except ImportError:
    HAVE_KIVY = False

try:
    from kivy.graphics.opengl import *
    HAVE_GL = True
except ImportError:
    HAVE_GL = False

HAVE_ALL = HAVE_KIVY and HAVE_GL

if HAVE_ALL:

    def getGLPatchWidget(parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):

        from kivy.modules import inspector
        from kivy.core.window import Window
        from kivy.graphics import Mesh
        from kivy.uix.widget import Widget
        from kivy.uix.boxlayout import BoxLayout
        from kivy.core.window import Window
        from kivy.uix.widget import Widget
        from kivy.graphics.transformation import Matrix

        from kivy.graphics import Fbo, Rectangle

        from pymor.grids.constructions import flatten_grid

        class GLPatchWidgetFBO(Widget):

            def __init__(self, grid, vmin=None, vmax=None, bounding_box=([0, 0], [1, 1]), codim=2):
                assert grid.reference_element in (triangle, square)
                assert grid.dim == 2
                assert codim in (0, 2)

                super(GLPatchWidgetFBO, self).__init__()

                self.grid = grid

                subentities, coordinates, entity_map = flatten_grid(grid)

                self.subentities = subentities
                self.entity_map = entity_map
                self.reference_element = grid.reference_element
                self.vmin = vmin
                self.vmax = vmax
                self.bounding_box = bounding_box
                self.codim = codim

                bb = self.bounding_box
                size_ = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])

                self.shift = bb[0]
                self.scale = 1. / size_

                # setup buffers
                if self.reference_element == triangle:
                    if codim == 2:
                        self.vertex_data = np.empty(len(coordinates),
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.asarray(subentities)
                    else:
                        self.vertex_data = np.empty(len(subentities) * 3,
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.arange(len(subentities) * 3, dtype=np.uint32)
                else:
                    if codim == 2:
                        self.vertex_data = np.empty(len(coordinates),
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
                    else:
                        self.vertex_data = np.empty(len(subentities) * 6,
                                                    dtype=[('position', 'f4', 2), ('color', 'f4', 1)])
                        self.indices = np.arange(len(subentities) * 6, dtype=np.uint32)

                self.indices = np.ascontiguousarray(self.indices)

                self.vertex_data['color'] = 1

                self.set_coordinates(coordinates)
                self.meshes = None

                with self.canvas:
                    self.fbo = Fbo(use_parent_modelview=True, size=(100, 100))
                    self.rect = Rectangle(texture=self.fbo.texture)

                self.fbo.shader.vs = VS
                self.fbo.shader.fs = FS

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                inspector.create_inspector(Window, self)

            def update_meshes(self):
                print("update_meshes()", len(self.meshes))
                start = time.time()
                num_meshes = len(self.meshes)
                max_vertices = 2**16//3

                if num_meshes == 1:
                    vert = self.vertex_data
                    vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                    self.meshes[0].vertices = vert
                else:
                    for i in range(num_meshes-1):
                        ind = self.indices[i*max_vertices:(i+1)*max_vertices].flatten()
                        vert = self.vertex_data[ind]
                        vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                        self.meshes[i].vertices = vert

                    i = num_meshes - 1
                    ind = self.indices[i*max_vertices:].flatten()
                    vert = self.vertex_data[ind]
                    vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                    self.meshes[-1].vertices = vert

                stop = time.time()

                print("Mesh update took {} seconds".format(stop-start))

            # todo optimization
            def create_meshes(self):

                print("create_meshes()")
                start = time.time()
                max_vertices = 2**16//3

                num_vertices = len(self.indices)
                num_meshes = int(math.ceil(num_vertices/max_vertices))

                print("num_meshes:", num_meshes)
                print("num_vertices:", num_vertices)
                print("max_vertices:", max_vertices)

                vertex_format = [
                    (b'v_pos', 2, 'float'),
                    (b'v_color', 1, 'float'),
                ]

                if num_meshes == 1:
                    ind = self.indices.flatten()
                    vert = self.vertex_data
                    vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                    self.meshes = [Mesh(vertices=vert, indices=ind, fmt=vertex_format, mode='triangles')]
                else:
                    self.meshes = []
                    for i in range(num_meshes-1):
                        ind = self.indices[i*max_vertices:(i+1)*max_vertices].flatten()
                        vert = self.vertex_data[ind]
                        vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                        self.meshes.append(Mesh(vertices=vert, indices=np.arange(len(ind)),
                                                fmt=vertex_format, mode='triangles'))
                    i = num_meshes - 1
                    ind = self.indices[i*max_vertices:].flatten()
                    vert = self.vertex_data[ind]
                    vert = np.array((vert['position'][:,0], vert['position'][:,1], vert['color'])).T.flatten()
                    self.meshes.append(Mesh(vertices=vert, indices=np.arange(len(ind)),
                                            fmt=vertex_format, mode='triangles'))

                #self.canvas.clear()

                for i, mesh in enumerate(self.meshes):
                    print("Mesh ", i)
                    #self.canvas.add(mesh)
                    self.fbo.add(mesh)

                end = time.time()

                print("Mesh splitting took {} seconds".format(end-start))

            def set_coordinates(self, coordinates):
                if self.codim == 2:
                    self.vertex_data['position'][:, 0:2] = coordinates
                    self.vertex_data['position'][:, 0:2] += self.shift
                    self.vertex_data['position'][:, 0:2] *= self.scale
                elif self.reference_element == triangle:
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data['position'][:, 0:2] = VERTEX_POS.reshape((-1, 2))
                else:
                    num_entities = len(self.subentities)
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data['position'][0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                    self.vertex_data['position'][num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))

            def set(self, U, vmin=None, vmax=None):
                self.vmin = self.vmin if vmin is None else vmin
                self.vmax = self.vmax if vmax is None else vmax

                U_buffer = self.vertex_data['color']
                if self.codim == 2:
                    U_buffer[:] = U[self.entity_map]
                elif self.reference_element == triangle:
                    U_buffer[:] = np.repeat(U, 3)
                else:
                    U_buffer[:] = np.tile(np.repeat(U, 3), 2)

                # normalize
                vmin = np.min(U) if self.vmin is None else self.vmin
                vmax = np.max(U) if self.vmax is None else self.vmax
                U_buffer -= vmin
                if (vmax - vmin) > 0:
                    U_buffer /= float(vmax - vmin)

                if self.meshes is None:
                    self.create_meshes()
                else:
                    self.update_meshes()

            def on_pos(self, instance, value):
                self.rect.pos = value

            def on_size(self, instance, value):
                self.rect.texture = self.fbo.texture
                self.rect.size = value

                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]


        class GLPatchWidgetFBOSpeed(Widget):

            MAX_VERTICES = 2**16//3  # this bound is pessimistic as the mesh splitting algorithm is dumb.
            FBO_SIZE = (100, 100)

            def __init__(self, grid, vmin=None, vmax=None, bounding_box=([0, 0], [1, 1]), codim=2):
                assert grid.reference_element in (triangle, square)
                assert grid.dim == 2
                assert codim in (0, 2)

                super(GLPatchWidgetFBOSpeed, self).__init__()

                self.grid = grid

                subentities, coordinates, entity_map = flatten_grid(grid)

                self.subentities = subentities
                self.entity_map = entity_map
                self.reference_element = grid.reference_element
                self.vmin = vmin
                self.vmax = vmax
                self.bounding_box = bounding_box
                self.codim = codim

                bb = self.bounding_box
                size_ = np.array([bb[1][0] - bb[0][0], bb[1][1] - bb[0][1]])

                self.shift = bb[0]
                self.scale = 1. / size_

                print("SHIFT/SCALE", self.shift, self.scale)

                # setup buffers
                if self.reference_element == triangle:
                    if codim == 2:
                        self.vertex_data = np.empty((len(coordinates), 3))
                        self.indices = np.asarray(subentities)
                    else:
                        self.vertex_data = np.empty((len(subentities)*3, 3))
                        self.indices = np.arange(len(subentities) * 3, dtype=np.uint32)
                else:
                    if codim == 2:
                        self.vertex_data = np.empty((len(coordinates), 3))
                        self.indices = np.vstack((subentities[:, 0:3], subentities[:, [0, 2, 3]]))
                    else:
                        self.vertex_data = np.empty((len(subentities)*6, 3))
                        self.indices = np.arange(len(subentities) * 6, dtype=np.uint32)

                self.set_coordinates(coordinates)
                self.meshes = None

                with self.canvas:
                    self.fbo = Fbo(use_parent_modelview=True, size=self.FBO_SIZE)
                    self.rect = Rectangle(texture=self.fbo.texture)

                self.fbo.shader.vs = VS
                self.fbo.shader.fs = FS

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                # this lets you inspect the user interface with the shortcut Ctrl+E
                inspector.create_inspector(Window, self)

            # todo optimization (prevent copying of vertices)
            # to do so the vertices for each mesh must be contingeous in memory
            def update_meshes(self):
                start = time.time()
                num_meshes = len(self.meshes)
                max_vertices = self.MAX_VERTICES

                if num_meshes == 1:
                    self.meshes[0].vertices = self.vertex_data.reshape((-1))
                else:
                    for i in range(num_meshes-1):
                        ind = self.indices[i*max_vertices:(i+1)*max_vertices].reshape((-1))
                        self.meshes[i].vertices = self.vertex_data[ind].reshape((-1))

                    i = num_meshes - 1
                    ind = self.indices[i*max_vertices:].reshape((-1))
                    self.meshes[-1].vertices = self.vertex_data[ind].reshape((-1))

                stop = time.time()

                print("Mesh update SPEED took {} seconds".format(stop-start))

            def create_meshes(self):

                print("create_meshes()")
                start = time.time()
                max_vertices = self.MAX_VERTICES

                num_vertices = len(self.indices)
                num_meshes = int(math.ceil(num_vertices/max_vertices))

                print("num_meshes:", num_meshes)
                print("num_vertices:", num_vertices)
                print("max_vertices:", max_vertices)

                vertex_format = [
                    (b'v_pos', 2, 'float'),
                    (b'v_color', 1, 'float'),
                ]

                if num_meshes == 1 or num_vertices < max_vertices:
                    # if the number of vertices doesn't exceed max_vertices we can use one mesh
                    ind = self.indices.flatten()
                    self.meshes = [Mesh(vertices=self.vertex_data.flatten(), indices=ind, fmt=vertex_format, mode='triangles')]
                else:
                    self.meshes = []
                    for i in range(num_meshes-1):
                        ind = self.indices[i*max_vertices:(i+1)*max_vertices].flatten()
                        self.meshes.append(Mesh(vertices=self.vertex_data[ind].flatten(), indices=np.arange(len(ind)),
                                                fmt=vertex_format, mode='triangles'))
                    i = num_meshes - 1
                    ind = self.indices[i*max_vertices:].flatten()
                    self.meshes.append(Mesh(vertices=self.vertex_data[ind].flatten(), indices=np.arange(len(ind)),
                                            fmt=vertex_format, mode='triangles'))

                for i, mesh in enumerate(self.meshes):
                    self.fbo.add(mesh)

                end = time.time()

                print("Mesh splitting took {} seconds".format(end-start))

            def set_coordinates(self, coordinates):
                if self.codim == 2:
                    self.vertex_data[:, 0:2][:, 0:2] = coordinates
                    self.vertex_data[:, 0:2][:, 0:2] += self.shift
                    self.vertex_data[:, 0:2][:, 0:2] *= self.scale
                elif self.reference_element == triangle:
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data[:, 0:2][:, 0:2] = VERTEX_POS.reshape((-1, 2))
                else:
                    num_entities = len(self.subentities)
                    VERTEX_POS = coordinates[self.subentities]
                    VERTEX_POS += self.shift
                    VERTEX_POS *= self.scale
                    self.vertex_data[:, 0:2][0:num_entities * 3, 0:2] = VERTEX_POS[:, 0:3, :].reshape((-1, 2))
                    self.vertex_data[:, 0:2][num_entities * 3:, 0:2] = VERTEX_POS[:, [0, 2, 3], :].reshape((-1, 2))

            def set(self, U, vmin=None, vmax=None):
                self.vmin = self.vmin if vmin is None else vmin
                self.vmax = self.vmax if vmax is None else vmax

                U_buffer = self.vertex_data[:, 2]
                if self.codim == 2:
                    U_buffer[:] = U[self.entity_map]
                elif self.reference_element == triangle:
                    U_buffer[:] = np.repeat(U, 3)
                else:
                    U_buffer[:] = np.tile(np.repeat(U, 3), 2)

                # normalize
                vmin = np.min(U) if self.vmin is None else self.vmin
                vmax = np.max(U) if self.vmax is None else self.vmax
                U_buffer -= vmin
                if (vmax - vmin) > 0:
                    U_buffer /= float(vmax - vmin)

                if self.meshes is None:
                    self.create_meshes()
                else:
                    self.update_meshes()

            def on_pos(self, instance, value):
                self.rect.pos = value

            def on_size(self, instance, value):
                self.rect.texture = self.fbo.texture
                self.rect.size = value

                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]

        return GLPatchWidgetFBOSpeed(grid=grid, vmin=vmin, vmax=vmax, bounding_box=bounding_box,
                          codim=codim)

    def getColorBarWidget(padding, U=None, vmin=None, vmax=None):

        from kivy.uix.boxlayout import BoxLayout
        from kivy.uix.label import Label
        from kivy.graphics.vertex_instructions import Mesh
        from kivy.graphics.transformation import Matrix
        from kivy.graphics import Fbo, Rectangle

        class ColorBarFBO(Widget):

            RESOLUTION = 10
            FBO_SIZE = (100, 100)
            BAR_WIDTH = 40

            def __init__(self):

                super(ColorBarFBO, self).__init__()

                with self.canvas:
                    self.fbo = Fbo(use_parent_modelview=True, size=self.FBO_SIZE)
                    self.rect = Rectangle(texture=self.fbo.texture)

                self.fbo.shader.vs = VS
                self.fbo.shader.fs = FS

                self.bind(pos=self.on_pos)
                self.bind(size=self.on_size)

                self.init_mesh()


            def init_mesh(self):
                x = np.array([0.0, 1.0])
                y = np.linspace(0.0, 1.0, self.RESOLUTION)
                vertices = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x)), np.repeat(y, len(x))])

                vertices = vertices.flatten()

                i = np.arange(self.RESOLUTION) * 2
                indices = np.transpose([i, i+1, i+3, i, i+2, i+3]).flatten()

                vertex_format = [
                    (b'v_pos', 2, 'float'),
                    (b'v_color', 1, 'float'),
                ]

                mesh = Mesh(vertices=vertices, indices=indices, fmt=vertex_format, mode='triangles')

                self.fbo.add(mesh)

            def on_pos(self, instance, value):
                x, y = value
                self.rect.pos = [self.center_x - self.BAR_WIDTH//2, y]

            def on_size(self, instance, value):
                width, height = value
                self.rect.size = [self.BAR_WIDTH, height]

                self.update_glsl()

            def update_glsl(self, *args):
                w, h = self.size
                w = max(w, 1)
                h = max(h, 1)
                proj = Matrix().view_clip(0, w, 0, h, 1, 100, 0)
                self.fbo['projection_mat'] = proj
                self.fbo['scale'] = [float(v) for v in self.size]


        class ColorBarWidget(BoxLayout):

            WIDTH = 80
            LABEL_HEIGHT = 40
            LABEL_COLOR = (0, 0, 0, 1)  # RGBA format

            def __init__(self, padding, U=None, vmin=None, vmax=None):
                super(ColorBarWidget, self).__init__(padding=padding, size_hint_x=None, width=self.WIDTH)
                self.label_min = Label(color=self.LABEL_COLOR, size_hint_y=None, height=self.LABEL_HEIGHT)
                self.label_max = Label(color=self.LABEL_COLOR, size_hint_y=None, height=self.LABEL_HEIGHT)
                self.colorbar = ColorBarFBO()

                super(ColorBarWidget, self).__init__(orientation='vertical')
                self.add_widget(self.label_max)
                self.add_widget(self.colorbar)
                self.add_widget(self.label_min)
                self.set(U, vmin, vmax)

            def build(self):
                return self

            def set(self, U=None, vmin=None, vmax=None):
                self.vmin = vmin if vmin is not None else (np.min(U) if U is not None else 0.0)
                self.vmax = vmax if vmax is not None else (np.max(U) if U is not None else 1.0)

                difference = abs(self.vmax - self.vmin)
                if difference == 0:
                    precision = 3
                else:
                    precision = math.log(max(abs(self.vmin), abs(self.vmax)) / difference, 10)
                    precision = int(min(max(precision, 3), 8))
                vmin_str = format(('{:.' + str(precision) + '}').format(self.vmin))
                vmax_str = format(('{:.' + str(precision) + '}').format(self.vmax))

                self.label_max.text = vmax_str
                self.label_min.text = vmin_str

        return ColorBarWidget(padding=padding, U=U, vmin=vmin, vmax=vmax)


else:
    def getGLPatchWidget(parent, grid, bounding_box=None, vmin=None, vmax=None, codim=2, dpi=100):
        return None

    def getColorBarWidget(padding, U=None, vmin=None, vmax=None):
        return None
