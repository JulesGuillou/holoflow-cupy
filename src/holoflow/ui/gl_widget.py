import numpy as np
from PySide6.QtOpenGLWidgets import QOpenGLWidget
from PySide6.QtGui import QSurfaceFormat
from PySide6.QtCore import Qt
import OpenGL.GL as gl
from OpenGL.GL import shaders


class HoloGLWidget(QOpenGLWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.texture_id = None
        self.shader_program = None
        self.vao = None
        self.vbo = None
        self.width_tex = 0
        self.height_tex = 0

        # State for the frame to be rendered
        self.next_frame = None
        self.is_dirty = False

    def initializeGL(self):
        """Called once when the OpenGL context is initialized."""
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        # 1. Compile Shaders
        vertex_src = """
            #version 330 core
            layout (location = 0) in vec2 position;
            layout (location = 1) in vec2 texCoord;
            out vec2 vTexCoord;
            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
                vTexCoord = texCoord;
            }
        """
        fragment_src = """
            #version 330 core
            in vec2 vTexCoord;
            out vec4 FragColor;
            uniform sampler2D frameTexture;
            uniform float dataMin;
            uniform float dataMax;
            
            void main() {
                float val = texture(frameTexture, vTexCoord).r;
                float normalized = (val - dataMin) / (dataMax - dataMin);
                normalized = clamp(normalized, 0.0, 1.0);
                FragColor = vec4(vec3(normalized), 1.0);
            }
        """
        vs = shaders.compileShader(vertex_src, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(fragment_src, gl.GL_FRAGMENT_SHADER)
        self.shader_program = shaders.compileProgram(vs, fs)

        # 2. Setup Geometry (Fullscreen Quad)
        vertices = np.array(
            [
                -1.0,
                -1.0,
                0.0,
                0.0,
                1.0,
                -1.0,
                1.0,
                0.0,
                -1.0,
                1.0,
                0.0,
                1.0,
                1.0,
                1.0,
                1.0,
                1.0,
            ],
            dtype=np.float32,
        )

        self.vao = gl.glGenVertexArrays(1)
        self.vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self.vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self.vbo)
        gl.glBufferData(
            gl.GL_ARRAY_BUFFER, vertices.nbytes, vertices, gl.GL_STATIC_DRAW
        )

        # Layout: Pos(0) = 2 floats, Tex(1) = 2 floats (Stride 16 bytes)
        gl.glVertexAttribPointer(
            0, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(0)
        )
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(
            1, 2, gl.GL_FLOAT, gl.GL_FALSE, 16, gl.ctypes.c_void_p(8)
        )
        gl.glEnableVertexAttribArray(1)

        # 3. Setup Texture
        self.texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def resizeGL(self, w, h):
        gl.glViewport(0, 0, w, h)

    def paintGL(self):
        """Called by Qt's event loop. Context is ALREADY current here."""
        if self.next_frame is None:
            return

        # 1. Upload only if we have new data
        if self.is_dirty:
            self._upload_texture()
            self.is_dirty = False

        # 2. Render
        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self.shader_program)

        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, "dataMin"), 0.0)
        gl.glUniform1f(gl.glGetUniformLocation(self.shader_program, "dataMax"), 1.0)

        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)
        gl.glBindVertexArray(self.vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    def _upload_texture(self):
        """Moves pixels from RAM to VRAM."""
        h, w = self.next_frame.shape
        gl.glBindTexture(gl.GL_TEXTURE_2D, self.texture_id)

        if w != self.width_tex or h != self.height_tex:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D,
                0,
                gl.GL_R32F,
                w,
                h,
                0,
                gl.GL_RED,
                gl.GL_FLOAT,
                self.next_frame,
            )
            self.width_tex, self.height_tex = w, h
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, w, h, gl.GL_RED, gl.GL_FLOAT, self.next_frame
            )

    def update_frame(self, frame_data: np.ndarray):
        """
        Receives the raw frame. Call this from your main loop.
        No OpenGL work happens here, so it won't stall the GIL.
        """
        # Ensure data is in the right format for GL
        self.next_frame = np.ascontiguousarray(frame_data, dtype=np.float32)
        self.is_dirty = True

        # Tell Qt to schedule a paint event (non-blocking)
        self.update()
