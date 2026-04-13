"""
gl_widget.py — OpenGL widget for real-time holographic frame display.

Renders a single float32 greyscale frame as a full-screen textured quad.
The fragment shader linearly maps [dataMin, dataMax] → [0, 1] before display.

Thread safety
─────────────
update_frame() may be called from any thread.  It only stores the new array
and sets a dirty flag; all OpenGL work happens in paintGL(), which is always
called on the Qt GUI thread with the context already current.
"""

import numpy as np
import OpenGL.GL as gl
from OpenGL.GL import shaders
from PySide6.QtOpenGLWidgets import QOpenGLWidget


_VERTEX_SRC = """
    #version 330 core
    layout (location = 0) in vec2 position;
    layout (location = 1) in vec2 texCoord;
    out vec2 vTexCoord;
    void main() {
        gl_Position = vec4(position, 0.0, 1.0);
        vTexCoord = texCoord;
    }
"""

_FRAGMENT_SRC = """
    #version 330 core
    in vec2 vTexCoord;
    out vec4 FragColor;
    uniform sampler2D frameTexture;
    uniform float dataMin;
    uniform float dataMax;
    void main() {
        float val = texture(frameTexture, vTexCoord).r;
        float normalized = clamp((val - dataMin) / (dataMax - dataMin), 0.0, 1.0);
        FragColor = vec4(vec3(normalized), 1.0);
    }
"""

# Full-screen quad: two triangles as a TRIANGLE_STRIP.
# Each vertex is (x, y, u, v) in NDC / texture space.
_QUAD_VERTICES = np.array(
    [
        #  x      y     u    v
        -1.0,  -1.0,  0.0, 0.0,
         1.0,  -1.0,  1.0, 0.0,
        -1.0,   1.0,  0.0, 1.0,
         1.0,   1.0,  1.0, 1.0,
    ],
    dtype=np.float32,
)


class HoloGLWidget(QOpenGLWidget):
    """OpenGL widget that displays a single float32 greyscale frame."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._texture_id = None
        self._shader_program = None
        self._vao = None
        self._vbo = None
        self._tex_width = 0
        self._tex_height = 0
        self._next_frame: np.ndarray | None = None
        self._is_dirty = False

    def initializeGL(self) -> None:
        gl.glClearColor(0.0, 0.0, 0.0, 1.0)

        vs = shaders.compileShader(_VERTEX_SRC, gl.GL_VERTEX_SHADER)
        fs = shaders.compileShader(_FRAGMENT_SRC, gl.GL_FRAGMENT_SHADER)
        self._shader_program = shaders.compileProgram(vs, fs)

        self._vao = gl.glGenVertexArrays(1)
        self._vbo = gl.glGenBuffers(1)
        gl.glBindVertexArray(self._vao)
        gl.glBindBuffer(gl.GL_ARRAY_BUFFER, self._vbo)
        gl.glBufferData(gl.GL_ARRAY_BUFFER, _QUAD_VERTICES.nbytes, _QUAD_VERTICES, gl.GL_STATIC_DRAW)

        stride = 4 * _QUAD_VERTICES.itemsize  # 4 floats per vertex
        gl.glVertexAttribPointer(0, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(0))
        gl.glEnableVertexAttribArray(0)
        gl.glVertexAttribPointer(1, 2, gl.GL_FLOAT, gl.GL_FALSE, stride, gl.ctypes.c_void_p(2 * _QUAD_VERTICES.itemsize))
        gl.glEnableVertexAttribArray(1)

        self._texture_id = gl.glGenTextures(1)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)

    def resizeGL(self, w: int, h: int) -> None:
        gl.glViewport(0, 0, w, h)

    def paintGL(self) -> None:
        if self._next_frame is None:
            return

        if self._is_dirty:
            self._upload_texture()
            self._is_dirty = False

        gl.glClear(gl.GL_COLOR_BUFFER_BIT)
        gl.glUseProgram(self._shader_program)
        gl.glUniform1f(gl.glGetUniformLocation(self._shader_program, "dataMin"), 0.0)
        gl.glUniform1f(gl.glGetUniformLocation(self._shader_program, "dataMax"), 1.0)
        gl.glActiveTexture(gl.GL_TEXTURE0)
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        gl.glBindVertexArray(self._vao)
        gl.glDrawArrays(gl.GL_TRIANGLE_STRIP, 0, 4)

    def _upload_texture(self) -> None:
        """Transfer the pending frame from RAM to VRAM."""
        h, w = self._next_frame.shape
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture_id)
        if w != self._tex_width or h != self._tex_height:
            gl.glTexImage2D(
                gl.GL_TEXTURE_2D, 0, gl.GL_R32F, w, h, 0,
                gl.GL_RED, gl.GL_FLOAT, self._next_frame,
            )
            self._tex_width, self._tex_height = w, h
        else:
            gl.glTexSubImage2D(
                gl.GL_TEXTURE_2D, 0, 0, 0, w, h,
                gl.GL_RED, gl.GL_FLOAT, self._next_frame,
            )

    def update_frame(self, frame_data: np.ndarray) -> None:
        """
        Store a new frame for the next paint event.

        Safe to call from any thread.  No OpenGL work is done here.
        """
        self._next_frame = np.ascontiguousarray(frame_data, dtype=np.float32)
        self._is_dirty = True
        self.update()  # schedules a paint event on the GUI thread
