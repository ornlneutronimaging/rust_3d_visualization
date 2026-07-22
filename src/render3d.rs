//! GPU volume rendering: orthographic raycasting of a 3-D texture.
//!
//! The UI thread owns a `RendererState` behind a mutex: it writes the render
//! parameters every frame and queues (down-sampled) volume / colormap data for
//! upload. All GL work happens inside the egui paint callback, on the GL
//! thread, via [`paint`].
//!
//! Coordinate conventions: the volume box is axis-aligned and centered at the
//! origin, with half-extents proportional to the voxel dimensions (largest
//! axis = 0.5). Box +x is image column, box +y is image row 0 upwards (the y
//! flip happens at texture sampling so slices appear as in the slice views),
//! box +z is the slice index. The camera is orthographic, looking down -z in
//! view space; `rot_view_to_volume` maps view space into the volume box.

use crate::volume::TextureData;
use eframe::glow::{self, HasContext};
use std::sync::{Arc, Mutex};

// ----- small column-major 3x3 matrix helpers --------------------------------

pub type Mat3 = [f32; 9];

pub fn mat3_identity() -> Mat3 {
    [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
}

pub fn mat3_mul(a: &Mat3, b: &Mat3) -> Mat3 {
    let mut m = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            m[j * 3 + i] = (0..3).map(|k| a[k * 3 + i] * b[j * 3 + k]).sum();
        }
    }
    m
}

pub fn mat3_transpose(m: &Mat3) -> Mat3 {
    let mut t = [0.0; 9];
    for j in 0..3 {
        for i in 0..3 {
            t[j * 3 + i] = m[i * 3 + j];
        }
    }
    t
}

pub fn mat3_apply(m: &Mat3, v: [f32; 3]) -> [f32; 3] {
    [
        m[0] * v[0] + m[3] * v[1] + m[6] * v[2],
        m[1] * v[0] + m[4] * v[1] + m[7] * v[2],
        m[2] * v[0] + m[5] * v[1] + m[8] * v[2],
    ]
}

pub fn rot_x(a: f32) -> Mat3 {
    let (s, c) = a.sin_cos();
    [1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c]
}

pub fn rot_y(a: f32) -> Mat3 {
    let (s, c) = a.sin_cos();
    [c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c]
}

// ----- render parameters ----------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq)]
pub enum RenderMode {
    Mip,
    Composite,
    Xray,
}

impl RenderMode {
    pub const ALL: [RenderMode; 3] = [RenderMode::Mip, RenderMode::Composite, RenderMode::Xray];

    pub fn label(self) -> &'static str {
        match self {
            RenderMode::Mip => "Max intensity (MIP)",
            RenderMode::Composite => "Composite",
            RenderMode::Xray => "X-ray (mean)",
        }
    }

    fn id(self) -> i32 {
        match self {
            RenderMode::Mip => 0,
            RenderMode::Composite => 1,
            RenderMode::Xray => 2,
        }
    }
}

#[derive(Clone)]
pub struct RenderParams {
    pub rot_view_to_volume: Mat3,
    pub zoom: f32,
    pub pan: [f32; 2],
    /// Viewport width / height.
    pub aspect: f32,
    /// Half-extents of the volume box (largest = 0.5).
    pub half: [f32; 3],
    pub wmin: f32,
    pub wmax: f32,
    pub mode: RenderMode,
    pub steps: i32,
    /// Opacity scale (Composite) / brightness gain (X-ray).
    pub density: f32,
    /// Clip box as fractions of the full box, per axis.
    pub clip_min: [f32; 3],
    pub clip_max: [f32; 3],
    pub bg: [f32; 3],
}

impl Default for RenderParams {
    fn default() -> Self {
        Self {
            rot_view_to_volume: mat3_identity(),
            zoom: 1.5,
            pan: [0.0, 0.0],
            aspect: 1.0,
            half: [0.5, 0.5, 0.5],
            wmin: 0.0,
            wmax: 1.0,
            mode: RenderMode::Mip,
            steps: 256,
            density: 12.0,
            clip_min: [0.0, 0.0, 0.0],
            clip_max: [1.0, 1.0, 1.0],
            bg: [0.08, 0.08, 0.09],
        }
    }
}

// ----- shared state between UI thread and GL thread -------------------------

pub struct RendererState {
    pub params: RenderParams,
    pub pending_volume: Option<TextureData>,
    pub pending_cmap: Option<[[u8; 3]; 256]>,
    pub error: Option<String>,
    pub has_volume: bool,
    gl: Option<GlObjects>,
}

impl RendererState {
    pub fn new() -> Arc<Mutex<RendererState>> {
        Arc::new(Mutex::new(RendererState {
            params: RenderParams::default(),
            pending_volume: None,
            pending_cmap: None,
            error: None,
            has_volume: false,
            gl: None,
        }))
    }
}

/// Called from inside the egui paint callback (GL thread).
pub fn paint(gl: &glow::Context, shared: &Arc<Mutex<RendererState>>) {
    let mut s = shared.lock().unwrap();
    if s.error.is_some() {
        return;
    }
    if s.gl.is_none() {
        match GlObjects::new(gl) {
            Ok(objs) => s.gl = Some(objs),
            Err(e) => {
                s.error = Some(e);
                return;
            }
        }
    }
    let state = &mut *s;
    let objs = state.gl.as_mut().expect("initialized above");
    if let Some(td) = state.pending_volume.take() {
        objs.upload_volume(gl, &td);
        state.has_volume = true;
    }
    if let Some(lut) = state.pending_cmap.take() {
        objs.upload_cmap(gl, &lut);
    }
    if state.has_volume {
        objs.draw(gl, &state.params);
    }
}

/// Free the GL objects; called from `App::on_exit`.
pub fn destroy(gl: &glow::Context, shared: &Arc<Mutex<RendererState>>) {
    let mut s = shared.lock().unwrap();
    if let Some(objs) = s.gl.take() {
        objs.destroy(gl);
    }
}

// ----- GL objects -----------------------------------------------------------

struct GlObjects {
    program: glow::Program,
    vao: glow::VertexArray,
    vol_tex: glow::Texture,
    cmap_tex: glow::Texture,
}

impl GlObjects {
    fn new(gl: &glow::Context) -> Result<Self, String> {
        unsafe {
            let program = gl.create_program().map_err(|e| format!("create program: {e}"))?;
            let shaders = [
                (glow::VERTEX_SHADER, VERTEX_SHADER_SRC),
                (glow::FRAGMENT_SHADER, FRAGMENT_SHADER_SRC),
            ];
            let mut compiled = Vec::new();
            for (kind, src) in shaders {
                let shader = gl.create_shader(kind).map_err(|e| format!("create shader: {e}"))?;
                gl.shader_source(shader, src);
                gl.compile_shader(shader);
                if !gl.get_shader_compile_status(shader) {
                    return Err(format!(
                        "shader compile error: {}",
                        gl.get_shader_info_log(shader)
                    ));
                }
                gl.attach_shader(program, shader);
                compiled.push(shader);
            }
            gl.link_program(program);
            if !gl.get_program_link_status(program) {
                return Err(format!("program link error: {}", gl.get_program_info_log(program)));
            }
            for shader in compiled {
                gl.detach_shader(program, shader);
                gl.delete_shader(shader);
            }

            let vao = gl
                .create_vertex_array()
                .map_err(|e| format!("create vao: {e}"))?;
            let vol_tex = gl.create_texture().map_err(|e| format!("create texture: {e}"))?;
            let cmap_tex = gl.create_texture().map_err(|e| format!("create texture: {e}"))?;

            Ok(Self {
                program,
                vao,
                vol_tex,
                cmap_tex,
            })
        }
    }

    fn upload_volume(&mut self, gl: &glow::Context, td: &TextureData) {
        let [dx, dy, dz] = td.dims;
        unsafe {
            gl.bind_texture(glow::TEXTURE_3D, Some(self.vol_tex));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 4);
            gl.tex_image_3d(
                glow::TEXTURE_3D,
                0,
                glow::R32F as i32,
                dx as i32,
                dy as i32,
                dz as i32,
                0,
                glow::RED,
                glow::FLOAT,
                glow::PixelUnpackData::Slice(Some(bytemuck::cast_slice(&td.data))),
            );
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_R, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_3D, param, value as i32);
            }
            gl.bind_texture(glow::TEXTURE_3D, None);
        }
    }

    fn upload_cmap(&mut self, gl: &glow::Context, lut: &[[u8; 3]; 256]) {
        let bytes: Vec<u8> = lut.iter().flatten().copied().collect();
        unsafe {
            gl.bind_texture(glow::TEXTURE_2D, Some(self.cmap_tex));
            gl.pixel_store_i32(glow::UNPACK_ALIGNMENT, 1);
            gl.tex_image_2d(
                glow::TEXTURE_2D,
                0,
                glow::RGB8 as i32,
                256,
                1,
                0,
                glow::RGB,
                glow::UNSIGNED_BYTE,
                glow::PixelUnpackData::Slice(Some(&bytes)),
            );
            for (param, value) in [
                (glow::TEXTURE_MIN_FILTER, glow::LINEAR),
                (glow::TEXTURE_MAG_FILTER, glow::LINEAR),
                (glow::TEXTURE_WRAP_S, glow::CLAMP_TO_EDGE),
                (glow::TEXTURE_WRAP_T, glow::CLAMP_TO_EDGE),
            ] {
                gl.tex_parameter_i32(glow::TEXTURE_2D, param, value as i32);
            }
            gl.bind_texture(glow::TEXTURE_2D, None);
        }
    }

    fn draw(&self, gl: &glow::Context, p: &RenderParams) {
        unsafe {
            gl.disable(glow::DEPTH_TEST);
            gl.disable(glow::BLEND);
            gl.disable(glow::CULL_FACE);
            gl.use_program(Some(self.program));

            gl.active_texture(glow::TEXTURE0);
            gl.bind_texture(glow::TEXTURE_3D, Some(self.vol_tex));
            gl.active_texture(glow::TEXTURE1);
            gl.bind_texture(glow::TEXTURE_2D, Some(self.cmap_tex));

            let loc = |name: &str| gl.get_uniform_location(self.program, name);
            gl.uniform_1_i32(loc("u_volume").as_ref(), 0);
            gl.uniform_1_i32(loc("u_cmap").as_ref(), 1);
            gl.uniform_matrix_3_f32_slice(loc("u_rot").as_ref(), false, &p.rot_view_to_volume);
            gl.uniform_1_f32(loc("u_zoom").as_ref(), p.zoom.max(1e-3));
            gl.uniform_2_f32(loc("u_pan").as_ref(), p.pan[0], p.pan[1]);
            gl.uniform_1_f32(loc("u_aspect").as_ref(), p.aspect.max(1e-3));
            gl.uniform_3_f32(loc("u_half").as_ref(), p.half[0], p.half[1], p.half[2]);
            gl.uniform_1_f32(loc("u_wmin").as_ref(), p.wmin);
            gl.uniform_1_f32(loc("u_wmax").as_ref(), p.wmax);
            gl.uniform_1_i32(loc("u_mode").as_ref(), p.mode.id());
            gl.uniform_1_i32(loc("u_steps").as_ref(), p.steps.clamp(8, 4096));
            gl.uniform_1_f32(loc("u_density").as_ref(), p.density);
            gl.uniform_3_f32(
                loc("u_clip_min").as_ref(),
                p.clip_min[0],
                p.clip_min[1],
                p.clip_min[2],
            );
            gl.uniform_3_f32(
                loc("u_clip_max").as_ref(),
                p.clip_max[0],
                p.clip_max[1],
                p.clip_max[2],
            );
            gl.uniform_3_f32(loc("u_bg").as_ref(), p.bg[0], p.bg[1], p.bg[2]);

            gl.bind_vertex_array(Some(self.vao));
            gl.draw_arrays(glow::TRIANGLES, 0, 3);
            gl.bind_vertex_array(None);
        }
    }

    fn destroy(self, gl: &glow::Context) {
        unsafe {
            gl.delete_program(self.program);
            gl.delete_vertex_array(self.vao);
            gl.delete_texture(self.vol_tex);
            gl.delete_texture(self.cmap_tex);
        }
    }
}

// ----- shaders --------------------------------------------------------------

const VERTEX_SHADER_SRC: &str = r#"#version 330 core
out vec2 v_ndc;
void main() {
    // Fullscreen triangle from gl_VertexID: (-1,-1), (3,-1), (-1,3).
    vec2 pos = vec2(float((gl_VertexID << 1) & 2), float(gl_VertexID & 2)) * 2.0 - 1.0;
    v_ndc = pos;
    gl_Position = vec4(pos, 0.0, 1.0);
}
"#;

const FRAGMENT_SHADER_SRC: &str = r#"#version 330 core
in vec2 v_ndc;
out vec4 frag_color;

uniform sampler3D u_volume;
uniform sampler2D u_cmap;
uniform mat3  u_rot;      // view space -> volume space
uniform float u_zoom;
uniform vec2  u_pan;      // view-space pan
uniform float u_aspect;   // viewport w/h
uniform vec3  u_half;     // half-extents of the volume box
uniform float u_wmin;
uniform float u_wmax;
uniform int   u_mode;     // 0 = MIP, 1 = composite, 2 = x-ray mean
uniform int   u_steps;
uniform float u_density;
uniform vec3  u_clip_min;
uniform vec3  u_clip_max;
uniform vec3  u_bg;

float sample_norm(vec3 p) {
    vec3 tc = p / (2.0 * u_half) + 0.5;
    float v = texture(u_volume, vec3(tc.x, 1.0 - tc.y, tc.z)).r;
    if (isnan(v)) v = u_wmin;
    return clamp((v - u_wmin) / max(u_wmax - u_wmin, 1e-30), 0.0, 1.0);
}

void main() {
    // Orthographic camera looking down -z in view space, square pixels.
    vec2 view_xy = v_ndc * vec2(u_aspect, 1.0);
    vec3 ro = u_rot * vec3(view_xy / u_zoom + u_pan, 2.0);
    vec3 rd = normalize(u_rot * vec3(0.0, 0.0, -1.0));

    vec3 bmin = mix(-u_half, u_half, u_clip_min);
    vec3 bmax = mix(-u_half, u_half, u_clip_max);

    vec3 rd_safe = vec3(
        abs(rd.x) < 1e-6 ? 1e-6 : rd.x,
        abs(rd.y) < 1e-6 ? 1e-6 : rd.y,
        abs(rd.z) < 1e-6 ? 1e-6 : rd.z);
    vec3 inv = 1.0 / rd_safe;
    vec3 t0 = (bmin - ro) * inv;
    vec3 t1 = (bmax - ro) * inv;
    vec3 tsm = min(t0, t1);
    vec3 tbg = max(t0, t1);
    float tn = max(max(tsm.x, tsm.y), tsm.z);
    float tf = min(min(tbg.x, tbg.y), tbg.z);
    if (tf <= tn) {
        frag_color = vec4(u_bg, 1.0);
        return;
    }

    float dt = (tf - tn) / float(u_steps);
    float t = tn + 0.5 * dt;

    if (u_mode == 0) {
        float m = 0.0;
        for (int i = 0; i < u_steps; i++) {
            m = max(m, sample_norm(ro + rd * t));
            t += dt;
        }
        frag_color = vec4(texture(u_cmap, vec2(m, 0.5)).rgb, 1.0);
    } else if (u_mode == 1) {
        vec3 acc = vec3(0.0);
        float aacc = 0.0;
        for (int i = 0; i < u_steps; i++) {
            float v = sample_norm(ro + rd * t);
            float a = 1.0 - exp(-v * u_density * dt);
            vec3 c = texture(u_cmap, vec2(v, 0.5)).rgb;
            acc += (1.0 - aacc) * a * c;
            aacc += (1.0 - aacc) * a;
            if (aacc > 0.995) break;
            t += dt;
        }
        frag_color = vec4(acc + (1.0 - aacc) * u_bg, 1.0);
    } else {
        float s = 0.0;
        for (int i = 0; i < u_steps; i++) {
            s += sample_norm(ro + rd * t);
            t += dt;
        }
        float m = clamp(s / float(u_steps) * u_density * 0.25, 0.0, 1.0);
        frag_color = vec4(texture(u_cmap, vec2(m, 0.5)).rgb, 1.0);
    }
}
"#;
