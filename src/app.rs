//! The viewer application: folder loading with progress, orthogonal slice
//! views (axial / coronal / sagittal) and an interactive 3-D rendering tab.

use crate::colormap::Colormap;
use crate::loader;
use crate::render3d::{self, Mat3, RenderMode, RendererState};
use crate::volume::{TextureData, Volume};
use anyhow::Result;
use egui::{Color32, RichText, Sense, Stroke, TextureHandle, TextureOptions};
use std::path::PathBuf;
use std::sync::mpsc::{Receiver, channel};
use std::sync::{Arc, Mutex};

const TEXTURE_RESOLUTIONS: [usize; 6] = [128, 192, 256, 320, 384, 512];

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tab {
    View3D,
    Slices,
}

/// Axial = fixed z, Coronal = fixed y, Sagittal = fixed x.
const SLICE_NAMES: [&str; 3] = ["Axial (Z)", "Coronal (Y)", "Sagittal (X)"];

enum LoadMsg {
    Progress(usize, usize),
    Done(Result<Volume>),
}

struct LoadJob {
    rx: Receiver<LoadMsg>,
    done: usize,
    total: usize,
}

pub struct ViewerApp {
    volume: Option<Arc<Volume>>,
    loading: Option<LoadJob>,
    ds_job: Option<Receiver<TextureData>>,
    status: String,

    // Display settings shared by all views.
    cmap: Colormap,
    wmin: f32,
    wmax: f32,

    tab: Tab,

    // Slice views: current index, cached texture and the (index, wmin, wmax,
    // cmap) key it was built for.
    slice_ix: [usize; 3],
    slice_tex: [Option<TextureHandle>; 3],
    slice_key: [Option<(usize, u32, u32, u8)>; 3],
    slice_readout: [String; 3],

    // 3-D view.
    renderer: Arc<Mutex<RendererState>>,
    rot: Mat3, // volume space -> view space
    zoom: f32,
    pan: [f32; 2],
    mode: RenderMode,
    steps: i32,
    density: f32,
    clip_min: [f32; 3],
    clip_max: [f32; 3],
    tex_res: usize,
    tex_dims: Option<[usize; 3]>,
}

fn initial_rotation() -> Mat3 {
    render3d::mat3_mul(&render3d::rot_x(0.35), &render3d::rot_y(-0.55))
}

impl ViewerApp {
    pub fn new() -> Self {
        Self {
            volume: None,
            loading: None,
            ds_job: None,
            status: "Browse or drop a folder of TIFF slices, or a multi-page TIFF file."
                .to_owned(),
            cmap: Colormap::Gray,
            wmin: 0.0,
            wmax: 1.0,
            tab: Tab::View3D,
            slice_ix: [0; 3],
            slice_tex: [None, None, None],
            slice_key: [None; 3],
            slice_readout: std::array::from_fn(|_| String::new()),
            renderer: RendererState::new(),
            rot: initial_rotation(),
            zoom: 1.5,
            pan: [0.0, 0.0],
            mode: RenderMode::Mip,
            steps: 256,
            density: 12.0,
            clip_min: [0.0; 3],
            clip_max: [1.0; 3],
            tex_res: 256,
            tex_dims: None,
        }
    }

    // ----- loading ----------------------------------------------------------

    /// Start loading `path` — a folder of TIFF slices or a single
    /// (multi-page) TIFF file — on a background thread.
    pub fn start_load(&mut self, path: PathBuf, ctx: &egui::Context) {
        let (tx, rx) = channel();
        let ctx2 = ctx.clone();
        let path2 = path.clone();
        std::thread::spawn(move || {
            let progress_tx = Mutex::new(tx.clone());
            let ctx3 = ctx2.clone();
            let result = loader::load_path(&path2, move |done, total| {
                let _ = progress_tx.lock().unwrap().send(LoadMsg::Progress(done, total));
                ctx3.request_repaint();
            });
            let _ = tx.send(LoadMsg::Done(result));
            ctx2.request_repaint();
        });
        self.loading = Some(LoadJob { rx, done: 0, total: 0 });
        self.status = format!("Loading {}…", path.display());
    }

    fn poll_load(&mut self, ctx: &egui::Context) {
        let mut result = None;
        if let Some(job) = &mut self.loading {
            while let Ok(msg) = job.rx.try_recv() {
                match msg {
                    LoadMsg::Progress(done, total) => {
                        job.done = done;
                        job.total = total;
                    }
                    LoadMsg::Done(res) => result = Some(res),
                }
            }
        }
        if let Some(res) = result {
            self.loading = None;
            match res {
                Ok(vol) => self.apply_volume(vol, ctx),
                Err(e) => self.status = format!("Load failed: {e:#}"),
            }
        }
    }

    fn apply_volume(&mut self, vol: Volume, ctx: &egui::Context) {
        let vol = Arc::new(vol);
        self.wmin = vol.p_low;
        self.wmax = vol.p_high;
        if self.wmax <= self.wmin {
            self.wmax = self.wmin + 1.0;
        }
        self.slice_ix = [vol.nz() / 2, vol.ny() / 2, vol.nx() / 2];
        self.slice_key = [None; 3];
        self.clip_min = [0.0; 3];
        self.clip_max = [1.0; 3];
        self.reset_view();
        {
            let mut r = self.renderer.lock().unwrap();
            r.pending_cmap = Some(self.cmap.lut());
            r.has_volume = false;
        }
        self.tex_dims = None;
        self.status = format!(
            "Loaded {} slices of {}x{} from {}",
            vol.nz(),
            vol.nx(),
            vol.ny(),
            vol.folder.display()
        );
        self.volume = Some(vol);
        self.spawn_downsample(ctx);
    }

    fn spawn_downsample(&mut self, ctx: &egui::Context) {
        let Some(vol) = &self.volume else { return };
        let vol = Arc::clone(vol);
        let res = self.tex_res;
        let ctx = ctx.clone();
        let (tx, rx) = channel();
        std::thread::spawn(move || {
            let td = vol.downsample_for_texture(res);
            let _ = tx.send(td);
            ctx.request_repaint();
        });
        self.ds_job = Some(rx);
    }

    fn poll_downsample(&mut self) {
        let Some(rx) = &self.ds_job else { return };
        if let Ok(td) = rx.try_recv() {
            self.tex_dims = Some(td.dims);
            self.renderer.lock().unwrap().pending_volume = Some(td);
            self.ds_job = None;
        }
    }

    fn browse(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new().set_title("Select a folder of reconstructed TIFF slices");
        if let Some(vol) = &self.volume {
            if let Some(parent) = vol.folder.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(dir) = dialog.pick_folder() {
            self.start_load(dir, ctx);
        }
    }

    fn browse_file(&mut self, ctx: &egui::Context) {
        let mut dialog = rfd::FileDialog::new()
            .set_title("Select a multi-page TIFF file")
            .add_filter("TIFF", loader::SUPPORTED_EXTENSIONS);
        if let Some(vol) = &self.volume {
            if let Some(parent) = vol.folder.parent() {
                dialog = dialog.set_directory(parent);
            }
        }
        if let Some(file) = dialog.pick_file() {
            self.start_load(file, ctx);
        }
    }

    /// Dim the window while TIFFs are dragged over it and start a load when
    /// a folder or TIFF file is dropped.
    fn handle_drag_and_drop(&mut self, ctx: &egui::Context) {
        if ctx.input(|i| !i.raw.hovered_files.is_empty()) {
            let painter = ctx.layer_painter(egui::LayerId::new(
                egui::Order::Foreground,
                egui::Id::new("drop_overlay"),
            ));
            let rect = ctx.content_rect();
            painter.rect_filled(rect, 0.0, Color32::from_black_alpha(160));
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Drop a multi-page TIFF file or a folder of TIFF slices",
                egui::FontId::proportional(22.0),
                Color32::WHITE,
            );
        }

        let dropped: Option<PathBuf> =
            ctx.input(|i| i.raw.dropped_files.iter().find_map(|f| f.path.clone()));
        let Some(path) = dropped else { return };
        if self.loading.is_some() {
            return;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        if path.is_dir() || loader::SUPPORTED_EXTENSIONS.contains(&ext.as_str()) {
            self.start_load(path, ctx);
        } else {
            self.status = format!(
                "Cannot load {}: not a TIFF file or a folder",
                path.display()
            );
        }
    }

    fn reset_view(&mut self) {
        self.rot = initial_rotation();
        self.zoom = 1.5;
        self.pan = [0.0, 0.0];
    }

    // ----- left control panel ----------------------------------------------

    fn ui_controls(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(vol) = self.volume.clone() else { return };

        ui.heading("Data");
        ui.label(RichText::new(vol.folder.display().to_string()).small().weak());
        ui.label(format!(
            "{} × {} × {}  (X × Y × Z)",
            vol.nx(),
            vol.ny(),
            vol.nz()
        ));
        if ui.button("📂 Browse folder…").clicked() {
            self.browse(ctx);
        }
        if ui.button("🗄 Open TIFF file…").clicked() {
            self.browse_file(ctx);
        }

        ui.separator();
        ui.heading("Display");

        let before = self.cmap;
        egui::ComboBox::from_label("Colormap")
            .selected_text(self.cmap.label())
            .show_ui(ui, |ui| {
                for c in Colormap::ALL {
                    ui.selectable_value(&mut self.cmap, c, c.label());
                }
            });
        if before != self.cmap {
            self.renderer.lock().unwrap().pending_cmap = Some(self.cmap.lut());
        }

        let (vmin, vmax) = (vol.vmin, vol.vmax.max(vol.vmin + 1e-30));
        ui.add(
            egui::Slider::new(&mut self.wmin, vmin..=vmax)
                .text("min")
                .smart_aim(false),
        );
        ui.add(
            egui::Slider::new(&mut self.wmax, vmin..=vmax)
                .text("max")
                .smart_aim(false),
        );
        if self.wmax <= self.wmin {
            self.wmax = self.wmin + (vmax - vmin).max(1e-30) * 1e-4;
        }
        ui.horizontal(|ui| {
            if ui.button("Auto contrast").clicked() {
                self.wmin = vol.p_low;
                self.wmax = vol.p_high.max(vol.p_low + 1e-30);
            }
            if ui.button("Full range").clicked() {
                self.wmin = vmin;
                self.wmax = vmax;
            }
        });

        ui.separator();
        ui.heading("3-D rendering");

        for m in RenderMode::ALL {
            ui.radio_value(&mut self.mode, m, m.label());
        }
        ui.add(egui::Slider::new(&mut self.steps, 64..=768).text("Quality (ray steps)"));
        ui.add_enabled(
            self.mode != RenderMode::Mip,
            egui::Slider::new(&mut self.density, 0.5..=60.0)
                .logarithmic(true)
                .text("Density / gain"),
        );

        let before_res = self.tex_res;
        egui::ComboBox::from_label("3-D texture size")
            .selected_text(format!("≤ {}³", self.tex_res))
            .show_ui(ui, |ui| {
                for r in TEXTURE_RESOLUTIONS {
                    ui.selectable_value(&mut self.tex_res, r, format!("≤ {r}³"));
                }
            });
        if before_res != self.tex_res {
            self.spawn_downsample(ctx);
        }
        match (&self.ds_job, self.tex_dims) {
            (Some(_), _) => {
                ui.horizontal(|ui| {
                    ui.spinner();
                    ui.label("Downsampling volume…");
                });
            }
            (None, Some([dx, dy, dz])) => {
                ui.label(RichText::new(format!("3-D texture: {dx} × {dy} × {dz}")).small().weak());
            }
            _ => {}
        }

        egui::CollapsingHeader::new("Clip box")
            .default_open(false)
            .show(ui, |ui| {
                for (axis, label) in ["X", "Y", "Z"].iter().enumerate() {
                    ui.label(*label);
                    ui.add(
                        egui::Slider::new(&mut self.clip_min[axis], 0.0..=1.0).text("min"),
                    );
                    ui.add(
                        egui::Slider::new(&mut self.clip_max[axis], 0.0..=1.0).text("max"),
                    );
                    if self.clip_max[axis] < self.clip_min[axis] {
                        self.clip_max[axis] = self.clip_min[axis];
                    }
                }
                if ui.button("Reset clip").clicked() {
                    self.clip_min = [0.0; 3];
                    self.clip_max = [1.0; 3];
                }
            });

        if ui.button("Reset view").clicked() {
            self.reset_view();
        }
    }

    // ----- 3-D tab ----------------------------------------------------------

    fn ui_3d(&mut self, ui: &mut egui::Ui) {
        let Some(vol) = self.volume.clone() else { return };

        let rect = ui.available_rect_before_wrap();
        if rect.width() < 10.0 || rect.height() < 10.0 {
            return;
        }
        let response = ui.allocate_rect(rect, Sense::click_and_drag());

        // Interaction: left-drag rotates, right-/shift-drag pans,
        // scroll or pinch zooms, double-click resets.
        let shift = ui.input(|i| i.modifiers.shift);
        let drag = response.drag_delta();
        if response.dragged_by(egui::PointerButton::Primary) && !shift {
            let k = 0.008;
            let q = render3d::mat3_mul(&render3d::rot_y(drag.x * k), &render3d::rot_x(drag.y * k));
            self.rot = render3d::mat3_mul(&q, &self.rot);
        } else if response.dragged_by(egui::PointerButton::Secondary)
            || (response.dragged_by(egui::PointerButton::Primary) && shift)
        {
            let s = 2.0 / (rect.height() * self.zoom);
            self.pan[0] -= drag.x * s;
            self.pan[1] += drag.y * s;
        }
        if response.double_clicked() {
            self.reset_view();
        }
        if response.hovered() {
            let (scroll, pinch) = ui.input(|i| (i.smooth_scroll_delta.y, i.zoom_delta()));
            self.zoom = (self.zoom * (scroll * 0.002).exp() * pinch).clamp(0.05, 200.0);
        }

        // Push the parameters for the paint callback.
        let (nx, ny, nz) = (vol.nx() as f32, vol.ny() as f32, vol.nz() as f32);
        let m = nx.max(ny).max(nz);
        {
            let mut r = self.renderer.lock().unwrap();
            r.params = render3d::RenderParams {
                rot_view_to_volume: render3d::mat3_transpose(&self.rot),
                zoom: self.zoom,
                pan: self.pan,
                aspect: rect.width() / rect.height(),
                half: [0.5 * nx / m, 0.5 * ny / m, 0.5 * nz / m],
                wmin: self.wmin,
                wmax: self.wmax,
                mode: self.mode,
                steps: self.steps,
                density: self.density,
                clip_min: self.clip_min,
                clip_max: self.clip_max,
                ..Default::default()
            };
        }

        let shared = Arc::clone(&self.renderer);
        let callback = egui_glow_callback(rect, shared);
        ui.painter().add(callback);

        self.draw_overlays(ui, rect);
    }

    fn draw_overlays(&self, ui: &egui::Ui, rect: egui::Rect) {
        let painter = ui.painter();

        if let Some(err) = &self.renderer.lock().unwrap().error {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                format!("3-D rendering unavailable:\n{err}"),
                egui::FontId::proportional(15.0),
                Color32::LIGHT_RED,
            );
            return;
        }
        if self.ds_job.is_some() && self.tex_dims.is_none() {
            painter.text(
                rect.center(),
                egui::Align2::CENTER_CENTER,
                "Preparing 3-D texture…",
                egui::FontId::proportional(16.0),
                Color32::GRAY,
            );
        }

        // Orientation axes, bottom-left. Box +y is image-up, so the image
        // "Y" (row) axis points along box -y.
        let origin = rect.left_bottom() + egui::vec2(46.0, -46.0);
        let axes: [([f32; 3], Color32, &str); 3] = [
            ([1.0, 0.0, 0.0], Color32::from_rgb(230, 80, 80), "X"),
            ([0.0, -1.0, 0.0], Color32::from_rgb(90, 200, 90), "Y"),
            ([0.0, 0.0, 1.0], Color32::from_rgb(90, 140, 240), "Z"),
        ];
        for (axis, color, label) in axes {
            let v = render3d::mat3_apply(&self.rot, axis);
            let dir = egui::vec2(v[0], -v[1]) * 30.0;
            painter.arrow(origin, dir, Stroke::new(2.0, color));
            painter.text(
                origin + dir + dir.normalized() * 8.0,
                egui::Align2::CENTER_CENTER,
                label,
                egui::FontId::monospace(12.0),
                color,
            );
        }

        painter.text(
            rect.left_top() + egui::vec2(8.0, 6.0),
            egui::Align2::LEFT_TOP,
            "drag: rotate  •  right/shift-drag: pan  •  scroll: zoom  •  double-click: reset",
            egui::FontId::proportional(12.0),
            Color32::from_gray(140),
        );
    }

    // ----- slices tab -------------------------------------------------------

    fn ui_slices(&mut self, ui: &mut egui::Ui) {
        let Some(vol) = self.volume.clone() else { return };
        // The slice images adapt to the window, but the sliders, the readout
        // lines and the images' minimum height can outgrow a short window
        // (small displays, the large-text mode). The panel height is measured
        // before entering the scroll area — inside it the available height is
        // unbounded — and the minimum image height below is what makes the
        // scroll bar appear.
        let panel_h = ui.available_height();
        egui::ScrollArea::vertical()
            .id_salt("slices_scroll")
            .auto_shrink([false, false])
            .show(ui, |ui| self.ui_slices_content(ui, &vol, panel_h));
    }

    fn ui_slices_content(&mut self, ui: &mut egui::Ui, vol: &Arc<Volume>, panel_h: f32) {
        let n_per_view = [vol.nz(), vol.ny(), vol.nx()];
        let lut = self.cmap.lut();

        ui.columns(3, |cols| {
            for view in 0..3 {
                let ui = &mut cols[view];
                ui.vertical_centered(|ui| {
                    ui.strong(SLICE_NAMES[view]);
                });

                let n = n_per_view[view].max(1);
                self.slice_ix[view] = self.slice_ix[view].min(n - 1);
                ui.add(
                    egui::Slider::new(&mut self.slice_ix[view], 0..=n - 1)
                        .text(format!("/ {}", n - 1)),
                );

                // Rebuild the texture if the slice, window or colormap changed.
                let key = (
                    self.slice_ix[view],
                    self.wmin.to_bits(),
                    self.wmax.to_bits(),
                    self.cmap as u8,
                );
                if self.slice_key[view] != Some(key) {
                    let image = slice_color_image(&vol, view, self.slice_ix[view], &lut, self.wmin, self.wmax);
                    self.slice_tex[view] = Some(ui.ctx().load_texture(
                        format!("slice_{view}"),
                        image,
                        TextureOptions::NEAREST,
                    ));
                    self.slice_key[view] = Some(key);
                }

                if let Some(tex) = &self.slice_tex[view] {
                    let img_size = tex.size_vec2();
                    // 30.0 keeps room for the readout line under the image.
                    let avail = egui::vec2(
                        ui.available_width(),
                        (panel_h - ui.min_rect().height() - ui.spacing().item_spacing.y - 30.0)
                            .max(120.0),
                    );
                    let scale = (avail.x / img_size.x)
                        .min(avail.y / img_size.y)
                        .clamp(0.001, 20.0);
                    let size = img_size * scale;
                    let resp = ui
                        .add(egui::Image::new(tex).fit_to_exact_size(size).sense(Sense::hover()));

                    self.slice_readout[view] = "—".to_owned();
                    if let Some(pos) = resp.hover_pos() {
                        let rel = pos - resp.rect.min;
                        let px = ((rel.x / resp.rect.width()) * img_size.x)
                            .clamp(0.0, img_size.x - 1.0) as usize;
                        let py = ((rel.y / resp.rect.height()) * img_size.y)
                            .clamp(0.0, img_size.y - 1.0) as usize;
                        let idx = self.slice_ix[view];
                        let (x, y, z) = match view {
                            0 => (px, py, idx),
                            1 => (px, idx, py),
                            _ => (idx, px, py),
                        };
                        let v = vol.data[[z, y, x]];
                        self.slice_readout[view] = format!("x={x}  y={y}  z={z}   value={v:.6}");
                    }
                    ui.monospace(&self.slice_readout[view]);
                }
            }
        });
    }

    // ----- top-level frames -------------------------------------------------

    fn ui_loading(&self, ui: &mut egui::Ui) {
        let Some(job) = &self.loading else { return };
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.35);
            if job.total > 0 && job.done >= job.total {
                ui.heading("Building volume & statistics…");
                ui.add_space(10.0);
                ui.spinner();
            } else {
                ui.heading("Loading reconstructed slices…");
                ui.add_space(10.0);
                let frac = if job.total > 0 {
                    job.done as f32 / job.total as f32
                } else {
                    0.0
                };
                ui.add(
                    egui::ProgressBar::new(frac)
                        .desired_width(420.0)
                        .show_percentage(),
                );
                ui.label(format!("{} / {} images", job.done, job.total));
            }
        });
    }

    fn ui_welcome(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.vertical_centered(|ui| {
            ui.add_space(ui.available_height() * 0.35);
            ui.heading("VENUS 3-D Volume Viewer");
            ui.add_space(6.0);
            ui.label("Visualize a reconstructed CT volume from a folder of TIFF slices");
            ui.label("or from a single multi-page TIFF file");
            ui.label(RichText::new("(e.g. the output folder of rust_ct_reconstruction)").weak());
            ui.add_space(16.0);
            if ui
                .button(RichText::new("  📂 Browse for a folder…  ").size(16.0))
                .clicked()
            {
                self.browse(ctx);
            }
            ui.add_space(8.0);
            if ui
                .button(RichText::new("  🗄 Browse for a TIFF file…  ").size(16.0))
                .clicked()
            {
                self.browse_file(ctx);
            }
            ui.add_space(16.0);
            ui.label(
                RichText::new("…or drag & drop a folder or TIFF file anywhere in this window")
                    .weak(),
            );
        });
    }
}

fn egui_glow_callback(
    rect: egui::Rect,
    shared: Arc<Mutex<RendererState>>,
) -> egui::PaintCallback {
    let cb = eframe::egui_glow::CallbackFn::new(move |_info, painter| {
        render3d::paint(painter.gl(), &shared);
    });
    egui::PaintCallback {
        rect,
        callback: Arc::new(cb),
    }
}

/// Extract one orthogonal slice and colorize it with the LUT.
/// view 0: axial (w=nx, h=ny), 1: coronal (w=nx, h=nz), 2: sagittal (w=ny, h=nz).
fn slice_color_image(
    vol: &Volume,
    view: usize,
    idx: usize,
    lut: &[[u8; 3]; 256],
    wmin: f32,
    wmax: f32,
) -> egui::ColorImage {
    let (nz, ny, nx) = vol.data.dim();
    let (w, h) = match view {
        0 => (nx, ny),
        1 => (nx, nz),
        _ => (ny, nz),
    };
    let scale = 255.0 / (wmax - wmin).max(1e-30);
    let mut rgb = vec![0u8; w * h * 3];
    let src = vol.data.as_slice().expect("volume is standard layout");

    let mut put = |i: usize, v: f32| {
        let bin = if v.is_finite() {
            ((v - wmin) * scale).clamp(0.0, 255.0) as usize
        } else {
            0
        };
        rgb[i * 3..i * 3 + 3].copy_from_slice(&lut[bin]);
    };

    match view {
        0 => {
            let base = idx.min(nz - 1) * ny * nx;
            for (i, v) in src[base..base + ny * nx].iter().enumerate() {
                put(i, *v);
            }
        }
        1 => {
            let y = idx.min(ny - 1);
            for z in 0..nz {
                let base = (z * ny + y) * nx;
                for x in 0..nx {
                    put(z * nx + x, src[base + x]);
                }
            }
        }
        _ => {
            let x = idx.min(nx - 1);
            for z in 0..nz {
                for y in 0..ny {
                    put(z * ny + y, src[(z * ny + y) * nx + x]);
                }
            }
        }
    }

    egui::ColorImage::from_rgb([w, h], &rgb)
}

impl eframe::App for ViewerApp {
    fn ui(&mut self, ui: &mut egui::Ui, _frame: &mut eframe::Frame) {
        let ctx = ui.ctx().clone();
        self.poll_load(&ctx);
        self.poll_downsample();
        self.handle_drag_and_drop(&ctx);

        egui::Panel::top("top").show(ui, |ui| {
            ui.horizontal(|ui| {
                ui.heading("VENUS 3-D Volume Viewer");
                if self.volume.is_some() {
                    ui.separator();
                    ui.selectable_value(&mut self.tab, Tab::View3D, "3-D volume");
                    ui.selectable_value(&mut self.tab, Tab::Slices, "Slices");
                }
                ui.separator();
                crate::theme::toggle_button(ui);
                crate::zoom::toggle_button(ui);
            });
        });

        egui::Panel::bottom("status").show(ui, |ui| {
            ui.label(RichText::new(&self.status).small());
        });

        if self.volume.is_some() && self.loading.is_none() {
            egui::Panel::left("controls")
                .resizable(false)
                .default_size(240.0)
                .show(ui, |ui| {
                    egui::ScrollArea::vertical().show(ui, |ui| {
                        self.ui_controls(ui, &ctx);
                    });
                });
        }

        egui::CentralPanel::default().show(ui, |ui| {
            if self.loading.is_some() {
                self.ui_loading(ui);
            } else if self.volume.is_none() {
                self.ui_welcome(ui, &ctx);
            } else {
                match self.tab {
                    Tab::View3D => self.ui_3d(ui),
                    Tab::Slices => self.ui_slices(ui),
                }
            }
        });
    }

    fn on_exit(&mut self, gl: Option<&eframe::glow::Context>) {
        if let Some(gl) = gl {
            render3d::destroy(gl, &self.renderer);
        }
    }
}
