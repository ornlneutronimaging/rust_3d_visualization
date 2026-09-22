//! 3-D Volume Viewer — visualize a reconstructed CT volume (a folder of TIFF
//! slices, e.g. the output of `rust_ct_reconstruction`, or a single
//! multi-page TIFF file) as orthogonal slices and an interactive GPU-rendered
//! 3-D volume.

use std::path::PathBuf;
use volume_3d_viewer::app::ViewerApp;
use volume_3d_viewer::loader::{self, Detector};

const USAGE: &str = "\
volume_3d_viewer — 3-D viewer for reconstructed CT volumes

USAGE:
  volume_3d_viewer [OPTIONS] [PATH]

ARGS:
  PATH   Either a folder containing the reconstructed slices as TIFF files,
         e.g. the output folder written by rust_ct_reconstruction
         (image_0000.tiff, image_0001.tiff, … — stacked in sorted filename
         order along Z), or a single multi-page TIFF file (one Z-slice per
         page). When omitted, browse or drag & drop from within the
         application.

OPTIONS:
  --detector <NAME>   Force the detector the slices are loaded as, which
                      decides their orientation: timepix (slices transposed),
                      ccd (flipped vertically), qhy (as-is, not decided yet)
                      or as-is. By default the detector is recognized from
                      the folder layout (images/tpx1, images/ikonxl, …) and
                      reconstructed slices outside those folders are shown
                      as-is; the side panel has a combobox to change it
  -h, --help          Show this help
";

fn parse_args() -> Result<(Option<PathBuf>, Option<Detector>), String> {
    let mut input = None;
    let mut detector = None;
    let mut args = std::env::args().skip(1);
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            "--detector" => {
                let v = args.next().ok_or("--detector requires timepix, ccd, qhy or as-is")?;
                detector = Some(Detector::parse(&v).ok_or_else(|| {
                    format!("invalid --detector '{v}': expected timepix, ccd, qhy or as-is")
                })?);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => {
                if input.is_some() {
                    return Err("Only one input path can be given".to_owned());
                }
                input = Some(PathBuf::from(a));
            }
        }
    }
    Ok((input, detector))
}

fn main() -> eframe::Result<()> {
    let (input, detector) = match parse_args() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    // Surface obvious input errors on stderr before the GUI opens.
    if let Some(path) = &input {
        if path.is_dir() {
            if let Err(e) = loader::list_tiffs_in_dir(path) {
                eprintln!("Error: {e:#}");
                std::process::exit(1);
            }
        } else if !path.is_file() {
            eprintln!("Error: {} does not exist", path.display());
            std::process::exit(1);
        }
    }

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1450.0, 940.0])
            .with_title("VENUS 3-D Volume Viewer"),
        ..Default::default()
    };

    eframe::run_native(
        "VENUS 3-D Volume Viewer",
        native_options,
        Box::new(move |cc| {
            install_fonts(&cc.egui_ctx);
            // Saved light/dark preference, shared by all the VENUS rust
            // tools (dark when none is saved); the toolbar has a toggle.
            cc.egui_ctx.set_theme(volume_3d_viewer::theme::load());
            cc.egui_ctx.set_zoom_factor(volume_3d_viewer::zoom::load());
            let mut app = ViewerApp::new();
            app.set_detector_override(detector);
            if let Some(path) = input {
                app.start_load(path, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}

/// egui's proportional family (Ubuntu-Light + the emoji fonts) has no glyph
/// for the arrows (→ ← ↑ ↓), bullets and similar symbols used in the labels,
/// which then show up as squares; the bundled monospace font Hack has them,
/// so it is appended as the last fallback of the proportional family.
fn install_fonts(ctx: &eframe::egui::Context) {
    let mut fonts = eframe::egui::FontDefinitions::default();
    if let Some(family) = fonts.families.get_mut(&eframe::egui::FontFamily::Proportional) {
        if !family.iter().any(|f| f == "Hack") {
            family.push("Hack".to_owned());
        }
    }
    ctx.set_fonts(fonts);
}
