//! 3-D Volume Viewer — visualize a reconstructed CT volume (a folder of TIFF
//! slices, e.g. the output of `rust_ct_reconstruction`, or a single
//! multi-page TIFF file) as orthogonal slices and an interactive GPU-rendered
//! 3-D volume.

use std::path::PathBuf;
use volume_3d_viewer::app::ViewerApp;
use volume_3d_viewer::loader;

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
  -h, --help   Show this help
";

fn parse_args() -> Result<Option<PathBuf>, String> {
    let mut input = None;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
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
    Ok(input)
}

fn main() -> eframe::Result<()> {
    let input = match parse_args() {
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
            // Saved light/dark preference, shared by all the VENUS rust
            // tools (dark when none is saved); the toolbar has a toggle.
            cc.egui_ctx.set_theme(volume_3d_viewer::theme::load());
            let mut app = ViewerApp::new();
            if let Some(path) = input {
                app.start_load(path, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
