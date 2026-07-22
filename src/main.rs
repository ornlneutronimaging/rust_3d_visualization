//! 3-D Volume Viewer — visualize a reconstructed CT volume (a folder of TIFF
//! slices, e.g. the output of `rust_ct_reconstruction`) as orthogonal slices
//! and an interactive GPU-rendered 3-D volume.

use std::path::PathBuf;
use volume_3d_viewer::app::ViewerApp;
use volume_3d_viewer::loader;

const USAGE: &str = "\
volume_3d_viewer — 3-D viewer for reconstructed CT volumes

USAGE:
  volume_3d_viewer [OPTIONS] [FOLDER]

ARGS:
  FOLDER   Folder containing the reconstructed slices as TIFF files, e.g. the
           output folder written by rust_ct_reconstruction
           (image_0000.tiff, image_0001.tiff, …). Files are stacked in sorted
           filename order along Z. When omitted, browse for a folder from
           within the application.

OPTIONS:
  -h, --help   Show this help
";

fn parse_args() -> Result<Option<PathBuf>, String> {
    let mut folder = None;
    for a in std::env::args().skip(1) {
        match a.as_str() {
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            s if s.starts_with('-') => return Err(format!("Unknown option: {s}")),
            _ => {
                if folder.is_some() {
                    return Err("Only one input folder can be given".to_owned());
                }
                folder = Some(PathBuf::from(a));
            }
        }
    }
    Ok(folder)
}

fn main() -> eframe::Result<()> {
    let folder = match parse_args() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Error: {e}\n\n{USAGE}");
            std::process::exit(2);
        }
    };

    // Surface obvious input errors on stderr before the GUI opens.
    if let Some(dir) = &folder {
        if !dir.is_dir() {
            eprintln!("Error: {} is not a folder", dir.display());
            std::process::exit(1);
        }
        if let Err(e) = loader::list_tiffs_in_dir(dir) {
            eprintln!("Error: {e:#}");
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
            // Always use the dark theme, regardless of the system/desktop theme.
            cc.egui_ctx.set_theme(egui::Theme::Dark);
            let mut app = ViewerApp::new();
            if let Some(dir) = folder {
                app.start_load(dir, &cc.egui_ctx);
            }
            Ok(Box::new(app))
        }),
    )
}
