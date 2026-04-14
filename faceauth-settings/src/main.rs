use gtk4::prelude::*;
use gtk4::{glib, Application, ApplicationWindow, Box as GtkBox, Label, Orientation, SpinButton, Switch};
use serde::{Deserialize, Serialize};
use std::cell::RefCell;
use std::fs;
use std::path::PathBuf;
use std::rc::Rc;

const APP_ID: &str = "io.secureface.FaceAuthSettings";

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Settings {
    enabled: bool,
    timeout_ms: u32,
}

impl Default for Settings {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_ms: 2500,
        }
    }
}

fn main() -> glib::ExitCode {
    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run()
}

fn build_ui(app: &Application) {
    let state = Rc::new(RefCell::new(load_settings().unwrap_or_default()));

    let window = ApplicationWindow::builder()
        .application(app)
        .title("SecureFace 设置")
        .default_width(360)
        .default_height(180)
        .build();

    let root = GtkBox::new(Orientation::Vertical, 12);
    root.set_margin_top(20);
    root.set_margin_bottom(20);
    root.set_margin_start(20);
    root.set_margin_end(20);

    let enabled_label = Label::new(Some("启用人脸认证"));
    enabled_label.set_halign(gtk4::Align::Start);
    root.append(&enabled_label);

    let enabled = Switch::builder().active(state.borrow().enabled).build();
    root.append(&enabled);

    let timeout_label = Label::new(Some("识别超时 (ms)"));
    timeout_label.set_halign(gtk4::Align::Start);
    root.append(&timeout_label);

    let timeout = SpinButton::with_range(500.0, 15000.0, 100.0);
    timeout.set_value(state.borrow().timeout_ms as f64);
    root.append(&timeout);

    window.set_child(Some(&root));
    window.present();

    let state_for_enabled = Rc::clone(&state);
    enabled.connect_active_notify(move |sw| {
        state_for_enabled.borrow_mut().enabled = sw.is_active();
        let _ = save_settings(&state_for_enabled.borrow());
    });

    let state_for_timeout = Rc::clone(&state);
    timeout.connect_value_changed(move |spin| {
        state_for_timeout.borrow_mut().timeout_ms = spin.value() as u32;
        let _ = save_settings(&state_for_timeout.borrow());
    });
}

fn config_path() -> anyhow::Result<PathBuf> {
    let base = xdg::BaseDirectories::with_prefix("secureface")?;
    Ok(base.place_config_file("settings.json")?)
}

fn load_settings() -> anyhow::Result<Settings> {
    let path = config_path()?;
    let data = fs::read_to_string(path)?;
    Ok(serde_json::from_str(&data)?)
}

fn save_settings(settings: &Settings) -> anyhow::Result<()> {
    let path = config_path()?;
    let data = serde_json::to_string_pretty(settings)?;
    fs::write(path, data)?;
    Ok(())
}
