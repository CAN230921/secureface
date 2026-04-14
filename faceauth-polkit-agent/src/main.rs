use gtk4::prelude::*;
use gtk4::{glib, Application, ApplicationWindow, Box as GtkBox, Button, Label, Orientation};
use std::rc::Rc;

const APP_ID: &str = "io.secureface.FaceAuthPolkitAgent";
const BUS_NAME: &str = "io.secureface.FaceAuth";
const OBJECT_PATH: &str = "/io/secureface/FaceAuth";
const INTERFACE: &str = "io.secureface.FaceAuth";

fn main() -> glib::ExitCode {
    let app = Application::builder().application_id(APP_ID).build();
    app.connect_activate(build_ui);
    app.run()
}

fn build_ui(app: &Application) {
    let user = std::env::var("USER").unwrap_or_else(|_| "unknown".to_string());
    let reason = std::env::var("FACEAUTH_REASON").unwrap_or_else(|_| "polkit".to_string());

    let runtime = Rc::new(
        tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("failed to create tokio runtime"),
    );

    let window = ApplicationWindow::builder()
        .application(app)
        .title("SecureFace Authentication")
        .default_width(420)
        .default_height(180)
        .build();

    let root = GtkBox::new(Orientation::Vertical, 12);
    root.set_margin_top(20);
    root.set_margin_bottom(20);
    root.set_margin_start(20);
    root.set_margin_end(20);

    let title = Label::new(Some("请面对摄像头完成身份验证"));
    title.set_wrap(true);
    root.append(&title);

    let status = Label::new(Some("正在等待验证…"));
    status.set_wrap(true);
    root.append(&status);

    let fallback_btn = Button::with_label("改用密码");
    root.append(&fallback_btn);

    window.set_child(Some(&root));
    window.present();

    let status_clone = status.clone();
    let window_clone = window.clone();
    let user_clone = user.clone();
    let reason_clone = reason.clone();
    let runtime_clone = Rc::clone(&runtime);

    glib::idle_add_local_once(move || {
        let result = runtime_clone.block_on(async {
            let conn = zbus::Connection::system().await?;
            let proxy = zbus::Proxy::new(&conn, BUS_NAME, OBJECT_PATH, INTERFACE).await?;
            let reply: (String, bool) = proxy
                .call("Authenticate", &(user_clone, reason_clone, 2500_u32))
                .await?;
            Ok::<(String, bool), anyhow::Error>(reply)
        });

        match result {
            Ok((engine_result, fallback)) if engine_result == "PASS" && !fallback => {
                status_clone.set_text("人脸验证通过，已授权。");
                window_clone.close();
            }
            Ok((engine_result, _)) => {
                status_clone.set_text(&format!(
                    "人脸验证结果：{}。请点击“改用密码”继续。",
                    engine_result
                ));
            }
            Err(e) => {
                status_clone.set_text(&format!("人脸服务不可用（{}），请改用密码。", e));
            }
        }
    });

    let window_weak = window.downgrade();
    fallback_btn.connect_clicked(move |_| {
        if let Some(window) = window_weak.upgrade() {
            window.close();
        }
    });
}
