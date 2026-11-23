//! The panels (stats overlay, messages, menu) draw, at a sensible size for the screen, and grow with it.
//! SP_PRESENT_DUMP=<folder> also writes them there as raw BGRA (w x h in the name) to look at.

use sp_present::display::FixedSettings;
use sp_present::menu::{Menu, Nav};
use sp_present::ui::{self, Image, MenuDrawer};

fn dump(name: &str, img: &Image) {
    if let Ok(dir) = std::env::var("SP_PRESENT_DUMP") {
        let _ = std::fs::write(format!("{dir}/{name}.{}x{}.bgra", img.w, img.h), &img.bgra);
    }
}

fn ink(img: &Image) -> usize {
    img.bgra.as_chunks::<4>().0.iter().filter(|p| p[0] > 150 && p[1] > 150 && p[2] > 150).count()
}

#[test]
fn the_overlays_and_messages_draw() {
    let parts = vec![("120 fps".to_string(), "main".to_string()), ("game 30".to_string(), "dim".to_string()),
                     ("41 ms".to_string(), "warn".to_string())];
    for ui in [0.8, 1.6, 3.0] {
        let m = ui::mini(&parts, "good", "", ui);
        assert!(m.w > 100.0 as usize && m.h > 10 && ink(&m) > 20, "{ui}: {}x{}", m.w, m.h);
        dump(&format!("mini{ui}"), &m);
        let m2 = ui::mini(&parts, "behind", "test: PSNR 34.2 dB", ui);
        assert!(m2.h > m.h);
        let d = ui::detail("120 fps  AI 3.1 ms  FSR 1 3200x1800\nYour Mac keeps up", ui, 0);
        assert!(d.h > m.h && ink(&d) > 20);
        assert_eq!(ui::detail("1", ui, d.w).w, d.w); // only grows
        dump(&format!("detail{ui}"), &d);
        let p = ui::pill("No picture from the PS5 for 3 s. Esc quits\nCheck chiaki-ng", ui);
        assert!(p.w > p.h && ink(&p) > 50);
        dump(&format!("pill{ui}"), &p);
    }
    let a = ui::mini(&parts, "good", "", 1.0);
    let b = ui::mini(&parts, "good", "", 2.0);
    assert!((b.w as f64 / a.w as f64 - 2.0).abs() < 0.2);
}

#[test]
fn the_menu_draws_and_keeps_its_size() {
    let mut model = FixedSettings::default();
    let mut menu = Menu::new();
    let mut d = MenuDrawer::default();
    let a = d.render(&mut menu, &model, 1.67);
    dump("menu_main", &a);
    menu.key(&mut model, Nav::Down);
    menu.key(&mut model, Nav::Down);
    menu.key(&mut model, Nav::Enter);
    let b = d.render(&mut menu, &model, 1.67);
    dump("menu_display", &b);
    assert_eq!((a.w, a.h), (b.w, b.h));
    assert_eq!(a.w, (780.0f64 * 1.67).round() as usize);
    assert!(ink(&a) > 500 && a != b);
}
