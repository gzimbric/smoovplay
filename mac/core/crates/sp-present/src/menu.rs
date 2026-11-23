//! The in-video settings menu's navigation (framegen/menu.py Menu). The main page has the quick picks and a row for
//! each section, and each section has its own page. Up and Down select, Left and Right change a value, and Enter
//! opens a section, changes a value or restarts. Esc or Backspace goes back a page, and on the main page it closes
//! the menu. The drawing is in [`crate::ui`].

use std::collections::BTreeMap;

use crate::display::{MenuModel, MenuRow};

/// A row: a setting, a section to open, or "Restart now".
#[derive(Clone, Debug, PartialEq)]
pub enum Row {
    Setting(MenuRow),
    Section(String),
    Restart,
}

impl Row {
    fn key(&self) -> String {
        match self {
            Row::Setting(r) => r.key.clone(),
            Row::Section(s) => format!("section:{s}"),
            Row::Restart => "restart".into(),
        }
    }
}

/// A menu key.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Nav {
    Up,
    Down,
    Left,
    Right,
    Enter,
    Back,
}

/// What a key did that the window acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuResult {
    Close,
    Restart,
}

/// The menu's state. `page` stays when the menu closes, so the player comes back to where they were.
#[derive(Debug, Default)]
pub struct Menu {
    pub open: bool,
    pub page: Option<String>,
    pub index: usize,
    sel: Option<String>,
    at: usize,
    left: BTreeMap<String, String>,
}

impl Menu {
    pub fn new() -> Menu {
        Menu::default()
    }

    /// A page's rows. The main page has the quick picks, then the sections. Restart now comes last when something
    /// needs it.
    pub fn page_rows(&self, model: &dyn MenuModel, page: Option<&str>) -> Vec<Row> {
        let mut rows: Vec<Row> = model.rows(page).into_iter().map(Row::Setting).collect();
        if page.is_none() {
            rows.extend(model.sections().into_iter().map(Row::Section));
        }
        if !model.restart_reasons().is_empty() {
            rows.push(Row::Restart);
        }
        rows
    }

    /// The rows of the page shown (the main page if its section has gone).
    pub fn rows(&mut self, model: &dyn MenuModel) -> Vec<Row> {
        if let Some(p) = &self.page
            && !model.sections().contains(p) {
                self.page = None;
            }
        let page = self.page.clone();
        self.page_rows(model, page.as_deref())
    }

    /// Keeps the highlight on the same row when rows appear or go.
    pub fn follow(&mut self, rows: &[Row]) {
        if rows.is_empty() {
            self.index = 0;
            return;
        }
        let keys: Vec<String> = rows.iter().map(Row::key).collect();
        if self.index == self.at
            && let Some(i) = self.sel.as_ref().and_then(|s| keys.iter().position(|k| k == s)) {
                self.index = i;
            }
        self.index = self.index.min(rows.len() - 1);
        self.sel = Some(keys[self.index].clone());
        self.at = self.index;
    }

    fn show(&mut self, model: &dyn MenuModel, page: Option<String>) {
        if let (Some(p), Some(s)) = (&self.page, &self.sel) {
            self.left.insert(p.clone(), s.clone());
        }
        let want = match &page {
            None => self.page.as_ref().map(|p| format!("section:{p}")),
            Some(p) => self.left.get(p).cloned(),
        };
        self.page = page;
        let keys: Vec<String> = self.rows(model).iter().map(Row::key).collect();
        self.index = want.and_then(|w| keys.iter().position(|k| *k == w)).unwrap_or(0);
        self.sel = keys.get(self.index).cloned();
        self.at = self.index;
    }

    /// A key while the menu is open.
    pub fn key(&mut self, model: &mut dyn MenuModel, nav: Nav) -> Option<MenuResult> {
        let rows = self.rows(model);
        self.follow(&rows);
        if nav == Nav::Back {
            if self.page.is_none() {
                return Some(MenuResult::Close);
            }
            self.show(model, None);
            return None;
        }
        if rows.is_empty() {
            return None;
        }
        if matches!(nav, Nav::Up | Nav::Down) {
            let n = rows.len() as i64;
            self.index = (self.index as i64 + if nav == Nav::Up { -1 } else { 1 }).rem_euclid(n) as usize;
            self.sel = Some(rows[self.index].key());
            self.at = self.index;
            return None;
        }
        match &rows[self.index] {
            Row::Restart => (nav == Nav::Enter).then_some(MenuResult::Restart),
            Row::Section(s) => {
                if matches!(nav, Nav::Enter | Nav::Right) {
                    self.show(model, Some(s.clone()));
                }
                None
            }
            Row::Setting(r) => {
                if r.disabled.is_none() {
                    model.step(&r.key, if nav == Nav::Left { -1 } else { 1 });
                }
                None
            }
        }
    }

    /// "In here: A, B and C." for a section row's help.
    pub fn section_help(&self, model: &dyn MenuModel, section: &str) -> String {
        let labels: Vec<String> = model.rows(Some(section)).into_iter().map(|r| r.label).collect();
        match labels.len() {
            0 => String::new(),
            1 => format!("In here: {}.", labels[0]),
            n => format!("In here: {} and {}.", labels[..n - 1].join(", "), labels[n - 1]),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::display::FixedSettings;

    #[test]
    fn pages_and_keys() {
        let mut m = FixedSettings::default();
        let mut menu = Menu::new();
        let rows = menu.rows(&m);
        assert_eq!(rows.len(), 3); // two quick picks, one section
        assert!(matches!(rows[2], Row::Section(_)));
        menu.key(&mut m, Nav::Down);
        menu.key(&mut m, Nav::Right); // the upscaler
        assert_eq!(m.display.scaler, crate::display::Scaler::Bilinear);
        menu.key(&mut m, Nav::Down);
        menu.key(&mut m, Nav::Enter); // opens Display
        assert_eq!(menu.page.as_deref(), Some("Display"));
        menu.key(&mut m, Nav::Enter); // fullscreen on
        assert!(m.display.fullscreen);
        menu.key(&mut m, Nav::Back);
        assert_eq!((menu.page.as_deref(), menu.index), (None, 2)); // back on the section's row
        assert_eq!(menu.key(&mut m, Nav::Back), Some(MenuResult::Close));
        menu.key(&mut m, Nav::Up);
        menu.key(&mut m, Nav::Up);
        menu.key(&mut m, Nav::Up);
        assert_eq!(menu.index, 2); // wraps
    }
}
