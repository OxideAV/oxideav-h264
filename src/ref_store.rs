//! Reference picture store for inter-prediction motion compensation.
//!
//! The reconstruction pipeline needs access to reference pictures
//! (previously decoded frames marked "used for reference" — §8.2.5).
//! This module provides the [`RefPicProvider`] trait the reconstruction
//! layer consumes and a simple concrete [`RefPicStore`] that holds
//! [`Picture`] values indexed by caller-supplied keys.
//!
//! Spec references follow ITU-T Rec. H.264 (08/2024):
//! - §8.2.4 reference picture list construction — this module is the
//!   bridge between per-slice RefPicList0 / RefPicList1 (produced by
//!   [`crate::ref_list`]) and the decoded-picture samples (a
//!   [`Picture`]) used by §8.4.2 inter prediction.
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024).

use crate::picture::Picture;

/// Trait the reconstruction layer consumes to fetch reference picture
/// samples for inter prediction (§8.4.2).
///
/// - `list` is `0` for RefPicList0 and `1` for RefPicList1.
/// - `idx` is the position in that list.
///
/// Returns `None` when the requested slot is out of bounds or carries
/// no picture — the caller surfaces this as
/// [`crate::reconstruct::ReconstructError::MissingRefPic`].
pub trait RefPicProvider {
    fn ref_pic(&self, list: u8, idx: u32) -> Option<&Picture>;

    /// §8.4.1.2.3 — POC of the picture at `(list, idx)`.
    /// Default implementation delegates to `ref_pic`. Callers can
    /// override for efficiency.
    fn ref_pic_poc(&self, list: u8, idx: u32) -> Option<i32> {
        self.ref_pic(list, idx).map(|p| p.pic_order_cnt)
    }

    /// §8.4.1.2.3 — POCs of the current slice's full RefPicList0.
    /// Used by the MapColToList0 derivation in temporal direct mode
    /// to locate the index in the current slice's RefPicList0 that
    /// matches a colocated picture's per-block L0 reference.
    ///
    /// Default returns an empty slice — call sites that do not need
    /// this derivation (I/P slice reconstruction) need not override.
    fn ref_list_0_pocs(&self) -> &[i32] {
        &[]
    }

    /// §8.4.1.2.3 — parallel to `ref_list_0_pocs`: whether entry `k`
    /// is a long-term reference in the current slice.
    fn ref_list_0_longterm(&self) -> &[bool] {
        &[]
    }

    /// §8.4.1.2.2 — parallel to `ref_list_0_longterm` but for
    /// RefPicList1. Used by B-slice spatial direct mode (colZeroFlag
    /// is suppressed when `RefPicList1[0]` is a long-term reference).
    ///
    /// Default returns an empty slice — call sites that do not need
    /// this derivation (I/P/SP/SI slice reconstruction, B-slice
    /// temporal direct) need not override.
    fn ref_list_1_longterm(&self) -> &[bool] {
        &[]
    }
}

/// A caller-supplied store mapping list indices to decoded pictures.
///
/// Usage:
///   1. `insert(key, picture)` for each decoded reference picture.
///   2. `set_list_0(keys)` / `set_list_1(keys)` with the per-slice
///      ref-picture-list key arrays (produced by [`crate::ref_list`]).
///   3. Pass `&store` into [`crate::reconstruct::reconstruct_slice`].
///
/// `pictures` is a sparse `Vec<Option<Picture>>` indexed by key, so
/// callers can reuse numeric keys across pictures.
pub struct RefPicStore {
    pictures: Vec<Option<Picture>>,
    ref_pic_list_0: Vec<u32>,
    ref_pic_list_1: Vec<u32>,
}

impl Default for RefPicStore {
    fn default() -> Self {
        Self::new()
    }
}

impl RefPicStore {
    pub fn new() -> Self {
        Self {
            pictures: Vec::new(),
            ref_pic_list_0: Vec::new(),
            ref_pic_list_1: Vec::new(),
        }
    }

    /// Insert a picture at `key`. Grows the storage vector as needed.
    pub fn insert(&mut self, key: u32, pic: Picture) {
        let k = key as usize;
        if k >= self.pictures.len() {
            self.pictures.resize_with(k + 1, || None);
        }
        self.pictures[k] = Some(pic);
    }

    /// Replace the RefPicList0 key array.
    pub fn set_list_0(&mut self, keys: Vec<u32>) {
        self.ref_pic_list_0 = keys;
    }

    /// Replace the RefPicList1 key array.
    pub fn set_list_1(&mut self, keys: Vec<u32>) {
        self.ref_pic_list_1 = keys;
    }

    /// Fetch a picture directly by key.
    pub fn get_by_key(&self, key: u32) -> Option<&Picture> {
        self.pictures.get(key as usize).and_then(|p| p.as_ref())
    }
}

impl RefPicProvider for RefPicStore {
    fn ref_pic(&self, list: u8, idx: u32) -> Option<&Picture> {
        let keys = match list {
            0 => &self.ref_pic_list_0,
            1 => &self.ref_pic_list_1,
            _ => return None,
        };
        let key = keys.get(idx as usize).copied()?;
        self.get_by_key(key)
    }
}

/// Unit provider used by call sites that only decode I-slices. Every
/// `ref_pic` query returns `None`.
pub struct NoRefs;

impl RefPicProvider for NoRefs {
    fn ref_pic(&self, _list: u8, _idx: u32) -> Option<&Picture> {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pic(w: u32, fill: i32) -> Picture {
        let mut p = Picture::new(w, w, 1, 8, 8);
        for v in p.luma.iter_mut() {
            *v = fill;
        }
        p
    }

    #[test]
    fn store_insert_and_get_by_key() {
        let mut s = RefPicStore::new();
        s.insert(0, pic(16, 10));
        s.insert(3, pic(16, 40));
        assert!(s.get_by_key(0).is_some());
        // Sparse slot at 1 and 2 is None.
        assert!(s.get_by_key(1).is_none());
        assert!(s.get_by_key(2).is_none());
        assert!(s.get_by_key(3).is_some());
        assert!(s.get_by_key(4).is_none());
    }

    #[test]
    fn store_list0_lookup() {
        let mut s = RefPicStore::new();
        s.insert(5, pic(16, 42));
        s.insert(7, pic(16, 99));
        s.set_list_0(vec![5, 7]);
        assert_eq!(s.ref_pic(0, 0).unwrap().luma[0], 42);
        assert_eq!(s.ref_pic(0, 1).unwrap().luma[0], 99);
        // Out-of-bounds index.
        assert!(s.ref_pic(0, 2).is_none());
        // Wrong list.
        assert!(s.ref_pic(1, 0).is_none());
    }

    #[test]
    fn store_list1_lookup_after_set() {
        let mut s = RefPicStore::new();
        s.insert(0, pic(16, 1));
        s.set_list_1(vec![0]);
        assert_eq!(s.ref_pic(1, 0).unwrap().luma[0], 1);
    }

    #[test]
    fn store_unknown_list_returns_none() {
        let s = RefPicStore::new();
        assert!(s.ref_pic(2, 0).is_none());
    }

    #[test]
    fn no_refs_always_none() {
        let nr = NoRefs;
        assert!(nr.ref_pic(0, 0).is_none());
        assert!(nr.ref_pic(1, 0).is_none());
        assert!(nr.ref_pic(5, 99).is_none());
    }
}
