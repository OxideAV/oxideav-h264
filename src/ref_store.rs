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
use std::collections::HashMap;

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

    /// Round-416 PAFF — §8.4.1.4 Table 8-10: parity of the reference
    /// FIELD at `(list, idx)` when the current slice is a coded field
    /// picture (`0` = top field, `1` = bottom field). `None` for frame
    /// references or non-field slices; the default suits every
    /// provider that never serves field-picture slices.
    fn ref_field_parity(&self, _list: u8, _idx: u32) -> Option<u8> {
        None
    }

    /// §8.4.1.2.3 MapColToList0 — picture-identity view of the current
    /// slice's RefPicList0: per-entry DPB storage keys, field parities
    /// (`None` for frame/pair units) and containing frame-level UNIT
    /// keys. Defaults return empty slices — providers that never serve
    /// temporal-direct B slices need not override.
    fn ref_list_0_keys(&self) -> &[u32] {
        &[]
    }
    fn ref_list_0_unit_keys(&self) -> &[u32] {
        &[]
    }
    fn ref_list_0_parities(&self) -> &[Option<u8>] {
        &[]
    }
    /// §8.4.1.2.1 — identity of the picture at `(list, idx)`: the
    /// entry's storage key, parity (`None` = frame/pair unit) and its
    /// containing frame-level unit key.
    fn ref_entry_identity(&self, _list: u8, _idx: u32) -> Option<(u32, Option<u8>, u32)> {
        None
    }
    /// §8.4.1.2.3 — TopFieldOrderCnt / BottomFieldOrderCnt of the
    /// frame-level unit at `(list, idx)` of a FRAME slice's list (used
    /// for the per-field tb/td distances of MBAFF field macroblocks
    /// and the Table 8-6 / eq. 8-182 topAbsDiffPOC comparison).
    fn ref_entry_unit_focs(&self, _list: u8, _idx: u32) -> Option<(i32, i32)> {
        None
    }
    /// §8.4.1.2.1 Table 8-6 — when the entry at `(list, idx)` of a
    /// FRAME slice's list is a complementary field PAIR, the stored
    /// coded-field picture of the given parity (`false` = top). `None`
    /// for genuine frame references.
    fn ref_pair_field(&self, _list: u8, _idx: u32, _bottom: bool) -> Option<&Picture> {
        None
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
/// `pictures` maps caller-supplied numeric keys to stored pictures.
/// Keys may be minted monotonically without bounding memory: callers
/// prune dropped references via [`RefPicStore::retain_keys`], so the
/// live set stays at DPB size (§8.2.5 keeps at most
/// `max_num_ref_frames` + in-flight pictures marked "used for
/// reference"). A sparse `Vec<Option<Picture>>` indexed by key was
/// used before round 430; with monotonic keys that representation
/// retained EVERY picture ever decoded for the whole session
/// (unbounded growth — the 2026-07-25 scheduled-fuzz OOM triage
/// surfaced it).
pub struct RefPicStore {
    pictures: HashMap<u32, Picture>,
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
            pictures: HashMap::new(),
            ref_pic_list_0: Vec::new(),
            ref_pic_list_1: Vec::new(),
        }
    }

    /// Insert (or replace) the picture stored at `key`.
    pub fn insert(&mut self, key: u32, pic: Picture) {
        self.pictures.insert(key, pic);
    }

    /// Drop every stored picture whose key is NOT in `live`.
    ///
    /// §8.2.5 — once a reference picture is marked "unused for
    /// reference" (sliding window or MMCO) no later slice can name it
    /// in a reference picture list, so its samples are dead weight.
    /// The decoder calls this after each marking pass with the keys
    /// still present in its DPB so store memory stays bounded by the
    /// DPB size instead of growing with every reference picture of
    /// the session. `live` is at most DPB-sized, so the linear
    /// `contains` scan is cheap.
    pub fn retain_keys(&mut self, live: &[u32]) {
        self.pictures.retain(|k, _| live.contains(k));
    }

    /// Number of pictures currently stored. Exposed for the decoder's
    /// bounded-memory accounting (and its regression tests).
    pub fn stored_count(&self) -> usize {
        self.pictures.len()
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
        self.pictures.get(&key)
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
        // Keys never inserted resolve to None.
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

    /// Round 430 — `retain_keys` drops every picture not named live so
    /// store memory stays bounded by the DPB instead of accumulating
    /// one picture per reference frame for the whole session (the
    /// 2026-07-25 scheduled-fuzz OOM triage found the old sparse-Vec
    /// store never released anything).
    #[test]
    fn store_retain_keys_prunes_dead_pictures() {
        let mut s = RefPicStore::new();
        for k in 0..10 {
            s.insert(k, pic(16, k as i32));
        }
        assert_eq!(s.stored_count(), 10);
        s.retain_keys(&[3, 7]);
        assert_eq!(s.stored_count(), 2);
        assert!(s.get_by_key(3).is_some());
        assert!(s.get_by_key(7).is_some());
        assert!(s.get_by_key(0).is_none());
        assert!(s.get_by_key(9).is_none());
        // Empty live set clears the store.
        s.retain_keys(&[]);
        assert_eq!(s.stored_count(), 0);
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
