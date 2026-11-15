//! Picture-level motion analysis ahead of the macroblock loop.
//!
//! The motion searches in [`crate::encoder::me`] look only at the
//! source picture, one reference plane and the macroblock position —
//! never at the state the macroblock loop carries (predicted motion
//! vectors, neighbour modes, the entropy coder). Every macroblock's
//! search can therefore run before the loop, on as many threads as the
//! caller granted, and the loop finds each answer where it would have
//! computed it. The coded stream does not depend on the thread count.
//!
//! [`MotionPlan::install`] runs the searches for one (source,
//! reference) pair and registers the results for the calling thread;
//! the returned guard keeps both planes borrowed, so a result can
//! never be read back for samples that have changed, and removes the
//! results when it drops. [`lookup_whole`] / [`lookup_cell`] are
//! consulted by the search entry points and answer only when every
//! parameter of the request (planes, geometry, search range) is the
//! one the plan was built with.

use std::cell::RefCell;
use std::marker::PhantomData;

use crate::encoder::me::{MeResult, quarter_pel_8x8_search, quarter_pel_16x16_search};

/// One reference plane's searches for every macroblock of a picture.
struct Layer {
    src: (usize, usize),
    src_stride: usize,
    reference: (usize, usize),
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    width_mbs: usize,
    height_mbs: usize,
    range: (i32, i32),
    whole: Vec<MeResult>,
    /// Search range and results of the four 8x8 cells, when requested.
    cell_range: Option<(i32, i32)>,
    cells: Vec<[MeResult; 4]>,
}

/// The output rows of one macroblock row: its index, its 16x16
/// results and its 8x8 cell results (empty when cells are not planned).
type RowOut<'r> = (usize, &'r mut [MeResult], &'r mut [[MeResult; 4]]);

thread_local! {
    static LAYERS: RefCell<Vec<Layer>> = const { RefCell::new(Vec::new()) };
}

/// What to search for each macroblock.
#[derive(Debug, Clone, Copy)]
pub(crate) struct PlanSpec {
    /// Integer range of the 16x16 search.
    pub range: (i32, i32),
    /// Integer range of the per-8x8 searches (`None`: not searched).
    pub cell_range: Option<(i32, i32)>,
}

/// Guard of an installed plan; see the module documentation.
pub(crate) struct MotionPlan<'a> {
    depth: usize,
    _planes: PhantomData<&'a [u8]>,
    /// The guard belongs to the thread whose registry it restores.
    _not_send: PhantomData<*const ()>,
}

impl<'a> MotionPlan<'a> {
    /// Search every macroblock of the `width_mbs` x `height_mbs`
    /// picture `src_y` against `ref_y` on `threads` threads and make
    /// the results available to this thread's searches until the guard
    /// drops.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn install(
        src_y: &'a [u8],
        src_stride: usize,
        ref_y: &'a [u8],
        ref_stride: usize,
        ref_w: u32,
        ref_h: u32,
        width_mbs: usize,
        height_mbs: usize,
        spec: PlanSpec,
        threads: usize,
    ) -> Self {
        let zero = MeResult {
            mv_x: 0,
            mv_y: 0,
            sad: 0,
        };
        let n = width_mbs * height_mbs;
        let mut whole = vec![zero; n];
        let mut cells = vec![[zero; 4]; if spec.cell_range.is_some() { n } else { 0 }];
        let row_work = |mb_y: usize, whole_row: &mut [MeResult], cell_row: &mut [[MeResult; 4]]| {
            for (mb_x, w) in whole_row.iter_mut().enumerate() {
                *w = quarter_pel_16x16_search(
                    src_y,
                    src_stride,
                    ref_y,
                    ref_stride,
                    ref_w,
                    ref_h,
                    mb_x,
                    mb_y,
                    spec.range.0,
                    spec.range.1,
                );
            }
            if let Some((rx, ry)) = spec.cell_range {
                for (mb_x, c) in cell_row.iter_mut().enumerate() {
                    *c = std::array::from_fn(|i| {
                        quarter_pel_8x8_search(
                            src_y,
                            src_stride,
                            ref_y,
                            ref_stride,
                            ref_w,
                            ref_h,
                            mb_x,
                            mb_y,
                            i % 2,
                            i / 2,
                            rx,
                            ry,
                        )
                    });
                }
            }
        };
        if width_mbs > 0 {
            // Deal the macroblock rows out round-robin: neighbouring
            // rows cost about the same, so every thread gets an even
            // share whatever the picture content.
            let workers = threads.clamp(1, height_mbs.max(1));
            let mut cell_rows: Vec<&mut [[MeResult; 4]]> = if cells.is_empty() {
                (0..height_mbs).map(|_| Default::default()).collect()
            } else {
                cells.chunks_mut(width_mbs).collect()
            };
            let mut shares: Vec<Vec<RowOut<'_>>> = (0..workers).map(|_| Vec::new()).collect();
            for ((mb_y, w), c) in whole
                .chunks_mut(width_mbs)
                .enumerate()
                .zip(cell_rows.drain(..))
            {
                shares[mb_y % workers].push((mb_y, w, c));
            }
            let run = |share: Vec<RowOut<'_>>| {
                for (mb_y, w, c) in share {
                    row_work(mb_y, w, c);
                }
            };
            if workers == 1 {
                shares.into_iter().for_each(run);
            } else {
                std::thread::scope(|s| {
                    let mut it = shares.into_iter();
                    let mine = it.next();
                    for share in it {
                        s.spawn(|| run(share));
                    }
                    if let Some(share) = mine {
                        run(share);
                    }
                });
            }
        }
        let layer = Layer {
            src: (src_y.as_ptr() as usize, src_y.len()),
            src_stride,
            reference: (ref_y.as_ptr() as usize, ref_y.len()),
            ref_stride,
            ref_w,
            ref_h,
            width_mbs,
            height_mbs,
            range: spec.range,
            whole,
            cell_range: spec.cell_range,
            cells,
        };
        let depth = LAYERS.with(|l| {
            let mut l = l.borrow_mut();
            l.push(layer);
            l.len() - 1
        });
        Self {
            depth,
            _planes: PhantomData,
            _not_send: PhantomData,
        }
    }
}

/// Both reference lists' plans of a B picture: the 16x16 search at
/// range 16 and the 8x8 cells at range 4, as the B macroblock coders
/// request them.
pub(crate) fn install_b<'a>(
    src_y: &'a [u8],
    src_stride: usize,
    ref_l0: &'a crate::encoder::EncodedFrameRef<'a>,
    ref_l1: &'a crate::encoder::EncodedFrameRef<'a>,
    width_mbs: usize,
    height_mbs: usize,
    threads: usize,
) -> (MotionPlan<'a>, MotionPlan<'a>) {
    let spec = PlanSpec {
        range: (16, 16),
        cell_range: Some((4, 4)),
    };
    let one = |r: &'a crate::encoder::EncodedFrameRef<'a>| {
        MotionPlan::install(
            src_y,
            src_stride,
            r.recon_y,
            r.width as usize,
            r.width,
            r.height,
            width_mbs,
            height_mbs,
            spec,
            threads,
        )
    };
    let l0 = one(ref_l0);
    (l0, one(ref_l1))
}

impl Drop for MotionPlan<'_> {
    fn drop(&mut self) {
        LAYERS.with(|l| l.borrow_mut().truncate(self.depth));
    }
}

impl Layer {
    #[allow(clippy::too_many_arguments)]
    fn covers(
        &self,
        src_y: &[u8],
        src_stride: usize,
        ref_y: &[u8],
        ref_stride: usize,
        ref_w: u32,
        ref_h: u32,
        mb_x: usize,
        mb_y: usize,
    ) -> bool {
        self.src == (src_y.as_ptr() as usize, src_y.len())
            && self.src_stride == src_stride
            && self.reference == (ref_y.as_ptr() as usize, ref_y.len())
            && self.ref_stride == ref_stride
            && self.ref_w == ref_w
            && self.ref_h == ref_h
            && mb_x < self.width_mbs
            && mb_y < self.height_mbs
    }
}

/// The planned 16x16 search result for this exact request, if any.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lookup_whole(
    src_y: &[u8],
    src_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    range: (i32, i32),
) -> Option<MeResult> {
    LAYERS.with(|l| {
        l.borrow()
            .iter()
            .find(|p| {
                p.range == range
                    && p.covers(
                        src_y, src_stride, ref_y, ref_stride, ref_w, ref_h, mb_x, mb_y,
                    )
            })
            .map(|p| p.whole[mb_y * p.width_mbs + mb_x])
    })
}

/// The planned 8x8 search result of cell `(sub_x, sub_y)` for this
/// exact request, if any.
#[allow(clippy::too_many_arguments)]
pub(crate) fn lookup_cell(
    src_y: &[u8],
    src_stride: usize,
    ref_y: &[u8],
    ref_stride: usize,
    ref_w: u32,
    ref_h: u32,
    mb_x: usize,
    mb_y: usize,
    sub_x: usize,
    sub_y: usize,
    range: (i32, i32),
) -> Option<MeResult> {
    if sub_x > 1 || sub_y > 1 {
        return None;
    }
    LAYERS.with(|l| {
        l.borrow()
            .iter()
            .find(|p| {
                p.cell_range == Some(range)
                    && p.covers(
                        src_y, src_stride, ref_y, ref_stride, ref_w, ref_h, mb_x, mb_y,
                    )
            })
            .map(|p| p.cells[mb_y * p.width_mbs + mb_x][sub_y * 2 + sub_x])
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoder::me::{search_quarter_pel_8x8, search_quarter_pel_16x16};

    fn plane(w: usize, h: usize, seed: usize) -> Vec<u8> {
        (0..w * h)
            .map(|k| {
                let (x, y) = (k % w, k / w);
                ((x * 7 + y * 13 + seed * 29 + (x * y) % 31) % 256) as u8
            })
            .collect()
    }

    /// A plan answers every search exactly as the search itself does,
    /// for every thread count, and stops answering once dropped or
    /// when any parameter of the request differs.
    #[test]
    fn plan_answers_match_direct_searches() {
        let (w, h) = (64usize, 48usize);
        let (src, reference, other) = (plane(w, h, 1), plane(w, h, 2), plane(w, h, 3));
        let direct_whole: Vec<MeResult> = (0..12)
            .map(|a| {
                quarter_pel_16x16_search(
                    &src,
                    w,
                    &reference,
                    w,
                    w as u32,
                    h as u32,
                    a % 4,
                    a / 4,
                    8,
                    8,
                )
            })
            .collect();
        for threads in [1usize, 2, 3, 8] {
            let spec = PlanSpec {
                range: (8, 8),
                cell_range: Some((4, 4)),
            };
            let guard = MotionPlan::install(
                &src, w, &reference, w, w as u32, h as u32, 4, 3, spec, threads,
            );
            for (a, &want) in direct_whole.iter().enumerate() {
                let (mb_x, mb_y) = (a % 4, a / 4);
                let got = lookup_whole(
                    &src,
                    w,
                    &reference,
                    w,
                    w as u32,
                    h as u32,
                    mb_x,
                    mb_y,
                    (8, 8),
                )
                .expect("planned");
                assert_eq!(
                    (got.mv_x, got.mv_y, got.sad),
                    (want.mv_x, want.mv_y, want.sad)
                );
                // The public entry point returns the planned value.
                let via = search_quarter_pel_16x16(
                    &src, w, w as u32, h as u32, &reference, w, w as u32, h as u32, mb_x, mb_y, 8,
                    8,
                );
                assert_eq!(
                    (via.mv_x, via.mv_y, via.sad),
                    (want.mv_x, want.mv_y, want.sad)
                );
                for cell in 0..4usize {
                    let want = quarter_pel_8x8_search(
                        &src,
                        w,
                        &reference,
                        w,
                        w as u32,
                        h as u32,
                        mb_x,
                        mb_y,
                        cell % 2,
                        cell / 2,
                        4,
                        4,
                    );
                    let got = search_quarter_pel_8x8(
                        &src,
                        w,
                        w as u32,
                        h as u32,
                        &reference,
                        w,
                        w as u32,
                        h as u32,
                        mb_x,
                        mb_y,
                        cell % 2,
                        cell / 2,
                        4,
                        4,
                    );
                    assert_eq!(
                        (got.mv_x, got.mv_y, got.sad),
                        (want.mv_x, want.mv_y, want.sad)
                    );
                }
            }
            // Another range, another reference, a position outside.
            assert!(
                lookup_whole(&src, w, &reference, w, w as u32, h as u32, 0, 0, (16, 16)).is_none()
            );
            assert!(lookup_whole(&src, w, &other, w, w as u32, h as u32, 0, 0, (8, 8)).is_none());
            assert!(
                lookup_whole(&src, w, &reference, w, w as u32, h as u32, 4, 0, (8, 8)).is_none()
            );
            assert!(
                lookup_cell(
                    &src,
                    w,
                    &reference,
                    w,
                    w as u32,
                    h as u32,
                    0,
                    0,
                    0,
                    0,
                    (8, 8)
                )
                .is_none()
            );
            drop(guard);
            assert!(
                lookup_whole(&src, w, &reference, w, w as u32, h as u32, 0, 0, (8, 8)).is_none()
            );
        }
    }
}
