//! Frame-level threading support.
//!
//! With a worker budget above one the decoder overlaps the decoding of
//! several pictures (§8 decoding process, applied per picture in
//! decoding order):
//!
//! * the §7.3.4 slice data of every slice is parsed (entropy decoded)
//!   on the worker pool as soon as its NAL unit arrives — the parse
//!   depends only on the slice header and the active parameter sets;
//! * the decoder thread runs everything that later pictures' decoding
//!   depends on in decoding order — §8.2.1 POC, §8.2.4 reference lists,
//!   the §8.4.1 motion derivation of every macroblock, §8.2.5 marking,
//!   §C.4 output ordering;
//! * the §8.4.2 / §8.5 sample reconstruction and §8.7 deblocking of a
//!   picture then run as one job on the pool, macroblock row by row.
//!   A finished row band is published ([`SharedPlanes`]) for the
//!   pictures that reference it: a row of a later picture waits only
//!   for the reference rows its motion vectors actually reach.
//!
//! Every picture is reconstructed by the same per-macroblock processes
//! as the single-threaded decoder, so the output is identical for every
//! budget.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::{Arc, Condvar, Mutex, OnceLock};

use oxideav_core::VideoFrame;

use crate::mc::PlaneSrc;
use crate::picture::MotionGrids;
use crate::sample_plane::Samples;

/// The sample planes of a picture being reconstructed on the worker
/// pool, published one macroblock row (16 luma rows, `MbHeightC`
/// chroma rows) at a time, plus its finished output frame.
#[doc(hidden)] // internal — frame-threading plumbing
pub struct SharedPlanes {
    width: usize,
    height: usize,
    chroma_width: usize,
    chroma_height: usize,
    /// Chroma rows per macroblock row (0 for monochrome).
    chroma_band_rows: usize,
    luma: Vec<OnceLock<Samples>>,
    cb: Vec<OnceLock<Samples>>,
    cr: Vec<OnceLock<Samples>>,
    /// Number of bands published, or `usize::MAX` once the job failed.
    progress: Mutex<usize>,
    cond: Condvar,
    /// The §C.4 output frame, set when the whole picture is done.
    output: Mutex<Option<VideoFrame>>,
    done: OnceLock<bool>,
    /// The picture's §8.4.1 motion, set once the job derived it (the
    /// co-located motion of later B slices).
    motion: OnceLock<Arc<MotionGrids>>,
}

impl SharedPlanes {
    pub(crate) fn new(
        width: usize,
        height: usize,
        chroma_width: usize,
        chroma_height: usize,
        chroma_band_rows: usize,
    ) -> Self {
        let bands = height.div_ceil(16);
        let mk = || (0..bands).map(|_| OnceLock::new()).collect::<Vec<_>>();
        Self {
            width,
            height,
            chroma_width,
            chroma_height,
            chroma_band_rows,
            luma: mk(),
            cb: mk(),
            cr: mk(),
            progress: Mutex::new(0),
            cond: Condvar::new(),
            output: Mutex::new(None),
            done: OnceLock::new(),
            motion: OnceLock::new(),
        }
    }

    /// Publish the picture's motion.
    pub(crate) fn set_motion(&self, m: MotionGrids) {
        let _ = self.motion.set(Arc::new(m));
        let _g = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        self.cond.notify_all();
    }

    /// Block until the picture's motion is published; `None` when the
    /// job producing it failed.
    pub(crate) fn wait_motion(&self) -> Option<Arc<MotionGrids>> {
        let mut g = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(m) = self.motion.get() {
                return Some(m.clone());
            }
            if *g == usize::MAX || self.done.get().is_some() {
                return self.motion.get().cloned();
            }
            g = self.cond.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Number of macroblock-row bands.
    pub(crate) fn bands(&self) -> usize {
        self.luma.len()
    }

    /// The luma plane (every band read must be published).
    pub(crate) fn luma_src(&self) -> PlaneSrc<'_> {
        PlaneSrc::Banded {
            bands: &self.luma,
            band_rows: 16,
            width: self.width,
            height: self.height,
        }
    }

    /// Chroma plane `plane` (0 = Cb, 1 = Cr).
    pub(crate) fn chroma_src(&self, plane: u8) -> PlaneSrc<'_> {
        PlaneSrc::Banded {
            bands: if plane == 0 { &self.cb } else { &self.cr },
            band_rows: self.chroma_band_rows.max(1),
            width: self.chroma_width,
            height: self.chroma_height,
        }
    }

    /// Publish band `b` from the picture's planes (`luma` /
    /// `cb` / `cr` are whole row-major planes).
    pub(crate) fn publish(&self, b: usize, luma: &Samples, cb: &Samples, cr: &Samples) {
        let w = self.width;
        let l0 = (16 * b).min(self.height);
        let l1 = (16 * b + 16).min(self.height);
        let _ = self.luma[b].set(luma.slice(l0 * w..l1 * w).to_plane());
        if self.chroma_band_rows > 0 {
            let cw = self.chroma_width;
            let c0 = (self.chroma_band_rows * b).min(self.chroma_height);
            let c1 = (self.chroma_band_rows * (b + 1)).min(self.chroma_height);
            let _ = self.cb[b].set(cb.slice(c0 * cw..c1 * cw).to_plane());
            let _ = self.cr[b].set(cr.slice(c0 * cw..c1 * cw).to_plane());
        }
        let mut p = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        if *p != usize::MAX && *p < b + 1 {
            *p = b + 1;
        }
        drop(p);
        self.cond.notify_all();
    }

    /// Block until bands `0..=b` are published. `false` when the job
    /// producing them failed.
    pub(crate) fn wait_band(&self, b: usize) -> bool {
        let b = b.min(self.bands().saturating_sub(1));
        let mut p = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if *p == usize::MAX {
                return false;
            }
            if *p > b {
                return true;
            }
            p = self.cond.wait(p).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Mark the job failed (a worker panicked): wakes every waiter.
    pub(crate) fn abort(&self) {
        *self.progress.lock().unwrap_or_else(|e| e.into_inner()) = usize::MAX;
        let _ = self.done.set(false);
        self.cond.notify_all();
    }

    /// Store the finished output frame and mark the picture done.
    pub(crate) fn finish(&self, frame: Option<VideoFrame>) {
        *self.output.lock().unwrap_or_else(|e| e.into_inner()) = frame;
        let _ = self.done.set(true);
        let _g = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        self.cond.notify_all();
    }

    /// `Some(ok)` once the job ended.
    pub(crate) fn is_done(&self) -> Option<bool> {
        self.done.get().copied()
    }

    /// Block until the job ended; `true` when it succeeded.
    pub(crate) fn wait_done(&self) -> bool {
        let mut g = self.progress.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(&ok) = self.done.get() {
                return ok;
            }
            g = self.cond.wait(g).unwrap_or_else(|e| e.into_inner());
        }
    }

    /// Take the output frame (after [`Self::wait_done`]).
    pub(crate) fn take_output(&self) -> Option<VideoFrame> {
        self.output.lock().unwrap_or_else(|e| e.into_inner()).take()
    }

    /// Assemble the whole picture's planes (after [`Self::wait_done`]).
    /// `None` for a plane none of whose bands was published.
    pub(crate) fn assemble(&self) -> (Option<Samples>, Option<Samples>, Option<Samples>) {
        let cat = |v: &[OnceLock<Samples>]| {
            let mut out: Option<Samples> = None;
            for b in v {
                if let Some(b) = b.get() {
                    out.get_or_insert_with(|| b.empty_like())
                        .extend_from(b.as_ref());
                }
            }
            out
        };
        (cat(&self.luma), cat(&self.cb), cat(&self.cr))
    }
}

type Task = Box<dyn FnOnce() + Send>;

struct PoolQueue {
    /// Short tasks the decoder thread waits on (slice-data parsing).
    high: VecDeque<Task>,
    /// Picture reconstruction jobs.
    low: VecDeque<Task>,
    shutdown: bool,
}

struct PoolShared {
    queue: Mutex<PoolQueue>,
    cond: Condvar,
}

/// A fixed set of worker threads running queued tasks, high-priority
/// tasks first, each kind in submission order. A task only ever waits
/// on work submitted before it (or runs it inline), so the pool cannot
/// deadlock.
pub(crate) struct Pool {
    shared: Arc<PoolShared>,
    workers: Vec<std::thread::JoinHandle<()>>,
}

// The pool's state is its task queue (behind a mutex) and the worker
// handles; nothing a caller observes can be left half-updated by a panic
// unwinding through the decoder, which kept the decoder unwind-safe
// before it owned a pool.
impl std::panic::UnwindSafe for Pool {}
impl std::panic::RefUnwindSafe for Pool {}

impl Pool {
    pub(crate) fn new(threads: usize) -> Self {
        let shared = Arc::new(PoolShared {
            queue: Mutex::new(PoolQueue {
                high: VecDeque::new(),
                low: VecDeque::new(),
                shutdown: false,
            }),
            cond: Condvar::new(),
        });
        let workers = (0..threads.max(1))
            .map(|i| {
                let shared = shared.clone();
                std::thread::Builder::new()
                    .name(format!("h264-dec-{i}"))
                    .spawn(move || {
                        loop {
                            let task = {
                                let mut q = shared.queue.lock().unwrap_or_else(|e| e.into_inner());
                                loop {
                                    if let Some(t) = q.high.pop_front() {
                                        break Some(t);
                                    }
                                    if let Some(t) = q.low.pop_front() {
                                        break Some(t);
                                    }
                                    if q.shutdown {
                                        break None;
                                    }
                                    q = shared.cond.wait(q).unwrap_or_else(|e| e.into_inner());
                                }
                            };
                            match task {
                                Some(t) => t(),
                                None => return,
                            }
                        }
                    })
                    .expect("spawn h264 decode worker")
            })
            .collect();
        Self { shared, workers }
    }

    pub(crate) fn threads(&self) -> usize {
        self.workers.len()
    }

    fn push(&self, task: Task, high: bool) {
        let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
        if high {
            q.high.push_back(task);
        } else {
            q.low.push_back(task);
        }
        drop(q);
        self.shared.cond.notify_one();
    }

    /// Run one queued task on the calling thread (high-priority tasks
    /// first). `false` when the queue is empty. Every task only waits
    /// on work queued before it, which is running or done once it is
    /// dequeued, so a waiting caller can help without deadlocking.
    pub(crate) fn help_one(&self) -> bool {
        let task = {
            let mut q = self.shared.queue.lock().unwrap_or_else(|e| e.into_inner());
            q.high.pop_front().or_else(|| q.low.pop_front())
        };
        match task {
            Some(t) => {
                t();
                true
            }
            None => false,
        }
    }

    /// Queue a picture job.
    pub(crate) fn spawn_low(&self, f: impl FnOnce() + Send + 'static) {
        self.push(Box::new(f), false);
    }

    /// Queue a short task.
    pub(crate) fn spawn_high(&self, f: impl FnOnce() + Send + 'static) {
        self.push(Box::new(f), true);
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shared
            .queue
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .shutdown = true;
        self.shared.cond.notify_all();
        for w in self.workers.drain(..) {
            let _ = w.join();
        }
    }
}

enum Lazy<I, O> {
    Pending(I),
    Running,
    /// `None` when the computation panicked.
    Done(Option<O>),
    Taken,
}

/// A computation that runs once, on whichever thread gets to it first:
/// a pool worker, or the consumer itself if no worker started it yet.
pub(crate) struct Deferred<I, O> {
    state: Mutex<Lazy<I, O>>,
    cond: Condvar,
    run: fn(I) -> O,
}

impl<I: Send + 'static, O: Send + 'static> Deferred<I, O> {
    /// Queue `run(input)` on `pool` as a high-priority task.
    pub(crate) fn spawn(pool: &Pool, input: I, run: fn(I) -> O) -> Arc<Self> {
        let d = Arc::new(Self {
            state: Mutex::new(Lazy::Pending(input)),
            cond: Condvar::new(),
            run,
        });
        let task = d.clone();
        pool.spawn_high(move || task.try_run());
        d
    }

    fn try_run(&self) {
        let input = {
            let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
            match std::mem::replace(&mut *s, Lazy::Running) {
                Lazy::Pending(i) => i,
                other => {
                    *s = other;
                    return;
                }
            }
        };
        let run = self.run;
        let out = catch_unwind(AssertUnwindSafe(move || run(input))).ok();
        *self.state.lock().unwrap_or_else(|e| e.into_inner()) = Lazy::Done(out);
        self.cond.notify_all();
    }

    /// The result: computed inline when no worker started it, else
    /// waited for; `None` when it panicked. Must be called at most once.
    pub(crate) fn take(&self) -> Option<O> {
        self.try_run();
        let mut s = self.state.lock().unwrap_or_else(|e| e.into_inner());
        loop {
            match std::mem::replace(&mut *s, Lazy::Taken) {
                Lazy::Done(o) => return o,
                Lazy::Running => {
                    *s = Lazy::Running;
                    s = self.cond.wait(s).unwrap_or_else(|e| e.into_inner());
                }
                _ => panic!("deferred result taken twice"),
            }
        }
    }
}

/// Run `f` as the job producing `planes`: a panic marks the planes
/// failed (waking their waiters) instead of leaving them hanging.
pub(crate) fn run_job(planes: &SharedPlanes, f: impl FnOnce()) {
    if catch_unwind(AssertUnwindSafe(f)).is_err() {
        planes.abort();
    }
}

// ---------------------------------------------------------------------
// The sample stage of a picture
// ---------------------------------------------------------------------

use crate::mb_grid::MbGrid;
use crate::picture::Picture;
use crate::pps::Pps;
use crate::reconstruct::{self, FtSlicePlan};
use crate::ref_store::RefPicProvider;
use crate::slice_data::SliceData;
use crate::slice_header::SliceHeader;
use crate::sps::Sps;

/// One reference-list entry for the sample stage: the picture (its
/// metadata, and its samples unless `planes` is set) and, for a
/// picture still being reconstructed, its published planes.
#[derive(Clone)]
pub(crate) struct FtRef {
    pub(crate) pic: Arc<Picture>,
    pub(crate) planes: Option<Arc<SharedPlanes>>,
}

/// The reference lists of one slice, owned, for the sample stage on a
/// pool worker. Mirrors what the decoder thread's provider answers for
/// the queries §8.4.2 makes.
/// The (top, bottom) stored fields of a complementary field pair.
pub(crate) type FieldPair = (Arc<Picture>, Arc<Picture>);

#[derive(Default)]
pub(crate) struct FtProvider {
    pub(crate) refs: [Vec<Option<FtRef>>; 2],
    pub(crate) parities: [Vec<Option<u8>>; 2],
    pub(crate) focs: [Vec<(i32, i32)>; 2],
    /// §8.4.1.2 metadata of the lists (as the decoder thread's provider
    /// answers it): list-0 order counts, long-term flags, DPB keys and
    /// frame-unit keys, and the two fields of a complementary field
    /// pair entry.
    pub(crate) pocs_0: Vec<i32>,
    pub(crate) longterm: [Vec<bool>; 2],
    pub(crate) keys: [Vec<u32>; 2],
    pub(crate) unit_keys: [Vec<u32>; 2],
    pub(crate) pair_fields: [Vec<Option<FieldPair>>; 2],
}

impl FtProvider {
    /// Give `RefPicList1[ 0 ]` — the co-located picture of §8.4.1.2 —
    /// its motion when it is a picture of the pool (waiting for its job
    /// to derive it). `false` when that job failed.
    pub(crate) fn resolve_colocated_motion(&mut self) -> bool {
        let Some(Some(r)) = self.refs[1].first_mut() else {
            return true;
        };
        let Some(planes) = r.planes.as_ref() else {
            return true;
        };
        match planes.wait_motion() {
            Some(m) => {
                let mut pic = (*r.pic).clone();
                pic.set_motion(&m);
                r.pic = Arc::new(pic);
                true
            }
            None => false,
        }
    }
}

impl RefPicProvider for FtProvider {
    fn ref_pic(&self, list: u8, idx: u32) -> Option<&Picture> {
        let l = self.refs.get(list as usize)?;
        l.get(idx as usize)?.as_ref().map(|r| &*r.pic)
    }

    fn ref_field_parity(&self, list: u8, idx: u32) -> Option<u8> {
        self.parities
            .get(list as usize)?
            .get(idx as usize)
            .copied()
            .flatten()
    }

    fn ref_entry_unit_focs(&self, list: u8, idx: u32) -> Option<(i32, i32)> {
        self.focs.get(list as usize)?.get(idx as usize).copied()
    }

    fn ref_planes(&self, list: u8, idx: u32) -> Option<&SharedPlanes> {
        let l = self.refs.get(list as usize)?;
        l.get(idx as usize)?.as_ref()?.planes.as_deref()
    }

    fn ref_list_0_pocs(&self) -> &[i32] {
        &self.pocs_0
    }

    fn ref_list_0_longterm(&self) -> &[bool] {
        &self.longterm[0]
    }

    fn ref_list_1_longterm(&self) -> &[bool] {
        &self.longterm[1]
    }

    fn ref_list_0_keys(&self) -> &[u32] {
        &self.keys[0]
    }

    fn ref_list_0_unit_keys(&self) -> &[u32] {
        &self.unit_keys[0]
    }

    fn ref_list_0_parities(&self) -> &[Option<u8>] {
        &self.parities[0]
    }

    fn ref_entry_identity(&self, list: u8, idx: u32) -> Option<(u32, Option<u8>, u32)> {
        let l = list as usize;
        let key = *self.keys.get(l)?.get(idx as usize)?;
        let parity = self.parities[l].get(idx as usize).copied().flatten();
        let unit = self.unit_keys[l].get(idx as usize).copied().unwrap_or(key);
        Some((key, parity, unit))
    }

    fn ref_pair_field(&self, list: u8, idx: u32, bottom: bool) -> Option<&Picture> {
        let (top, bot) = self
            .pair_fields
            .get(list as usize)?
            .get(idx as usize)?
            .as_ref()?;
        Some(if bottom { bot } else { top })
    }
}

/// One slice of a frame-threaded picture: its parsed data, the
/// parameter sets and reference lists it decodes with, and the plan
/// [`reconstruct::ft_motion_pass`] made.
pub(crate) struct FtSliceWork {
    pub(crate) sd: SliceData,
    pub(crate) header: SliceHeader,
    pub(crate) sps: Sps,
    pub(crate) pps: Pps,
    pub(crate) refs: FtProvider,
    pub(crate) plan: FtSlicePlan,
}

/// The §8.7 deblocking parameters of a picture.
pub(crate) struct FtDeblock<'a> {
    pub(crate) alpha_off: i32,
    pub(crate) beta_off: i32,
    pub(crate) sps: &'a Sps,
    pub(crate) pps: &'a Pps,
    pub(crate) mb_field_flags: &'a [bool],
}

/// Reconstruct the samples of a frame-threaded picture into `pic`:
/// macroblock row by macroblock row, the inter macroblocks (each once
/// the reference rows it reads are published), then the intra
/// macroblocks in decoding order; with `deblock`, row `r − 1` is
/// filtered once row `r` is reconstructed. With `publish`, every band
/// is published as soon as it is final. `false` when a reference
/// picture's job failed.
pub(crate) fn run_pixels(
    pic: &mut Picture,
    grid: &mut MbGrid,
    works: &[FtSliceWork],
    deblock: Option<FtDeblock<'_>>,
    publish: Option<&SharedPlanes>,
) -> bool {
    let mb_w = grid.width_in_mbs as usize;
    let mb_h = grid.height_in_mbs as usize;
    let mut rows_inter: Vec<Vec<(usize, usize)>> = vec![Vec::new(); mb_h];
    let mut rows_intra: Vec<Vec<(usize, usize)>> = vec![Vec::new(); mb_h];
    for (si, w) in works.iter().enumerate() {
        for (ji, j) in w.plan.inter.iter().enumerate() {
            if let Some(r) = rows_inter.get_mut(j.mb_y() as usize) {
                r.push((si, ji));
            }
        }
        for (ii, &(_, addr, _)) in w.plan.intra.iter().enumerate() {
            if let Some(r) = rows_intra.get_mut(addr as usize / mb_w.max(1)) {
                r.push((si, ii));
            }
        }
    }
    let cat = pic.chroma_array_type;
    let height = pic.height_in_samples;
    let chroma_band_rows: u32 = match cat {
        1 => 8,
        2 => 16,
        _ => 0,
    };
    let mut scratch = Picture::new(16, 16, cat, pic.bit_depth_luma, pic.bit_depth_chroma);
    let mut published = 0usize;
    let publish_upto = |pic: &Picture, published: &mut usize, end: usize| {
        if let Some(p) = publish {
            while *published < end.min(mb_h) {
                p.publish(*published, &pic.luma, &pic.cb, &pic.cr);
                *published += 1;
            }
        }
    };
    let mut needs: Vec<(*const SharedPlanes, usize, &SharedPlanes)> = Vec::new();
    for y in 0..mb_h {
        // Wait for the reference rows this row's motion reaches.
        needs.clear();
        for &(si, ji) in &rows_inter[y] {
            let w = &works[si];
            w.plan.inter[ji].ref_bands(cat, height, chroma_band_rows, |list, ref_idx, band| {
                if let Some(p) = w.refs.ref_planes(list, ref_idx as u32) {
                    let key = p as *const SharedPlanes;
                    match needs.iter_mut().find(|n| n.0 == key) {
                        Some(n) => n.1 = n.1.max(band),
                        None => needs.push((key, band, p)),
                    }
                }
            });
        }
        for &(_, band, p) in &needs {
            if !p.wait_band(band) {
                return false;
            }
        }
        for &(si, ji) in &rows_inter[y] {
            let w = &works[si];
            if let Err(e) = reconstruct::ft_inter_mb(
                &w.sd,
                &w.plan.inter[ji],
                &w.header,
                &w.sps,
                &w.pps,
                &w.refs,
                &mut scratch,
                pic,
            ) {
                eprintln!("h264 frame thread: inter macroblock skipped: {e}");
            }
        }
        for &(si, ii) in &rows_intra[y] {
            let w = &works[si];
            if let Err(e) = reconstruct::ft_intra_mb(
                &w.sd,
                w.plan.intra[ii],
                &w.header,
                &w.sps,
                &w.pps,
                pic,
                grid,
            ) {
                eprintln!("h264 frame thread: intra macroblock skipped: {e}");
            }
        }
        match &deblock {
            Some(d) => {
                if y >= 1 {
                    reconstruct::ft_deblock_row(
                        pic,
                        grid,
                        y as u32 - 1,
                        d.alpha_off,
                        d.beta_off,
                        d.sps,
                        d.pps,
                        d.mb_field_flags,
                    );
                    // Row y − 1's top edges were the last to touch row y − 2.
                    publish_upto(pic, &mut published, y - 1);
                }
            }
            None => publish_upto(pic, &mut published, y + 1),
        }
    }
    if let Some(d) = &deblock
        && mb_h > 0
    {
        reconstruct::ft_deblock_row(
            pic,
            grid,
            mb_h as u32 - 1,
            d.alpha_off,
            d.beta_off,
            d.sps,
            d.pps,
            d.mb_field_flags,
        );
    }
    publish_upto(pic, &mut published, mb_h);
    true
}
