//! Narrow sample storage for decoded pictures.
//!
//! §7.4.2.1.1 bounds every decoded sample by `( 1 << BitDepth ) − 1`
//! with `BitDepthY` / `BitDepthC` in 8..=14, so a plane of an 8-bit
//! component fits `u8` and every deeper one fits `u16`. The
//! arithmetic of the decoding process (§8.3 / §8.4 / §8.5 / §8.7) runs
//! in `i32` on small working blocks; the planes only store the clipped
//! results (eq. 5-7 `Clip1Y` / `Clip1C`), a quarter / half the memory
//! traffic of `i32` storage.
//!
//! [`Samples`] picks the width from the component's bit depth once, at
//! allocation; the hot kernels dispatch on it once per block and run
//! monomorphised over [`Sample`].
//!
//! Clean-room: derived only from ITU-T Rec. H.264 (08/2024).

/// A stored sample type: `u8` for 8-bit components, `u16` above.
pub trait Sample: Copy + Default + Send + Sync + PartialEq + std::fmt::Debug + 'static {
    /// The sample as a signed working value.
    fn get(self) -> i32;
    /// Store a working value. Decoded samples are always in
    /// `0..=( 1 << BitDepth ) − 1` (Clip1); out-of-range values (only
    /// ever fabricated by malformed input paths) saturate.
    fn put(v: i32) -> Self;
}

impl Sample for u8 {
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn put(v: i32) -> Self {
        v.clamp(0, u8::MAX as i32) as u8
    }
}

impl Sample for u16 {
    #[inline(always)]
    fn get(self) -> i32 {
        self as i32
    }
    #[inline(always)]
    fn put(v: i32) -> Self {
        v.clamp(0, u16::MAX as i32) as u16
    }
}

/// One sample plane (row-major), `u8` for an 8-bit component and `u16`
/// for a deeper one.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Samples {
    U8(Vec<u8>),
    U16(Vec<u16>),
}

impl Default for Samples {
    fn default() -> Self {
        Samples::U8(Vec::new())
    }
}

/// A borrowed run of samples of a [`Samples`].
#[derive(Clone, Copy, Debug)]
pub enum SamplesRef<'a> {
    U8(&'a [u8]),
    U16(&'a [u16]),
}

/// A mutably borrowed run of samples of a [`Samples`].
#[derive(Debug)]
pub enum SamplesMut<'a> {
    U8(&'a mut [u8]),
    U16(&'a mut [u16]),
}

/// Whether a component of `bit_depth` bits stores as `u16`.
#[inline]
pub fn is_wide(bit_depth: u32) -> bool {
    bit_depth > 8
}

impl Samples {
    /// A zero-filled plane of `len` samples for a component of
    /// `bit_depth` bits.
    pub fn zeroed(len: usize, bit_depth: u32) -> Self {
        if is_wide(bit_depth) {
            Samples::U16(vec![0; len])
        } else {
            Samples::U8(vec![0; len])
        }
    }

    /// A plane holding `values` (each stored as by [`Sample::put`]).
    pub fn from_i32(values: &[i32], bit_depth: u32) -> Self {
        if is_wide(bit_depth) {
            Samples::U16(values.iter().map(|&v| u16::put(v)).collect())
        } else {
            Samples::U8(values.iter().map(|&v| u8::put(v)).collect())
        }
    }

    /// An empty plane of the same sample type.
    pub fn empty_like(&self) -> Self {
        match self {
            Samples::U8(_) => Samples::U8(Vec::new()),
            Samples::U16(_) => Samples::U16(Vec::new()),
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        match self {
            Samples::U8(v) => v.len(),
            Samples::U16(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Whether samples are stored as `u16`.
    #[inline]
    pub fn is_wide(&self) -> bool {
        matches!(self, Samples::U16(_))
    }

    /// Sample `i`.
    #[inline]
    pub fn get(&self, i: usize) -> i32 {
        match self {
            Samples::U8(v) => v[i] as i32,
            Samples::U16(v) => v[i] as i32,
        }
    }

    /// Store sample `i`.
    #[inline]
    pub fn set(&mut self, i: usize, value: i32) {
        match self {
            Samples::U8(v) => v[i] = u8::put(value),
            Samples::U16(v) => v[i] = u16::put(value),
        }
    }

    /// Set every sample to `value`.
    pub fn fill(&mut self, value: i32) {
        match self {
            Samples::U8(v) => v.fill(u8::put(value)),
            Samples::U16(v) => v.fill(u16::put(value)),
        }
    }

    /// Set sample `i` to `f(i)` for every `i`.
    pub fn fill_with(&mut self, mut f: impl FnMut(usize) -> i32) {
        match self {
            Samples::U8(v) => v
                .iter_mut()
                .enumerate()
                .for_each(|(i, s)| *s = u8::put(f(i))),
            Samples::U16(v) => v
                .iter_mut()
                .enumerate()
                .for_each(|(i, s)| *s = u16::put(f(i))),
        }
    }

    /// The samples as working values.
    pub fn to_i32_vec(&self) -> Vec<i32> {
        match self {
            Samples::U8(v) => v.iter().map(|&s| s.get()).collect(),
            Samples::U16(v) => v.iter().map(|&s| s.get()).collect(),
        }
    }

    /// Borrow samples `range`.
    #[inline]
    pub fn slice(&self, range: std::ops::Range<usize>) -> SamplesRef<'_> {
        match self {
            Samples::U8(v) => SamplesRef::U8(&v[range]),
            Samples::U16(v) => SamplesRef::U16(&v[range]),
        }
    }

    /// Borrow every sample.
    #[inline]
    pub fn as_ref(&self) -> SamplesRef<'_> {
        match self {
            Samples::U8(v) => SamplesRef::U8(v),
            Samples::U16(v) => SamplesRef::U16(v),
        }
    }

    /// Mutably borrow every sample.
    #[inline]
    pub fn as_mut(&mut self) -> SamplesMut<'_> {
        match self {
            Samples::U8(v) => SamplesMut::U8(v),
            Samples::U16(v) => SamplesMut::U16(v),
        }
    }

    /// Mutably borrow samples `range`.
    #[inline]
    pub fn slice_mut(&mut self, range: std::ops::Range<usize>) -> SamplesMut<'_> {
        match self {
            Samples::U8(v) => SamplesMut::U8(&mut v[range]),
            Samples::U16(v) => SamplesMut::U16(&mut v[range]),
        }
    }

    /// Store `src` (working values) at samples `off..off + src.len()`.
    #[inline]
    pub fn write_i32(&mut self, off: usize, src: &[i32]) {
        self.slice_mut(off..off + src.len()).write_i32(src);
    }

    /// Load samples `off..off + dst.len()` into `dst`.
    #[inline]
    pub fn read_i32(&self, off: usize, dst: &mut [i32]) {
        self.slice(off..off + dst.len()).read_i32(dst);
    }

    /// Copy samples `src_range` of `src` (same sample type) to `dst_off`.
    #[inline]
    pub fn copy_from(&mut self, dst_off: usize, src: SamplesRef<'_>) {
        let n = src.len();
        self.slice_mut(dst_off..dst_off + n).copy_from(src);
    }

    /// Append `src` (same sample type).
    pub fn extend_from(&mut self, src: SamplesRef<'_>) {
        match (self, src) {
            (Samples::U8(d), SamplesRef::U8(s)) => d.extend_from_slice(s),
            (Samples::U16(d), SamplesRef::U16(s)) => d.extend_from_slice(s),
            _ => panic!("sample plane width mismatch"),
        }
    }
}

impl<'a> SamplesRef<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SamplesRef::U8(v) => v.len(),
            SamplesRef::U16(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    #[inline]
    pub fn get(&self, i: usize) -> i32 {
        match self {
            SamplesRef::U8(v) => v[i] as i32,
            SamplesRef::U16(v) => v[i] as i32,
        }
    }

    /// Samples `range`.
    #[inline]
    pub fn slice(&self, range: std::ops::Range<usize>) -> SamplesRef<'a> {
        match *self {
            SamplesRef::U8(v) => SamplesRef::U8(&v[range]),
            SamplesRef::U16(v) => SamplesRef::U16(&v[range]),
        }
    }

    /// Samples from `start` on.
    #[inline]
    pub fn slice_from(&self, start: usize) -> SamplesRef<'a> {
        match *self {
            SamplesRef::U8(v) => SamplesRef::U8(&v[start..]),
            SamplesRef::U16(v) => SamplesRef::U16(&v[start..]),
        }
    }

    /// Load the samples into `dst` (`dst.len()` of them).
    #[inline]
    pub fn read_i32(&self, dst: &mut [i32]) {
        match self {
            SamplesRef::U8(v) => {
                for (d, &s) in dst.iter_mut().zip(v.iter()) {
                    *d = s as i32;
                }
            }
            SamplesRef::U16(v) => {
                for (d, &s) in dst.iter_mut().zip(v.iter()) {
                    *d = s as i32;
                }
            }
        }
    }

    /// An owned plane holding these samples.
    pub fn to_plane(&self) -> Samples {
        match self {
            SamplesRef::U8(v) => Samples::U8(v.to_vec()),
            SamplesRef::U16(v) => Samples::U16(v.to_vec()),
        }
    }
}

impl<'a> SamplesMut<'a> {
    #[inline]
    pub fn len(&self) -> usize {
        match self {
            SamplesMut::U8(v) => v.len(),
            SamplesMut::U16(v) => v.len(),
        }
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// An empty borrow (for `std::mem::take`-style splitting).
    #[inline]
    pub fn empty(&self) -> SamplesMut<'a> {
        match self {
            SamplesMut::U8(_) => SamplesMut::U8(&mut []),
            SamplesMut::U16(_) => SamplesMut::U16(&mut []),
        }
    }

    /// Split at sample `mid`, consuming the borrow.
    #[inline]
    pub fn split_at(self, mid: usize) -> (SamplesMut<'a>, SamplesMut<'a>) {
        match self {
            SamplesMut::U8(v) => {
                let (a, b) = v.split_at_mut(mid);
                (SamplesMut::U8(a), SamplesMut::U8(b))
            }
            SamplesMut::U16(v) => {
                let (a, b) = v.split_at_mut(mid);
                (SamplesMut::U16(a), SamplesMut::U16(b))
            }
        }
    }

    /// Samples from `start` on, consuming the borrow.
    #[inline]
    pub fn into_tail(self, start: usize) -> SamplesMut<'a> {
        match self {
            SamplesMut::U8(v) => SamplesMut::U8(&mut v[start..]),
            SamplesMut::U16(v) => SamplesMut::U16(&mut v[start..]),
        }
    }

    /// Reborrow samples `range`.
    #[inline]
    pub fn slice_mut(&mut self, range: std::ops::Range<usize>) -> SamplesMut<'_> {
        match self {
            SamplesMut::U8(v) => SamplesMut::U8(&mut v[range]),
            SamplesMut::U16(v) => SamplesMut::U16(&mut v[range]),
        }
    }

    /// Reborrow samples `range` immutably.
    #[inline]
    pub fn slice(&self, range: std::ops::Range<usize>) -> SamplesRef<'_> {
        match self {
            SamplesMut::U8(v) => SamplesRef::U8(&v[range]),
            SamplesMut::U16(v) => SamplesRef::U16(&v[range]),
        }
    }

    /// Store `src` (working values, `src.len()` of them).
    #[inline]
    pub fn write_i32(&mut self, src: &[i32]) {
        match self {
            SamplesMut::U8(v) => {
                for (d, &s) in v.iter_mut().zip(src) {
                    *d = u8::put(s);
                }
            }
            SamplesMut::U16(v) => {
                for (d, &s) in v.iter_mut().zip(src) {
                    *d = u16::put(s);
                }
            }
        }
    }

    /// Copy `src` (same sample type and length) into this run.
    #[inline]
    pub fn copy_from(&mut self, src: SamplesRef<'_>) {
        match (self, src) {
            (SamplesMut::U8(d), SamplesRef::U8(s)) => d.copy_from_slice(s),
            (SamplesMut::U16(d), SamplesRef::U16(s)) => d.copy_from_slice(s),
            _ => panic!("sample plane width mismatch"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn width_follows_bit_depth() {
        assert!(!Samples::zeroed(4, 8).is_wide());
        for bd in 9..=14 {
            assert!(Samples::zeroed(4, bd).is_wide());
        }
    }

    #[test]
    fn store_saturates_and_round_trips() {
        let mut p = Samples::zeroed(3, 8);
        p.write_i32(0, &[-5, 128, 300]);
        assert_eq!(p.to_i32_vec(), vec![0, 128, 255]);
        let mut q = Samples::zeroed(2, 14);
        q.write_i32(0, &[16383, -1]);
        assert_eq!(q.to_i32_vec(), vec![16383, 0]);
    }

    #[test]
    fn split_copy_and_extend() {
        let mut p = Samples::from_i32(&[1, 2, 3, 4, 5, 6], 10);
        let (mut a, b) = p.as_mut().split_at(3);
        a.copy_from(b.slice(0..3));
        assert_eq!(p.to_i32_vec(), vec![4, 5, 6, 4, 5, 6]);
        let mut e = p.empty_like();
        e.extend_from(p.slice(1..3));
        assert_eq!(e.to_i32_vec(), vec![5, 6]);
    }
}
