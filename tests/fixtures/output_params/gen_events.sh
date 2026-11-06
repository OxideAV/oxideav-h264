#!/bin/sh
# Regenerates the Frame::StreamParams splice fixtures with black-box
# ffmpeg/libx264 (run from this directory: sh gen_events.sh). Each
# segment is its own coded video sequence (IDR + SPS/PPS, 3 B-frames
# between anchors so the §C.4 reorder queue holds old-shape pictures
# when the new SPS arrives); the segments are concatenated as raw
# Annex B byte streams.
set -e
seg() { # out size pix_fmt
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=$2:rate=25:duration=0.4" \
    -pix_fmt "$3" -c:v libx264 -preset ultrafast -bf 3 -g 6 -f h264 "$1"
}
seg s_a.h264 64x48 yuv420p
seg s_b.h264 80x46 yuv420p
seg s_c.h264 80x46 yuv420p10le
seg s_d.h264 80x46 yuv422p
# size, crop + size, 8 -> 10-bit, 10-bit 4:2:0 -> 8-bit 4:2:2, back to the first shape
cat s_a.h264 s_b.h264 s_c.h264 s_d.h264 s_a.h264 > splice_5_shapes.h264
# two coded video sequences of one shape: no event
cat s_a.h264 s_a.h264 > splice_same_shape.h264
rm -f s_a.h264 s_b.h264 s_c.h264 s_d.h264
: > expected_events.txt
for f in splice_*.h264; do
  printf '%s ' "$f" >> expected_events.txt
  ffprobe -v error -select_streams v:0 -show_entries frame=width,height,pix_fmt \
    -of csv=p=0 "$f" | sed "s/,*$//" | uniq -c | awk '{printf "%s:%s ", $1, $2}' >> expected_events.txt
  echo >> expected_events.txt
done
