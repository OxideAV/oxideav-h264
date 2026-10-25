#!/bin/sh
# Regenerates the output_params fixtures with black-box ffmpeg/libx264 and
# records what ffprobe reports for each in expected.txt. Run from this
# directory: sh gen.sh
#
# Notes: on a raw H.264 elementary stream ffprobe's r_frame_rate is the
# VUI tick rate time_scale / num_units_in_tick, twice the §E.2.1 frame
# rate; libx264 has no 4:0:0 output, so the monochrome layout is covered
# by the crate's own encoder inside the test.
set -e
enc() { # name pix_fmt size rate extra-x264-params
  ffmpeg -v error -y -f lavfi -i "testsrc2=size=$3:rate=$4:duration=0.2" \
    -pix_fmt "$2" -c:v libx264 -preset ultrafast -bf 2 -g 5 \
    -x264-params "$5" -f h264 "$1.h264"
}
enc yuv420p_70x46_25  yuv420p     70x46 25    "colorprim=bt709:transfer=bt709:colormatrix=bt709"
enc yuv422p_64x48_30  yuv422p     64x48 30000/1001 "fullrange=on:colorprim=bt470bg:transfer=smpte170m:colormatrix=bt470bg"
enc yuv444p_48x32_24  yuv444p     48x32 24    ""
enc yuv420p10_64x48_50 yuv420p10le 64x48 50   "colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc"
enc yuv422p10_48x32_25 yuv422p10le 48x32 25   ""
enc yuv444p10_32x32_25 yuv444p10le 32x32 25   ""
# Mid-stream change: two coded video sequences of different shapes.
ffmpeg -v error -y -f lavfi -i "testsrc2=size=64x48:rate=25:duration=0.2" -pix_fmt yuv420p \
  -c:v libx264 -preset ultrafast -bf 2 -g 5 -f h264 part_a.h264
ffmpeg -v error -y -f lavfi -i "testsrc2=size=48x32:rate=25:duration=0.2" -pix_fmt yuv420p10le \
  -c:v libx264 -preset ultrafast -bf 2 -g 5 -f h264 part_b.h264
cat part_a.h264 part_b.h264 > change_64x48_8bit_then_48x32_10bit.h264
rm -f part_a.h264 part_b.h264
: > expected.txt
for f in yuv*.h264; do
  printf '%s ' "$f" >> expected.txt
  ffprobe -v error -select_streams v:0 -count_frames \
    -show_entries stream=width,height,pix_fmt,r_frame_rate,color_range,color_space,color_primaries,color_transfer,nb_read_frames \
    -of default=noprint_wrappers=1 "$f" | tr '\n' ' ' >> expected.txt
  echo >> expected.txt
done
