# Moonless night
This example is a Moonless night with part of an ISS pass.

## The source image captured by allsky
**NOTE** For testing this image has had all of the annotations removed. This was to test the star count without using the manually created mask. Normally the manually created mask will remove the annotations anyway

![Source Image](../../docimages/asi178mm/nomoon/source.png?raw=true "Source Image")

## The manually created mask
![Image Mask](../../docimages/asi178mm/nomoon/mask.png?raw=true "Image Mask")

## The auto created mask

**NOTE** Since there are no large bright areas in the image the auto mask is all white
![Auto Mask](../../docimages/asi178mm/nomoon/automask.png?raw=true "Auto Mask")

## The masked image
![Masked Image](../../docimages/asi178mm/nomoon/masked.png?raw=true "Masked Image")

## The clean image after the star counting
![Counted Stars](../../docimages/asi178mm/nomoon/clean.png?raw=true "Counted Stars")

## The final image annotated to show the stars
![Counted Stars Annotated](../../docimages/asi178mm/nomoon/annotated.png?raw=true "Counted Stars Annotated")

# Output from the test
```
starcounttest.py -vvv -i testimages/nomoon2048x2048-1.5mm-178mm.png -m masks/nomoon2048x2048-1.5mm-178mm.png 
Init Complete took 0.33 Seconds. Elapsed Time 0.33 Seconds.
De Noise Complete took 0.0 Seconds. Elapsed Time 0.33 Seconds.
Contrast Adjustment Complete took 0.06 Seconds. Elapsed Time 0.39 Seconds.
Auto Brightness Complete took 0.05 Seconds. Elapsed Time 0.44 Seconds.
mask Creation Complete took 0.87 Seconds. Elapsed Time 1.31 Seconds.
StarCount Complete - 581 Stars Found took 5.39 Seconds. Elapsed Time 6.69 Seconds.
```