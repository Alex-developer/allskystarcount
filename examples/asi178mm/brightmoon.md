# A Bright Moon
This example is a bright Moon with high cloud on the horizon.

## The source image captured by allsky
**NOTE** For testing this image has had all of the annotations removed. This was to test the star count without using the manually created mask. Normally the manually created mask will remove the annotations anyway

![Source Image](../../docimages/asi178mm/brightmoon/source.png?raw=true "Source Image")

## The manually created mask
![Image Mask](../../docimages/asi178mm/brightmoon/mask.png?raw=true "Image Mask")

## The auto created mask

**NOTE** The white areas are the parts we will kepp int he final image

![](../../docimages/asi178mm/brightmoon/automask.png?raw=true "Auto Mask")
![](../../docimages/asi178mm/brightmoon/automask.png?raw=true "Auto Mask")

## The masked image
![Masked Image](../../docimages/asi178mm/brightmoon/masked.png?raw=true "Masked Image")

## The clean image after the star counting
![Counted Stars](../../docimages/asi178mm/brightmoon/clean.png?raw=true "Counted Stars")

## The final image annotated to show the stars
![Counted Stars Annotated](../../docimages/asi178mm/brightmoon/annotated.png?raw=true "Counted Stars Annotated")

# Output fromt the test
```
starcounttest.py -vvv -i testimages/test4.png -m masks/2048x2048-1.5mm-178mm.png 
Init Complete took 0.33 Seconds. Elapsed Time 0.33 Seconds.
De Noise Complete took 0.0 Seconds. Elapsed Time 0.33 Seconds.
Contrast Adjustment Complete took 0.05 Seconds. Elapsed Time 0.38 Seconds.
Auto Brightness Complete took 0.05 Seconds. Elapsed Time 0.43 Seconds.
mask Creation Complete took 1.33 Seconds. Elapsed Time 1.76 Seconds.
StarCount Complete - 77 Stars Found took 1.95 Seconds. Elapsed Time 3.71 Seconds.
```