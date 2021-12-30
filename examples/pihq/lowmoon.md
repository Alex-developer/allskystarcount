# Moon Low to the horizon
This example of the Moon low to the horizon with either rain or condensation on the dome.

The images used here are courtesy of calberts  https://github.com/calberts Thanks :-)

## The source image captured by allsky
![Source Image](../../docimages/pihq/lowmoon/source.jpg?raw=true "Source Image")

## The manually created mask
![Image Mask](../../docimages/pihq/lowmoon/mask.png?raw=true "Image Mask")

## The auto created mask
![Auto Mask](../../docimages/pihq/lowmoon/automask.jpg?raw=true "Auto Mask")

## The masked image
![Masked Image](../../docimages/pihq/lowmoon/masked.jpg?raw=true "Masked Image")

## The clean image after the star counting
![Counted Stars](../../docimages/pihq/lowmoon/clean.jpg?raw=true "Counted Stars")

## The final image annotated to show the stars
![Counted Stars Annotated](../../docimages/pihq/lowmoon/annotated.jpg?raw=true "Counted Stars Annotated")

# Output from the test
```
starcounttest.py -vvv -i testimages/pihq/calberts1.jpg -m masks/pihq/calberts.png 
Init Complete took 0.21 Seconds. Elapsed Time 0.21 Seconds.
De Noise Complete took 0.0 Seconds. Elapsed Time 0.21 Seconds.
Contrast Adjustment Complete took 0.04 Seconds. Elapsed Time 0.25 Seconds.
Auto Brightness Complete took 0.03 Seconds. Elapsed Time 0.28 Seconds.
mask Creation Complete took 0.8 Seconds. Elapsed Time 1.09 Seconds.
StarCount Complete - 251 Stars Found took 2.26 Seconds. Elapsed Time 3.34 Seconds
```