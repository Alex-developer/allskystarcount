# allskystarcount

Testing the ability to count the number of stars in an image captured by Thomas Jacquin's allsky software - https://github.com/thomasjacquin/allsky

This is to determine if the sky is clear or not. This is usefull for us astro photographers

**THIS CODE IS BEING TESTED AND SHOULD NOT BE USED IN PRODUCTION AS IT WILL CHANGE FREQUENTLY**

# How does it work
The current processing has been created by me messign about with various ideas. Currently the flow is as follows

1) Apply a mask to the image from allsky. This mask is created manually and designed to remove areas of the image that we know we dont wanto to check. In my example below the mask is used to remove the house and streetlights
2) De Noise the image, this is to help reduce any false positives
3) Adjust the contract in the image to help pull out any stars
4) Create an automatic mask to remove any large bright areas of the image, this is intended to try and deal with the Moon
5) Count the stars using either external or internal star templates

# Installation
At present there is no automated way to install this as its for testing only. If you install my allsky annotater it will install all of the dependencies required for this script to run. 

The python dependencies are

- python3
  - cv2
  - ephem
  - numpy 
  - skyimage

# Examples
The examples folder contains several examples of running the script on various camreas under a variety of conditions.

**NOTE** I am adding more examples, I have a load for the pi hq camera to add**

- ASI120MC
- ASI178MM
  - [A Moonless night](examples/asi178mm/moonlessnight.md)
  - [A Bright Moon](examples/asi178mm/brightmoon.md)  
- PiHQ Camera
  - [A Low Moon](examples/pihq/lowmoon.md)  

As you can see from the examples there is still a lot of work to do. Some of the issues I need to deal with

- The denoise is VERY slow but does yield better results
- Aircraft are a pain. The flashing lights appears as stars
- Scratches in the dome generate a lot fo false positives when the Moon is bright
- Rain confuses the algorithms and generates a lot of false positives

# Temperature Method
Currently I use a different method for calculating if the sky is clear by measuring the difference between the ambient and sky temperatures.

I use a MLX90614-DCI to measure the ambient and sky temperature and then calculate, over time, if the sky is clear or not. (**NOTE the MLX90614-DCI has a 5 degree FOV whereas the MLX90614 has a 90 degree FOV***). The solution is not perfect but from my testing is about 90% accurate.

The obvious issue with this method is it requires external hardware. The MLX90614-DCI is quite hard to get (In the UK), the MLX90614 is easily available.

![AG Method](docimages/agmethod.png?raw=true "Temperature Method")