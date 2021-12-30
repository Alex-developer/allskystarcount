import os
import sys
import cv2
import ephem
import numpy 
import pathlib
import re
import csv
import json

from datetime import datetime
from datetime import date
from time import time

from math import radians
from math import degrees
from math import sqrt

from skimage import measure
import numpy as np
import argparse

from pathlib import Path

import cv2

class STARCOUNT():
    _starTemplatesSrc = []
    _starTemplates = []

    _imageFileSrc = "testimages/test1.png"
    _imageMaskSrc = "starmasks/mask.png"
    _sourceImage = None
    _imageMask = None
    _outputFile = "outputFile"
    _imageExtension = ""

    _detectionThreshold = 0.5
    _distanceThreshold = 10

    _observerLat = "52N"
    _observerLon = "0.2E"
    _moonAzimuth = None
    _moonElevation = None
    _moonIllumination = None

    _isRaining = False
    _lastRainTime = datetime.now()

    _detectionThreshold = 0.6
    _distanceThreshold = 20
    _starTemplate = 5
    _starCount = 0

    _startTime = 0
    _lastTimer = None
    
    _allSkyHomeDirectory = None
    _allSkyVariables = None
    _cameraConfig = None

    _logLevel = False

    def __init__(self, logLevel):    
        self._checkForAllsky()
        self._logLevel = logLevel
    
    def processImage(self):
        self._startTime = time()
        if self._loadImage():
            self._initialiseMoon()
            if self._loadStarTemplate():
                self._timer("Init Complete")       
                #self._deNoiseImage()
                self._timer("De Noise Complete")
                self._adjustContrast()
                self._timer("Contrast Adjustment Complete")
                self._automatic_brightness_and_contrast()
                self._timer("Auto Brightness Complete")
                self._maskImage()
                self._timer("mask Creation Complete")
                self._countStars()
                self._timer("StarCount Complete - " + str(self._starCount) + " Stars Found")            

    def processDirectory(self, directory):
        if Path(directory).is_dir():
            for filename in os.listdir(directory):
                f = os.path.join(directory, filename)
                if os.path.isfile(f):
                    self._imageFileSrc = f
                    p = Path(f)
                    self._outputFile = os.path.join(directory, p.stem)
                    self.processImage()
        else:
            print("ERROR: Directory", directory, "Not Found")

    def _checkForAllsky(self):
        try:
            self._allSkyHomeDirectory = os.environ['ALLSKY_HOME']

            with open(self._allSkyHomeDirectory + "/config/config.sh") as stream: # dont hard code dir seps
                contents = stream.read().strip()

            var_declarations = re.findall(r"^[a-zA-Z0-9_]+=.*$", contents, flags=re.MULTILINE)
            reader = csv.reader(var_declarations, delimiter="=")
            self._allSkyVariables = dict(reader)  

            try:
                if self._allSkyVariables["CAMERA"] == "ZWO":
                    camerSettingsFile = "/etc/raspap/settings_ZWO.json"
                else:
                    camerSettingsFile = "/etc/raspap/settings_RPiHQ.json"

                allskySettingsFile = open(camerSettingsFile, 'r')
                self._cameraConfig = json.load(allskySettingsFile)

                self._observerLat = self._cameraConfig["latitude"]
                self._observerLon = self._cameraConfig["longitude"]
            except FileNotFoundError:
                pass
        except KeyError:
            pass

    def _convertLatLon(self, input):
        multiplier = 1 if input[-1] in ['N', 'E'] else -1
        return multiplier * sum(float(x) / 60 ** n for n, x in enumerate(input[:-1].split('-')))

    def _automatic_brightness_and_contrast(image, clip_hist_percent = 0):
        gray = image

        # Calculate grayscale histogram
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(image, alpha = alpha, beta = beta)
        return (auto_result, alpha, beta)

    def _loadImage(self):
        imageLoaded = True
        try:
            self._sourceImage = cv2.imread(self._imageFileSrc,cv2.IMREAD_GRAYSCALE)
            self._imageExtension = pathlib.Path(self._imageFileSrc).suffix
            if self._imageMaskSrc is not None:
                self._imageMask = cv2.imread(self._imageMaskSrc,cv2.IMREAD_GRAYSCALE)
                if self._sourceImage.shape == self._imageMask.shape:
                    self._sourceImage = cv2.bitwise_and(src1=self._sourceImage, src2=self._imageMask)
                else:
                    print("ERROR: Image mask and caputred frame are different dimensions. Allsky Image is", self._sourceImage.shape, " mask is",  self._imageMask.shape)
                    imageLoaded = False
        except Exception as e:
            print("ERROR: File " + self._imageFileSrc + " not found")
            imageLoaded = False

        return imageLoaded

    def _loadStarTemplate(self):
        templateOk = True

        if len(self._starTemplatesSrc) != 0:
            for template in self._starTemplatesSrc:
                self._starTemplates.append(cv2.imread(template, cv2.IMREAD_GRAYSCALE))
        else:
            starTemplate = numpy.zeros([20, 20], dtype=numpy.uint8)
            cv2.circle(
                img=starTemplate,
                center=(9, 9),
                radius=2,
                color=(255, 255, 255),
                thickness=cv2.FILLED,
            )

            starTemplate = cv2.blur(
                src=starTemplate,
                ksize=(3, 3),
            )

            self._starTemplates.append(starTemplate)
            if self._logLevel > 3:
                cv2.imwrite("autostartemplate1.png", starTemplate, params=None)

            starTemplate = numpy.zeros([20, 20], dtype=numpy.uint8)
            cv2.circle(
                img=starTemplate,
                center=(9, 9),
                radius=3,
                color=(255, 255, 255),
                thickness=cv2.FILLED,
            )

            starTemplate = cv2.blur(
                src=starTemplate,
                ksize=(5, 5),
            )
            self._starTemplates.append(starTemplate)            
            if self._logLevel > 3:
                cv2.imwrite("autostartemplate2.png", starTemplate, params=None)

        for index, template in enumerate(self._starTemplates):
            height, width = template.shape

            if height > self._distanceThreshold or width > self._distanceThreshold:
                print("ERROR: Star", index, "template is bigger than the distance threshold")
                templateOk = False
        
        return templateOk

    def _initialiseMoon(self):
        lat = radians(self._convertLatLon(self._observerLat))
        lon = radians(self._convertLatLon(self._observerLon))

        observer = ephem.Observer()  
        observer.lat = lat
        observer.long = lon 
        moon = ephem.Moon()      
        observer.date = date.today()
        observer.date = datetime.now()        
        moon.compute(observer)  
            
        self._moonAzimuth = moon.az
        self._moonElevation = degrees(moon.alt)
        self._moonIllumination = round(moon.phase, 2)

    def _processImage(self):
        processImage = True
        # If Moon greater than 50% and more than 10 deegrees above the horizon forget trying to process
        if self._moonIllumination > 50:
            if self._moonElevation > 10:
                processImage = False

        #If raining or rained in the last 30 mins don't process the image
        #isRaining, lastRainTime = rainSensor.getRainStatus()
        if self._isRaining:
            processImage = False
        else:
            if self.__lastRainTime is not None:
                mins = self.getTimeDifferenceFromNow(self._lastRainTime, datetime.now())
                if mins < 30:
                    processImage = False

        return processImage

    def _deNoiseImage(self):
         self._sourceImage = cv2.fastNlMeansDenoising(self._sourceImage)

    def _automatic_brightness_and_contrast(self, clip_hist_percent = 0):


        # Calculate grayscale histogram
        hist = cv2.calcHist([self._sourceImage], [0], None, [256], [0, 256])
        hist_size = len(hist)

        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index - 1] + float(hist[index]))

        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum / 100.0)
        clip_hist_percent /= 2.0

        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1

        # Locate right cut
        maximum_gray = hist_size - 1
        while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
            maximum_gray -= 1

        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha

        auto_result = cv2.convertScaleAbs(self._sourceImage, alpha = alpha, beta = beta)
        return (auto_result, alpha, beta)

    def _adjustContrast(self, contrast = 10):
        Alpha = float(131 * (contrast + 127)) / (127 * (131 - contrast))
        Gamma = 127 * (1 - Alpha)
        self._sourceImage = cv2.addWeighted(self._sourceImage, Alpha,self._sourceImage, 0, Gamma)

    def _maskImage(self):

        # threshold the image to reveal light regions in the
        # blurred image
        ret, thresh = cv2.threshold(self._sourceImage, 90, 255, cv2.THRESH_BINARY)

        # perform a series of erosions and dilations to remove
        # any small blobs of noise from the thresholded image
        thresh = cv2.erode(thresh, None, iterations=3)
        thresh = cv2.dilate(thresh, None, iterations=5)

        if self._logLevel > 2:
            self._saveImage(thresh, "-threshhold")

        # perform a connected component analysis on the thresholded
        # image, then initialize a mask to store only the "large"
        # components
        labels = measure.label(thresh, background=0)
        mask = np.zeros(thresh.shape, dtype="uint8")

        # loop over the unique components
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue

            # otherwise, construct the label mask and count the
            # number of pixels 
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            numPixels = cv2.countNonZero(labelMask)

            # if the number of pixels in the component is sufficiently
            # large, then add it to our mask of "large blobs"
            if numPixels > 400:
                mask = cv2.add(mask, labelMask)


        mask2 = cv2.bitwise_not(mask)

        if self._logLevel > 2:
            self._saveImage(mask2, "-mask")

        dst = cv2.addWeighted(self._sourceImage, 0.7, mask, 0.3, 0)
        
        if self._logLevel > 2:        
            self._saveImage(dst, "-masked")

        if self._logLevel > 2:
            cv2.imwrite("calc-mask.png", mask2, params=None)
        
        if self._imageMask is not None:
            self._sourceImage = cv2.bitwise_and(src1=self._sourceImage, src2=self._imageMask)         
        self._sourceImage = cv2.bitwise_and(src1=self._sourceImage, src2=mask2)    

    def _countStars(self):
        detectedImageClean = self._sourceImage.copy()
        sourceImageCopy = self._sourceImage.copy()
        
        starList = list()
        for index, template in enumerate(self._starTemplates):

            templateWidth, templateHeight = template.shape[::-1]

            try:
                result = cv2.matchTemplate(sourceImageCopy, template, cv2.TM_CCOEFF_NORMED)
            except:
                print("Star template match failed", file=sys.stderr)
            else:
                loc = numpy.where(result >= self._detectionThreshold)

                templateStarList = list()
                for pt in zip(*loc[::-1]):
                    for star in starList:
                        distance = sqrt(((pt[0] - star[0]) ** 2) + ((pt[1] - star[1]) ** 2))
                        if (distance < self._distanceThreshold):
                            break
                    else:
                        starList.append(pt)
                        templateStarList.append(pt)

                wOffset = int(templateWidth/2)
                hOffset = int(templateHeight/2)

                for star in templateStarList:
                    if index == 0:
                        cv2.circle(self._sourceImage, (star[0] + wOffset, star[1] + hOffset), 10, (255, 255, 255), 1)
                        pass
                    else:
                        cv2.rectangle(self._sourceImage, (star[0], star[1]), (star[0]+templateWidth, star[1] + templateHeight), (255, 255, 255), 1)

        self._starCount = len(starList)
        
        text = "Found " + str(self._starCount) + " Stars"
        if self._logLevel > 1:
            self._saveImage(detectedImageClean, "-clean", text)
        self._saveImage(self._sourceImage, "-result", text)

    def _saveImage(self, image, suffix, text = None):
        try:
            fileName = self._outputFile + suffix + self._imageExtension
            if text is not None:
                font = cv2.FONT_HERSHEY_SIMPLEX
                textsize = cv2.getTextSize(text, font, 1, 2)[0]
                textX = int((image.shape[1] - textsize[0]) / 2)
                textY = int(textsize[1]) + 5
                cv2.putText(image, text, (textX,textY), font, 1, (255, 255, 255), 2, cv2.LINE_AA)              
            cv2.imwrite(fileName, image, params=None)
        except:
            print("Failed to write output file " + fileName, file=sys.stderr)        

    def _timer(self, text):
        if self._lastTimer is None:
            elapsedSinceLastTime = time() - self._startTime
        else:
            elapsedSinceLastTime = time() - self._lastTimer
        
        lastText = str(round(elapsedSinceLastTime,2))
        self._lastTimer = time()

        elapsedTime = time() - self._startTime
        print(text + " took " + lastText + " Seconds. Elapsed Time " + str(round(elapsedTime,2)) + " Seconds.")

    def setImage(self, image):
        self._imageFileSrc = image

    def setAllSkyImage(self):
        imageFile = os.sep.join([self._allSkyHomeDirectory, self._cameraConfig["filename"]])
        self._imageFileSrc = imageFile

    def disableManualMask(self):
        self._imageMaskSrc = None

    def setManualMask(self, maskFile):
        self._imageMaskSrc = maskFile

def main():
    LOG_LEVELS = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    DEFAULT_LOG_LEVEL = "INFO"
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("-a", "--allsky", action="store_true", help="Use the latest image from All Sky")   
    group.add_argument("-i", "--image", help="The path to the image file to process")    
    maskGroup = parser.add_mutually_exclusive_group()
    maskGroup.add_argument("-d", "--dm", action="store_true", help="Disable manual mask") 
    maskGroup.add_argument("-m", "--mask", help="Manual mask file") 
    parser.add_argument("-p", "--path", help="Process all files in the specified directory")    
    parser.add_argument("-v", "--verbose", action="append_const", dest="log_level", const=1, help="Process all files in the specified directory")    
    arguments = parser.parse_args()

    log_level = LOG_LEVELS.index(DEFAULT_LOG_LEVEL)
    for adjustment in arguments.log_level or ():
        log_level = min(len(LOG_LEVELS) - 1, max(log_level + adjustment, 0))

    starCounter = STARCOUNT(log_level)

    if arguments.image:
        starCounter.setImage(arguments.image)

    if arguments.allsky:
        starCounter.setAllSkyImage()

    if arguments.dm:
        starCounter.disableManualMask()

    if arguments.mask:
        starCounter.setManualMask(arguments.mask)

    if arguments.path:
        starCounter.processDirectory(arguments.path)
    else:
        starCounter.processImage()

if __name__ == "__main__":
    main()