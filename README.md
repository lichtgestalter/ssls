# SSLS - Star System Lightcurve Simulator
The SSLS calculates the movements and eclipses of celestial bodies and produces a video of this.<br>
Specify mass, radius and other properties of some stars and planets in a configuration file. Then run "ssls.py <configfilename>" to produce the video.<br>
The video shows simultanously a view of the star system from top and from the side and the lightcurve of the system's total luminosity over time.<br>
Usually you do not need to look at or even modify the python code. Instead control the program's outcome with the config file. The meaning of all program parameters is documented in the config file.<br>
SSLS uses ffmpeg to convert the data into a video. Download ffmpeg from https://www.ffmpeg.org/download.html. Extract the zip file and add "<yourdriveandpath>\FFmpeg\bin" to Environment Variable PATH.<br>
<br>
Your questions and comments are welcome.<br>
Just open an issue on https://github.com/lichtgestalter/ssls/issues to get my attention :)<br>
