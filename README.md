Gait analysis program
This is a small python-program, that estimates the length of your steps, kadance and walking speed from a small video of a person walking. 
The program uses MediaPipe to determine different body parts and then calculates the relation between pixels and meters, from the persons written height.
The program also provides two graphs, one showing the distance between the angles, which makes it possible to determine the maximum number of pixels between each step. The other graph shows the number of pixels between the heel and the nose over time, which makes it possible to determine the most precise number of pixels between the nose and the heel, which is assumed to be equivalent of 90% of the entered height. From this, the program determines the distance of a single pixel and hereby the distance of a full step.

Video recording demands
The person walking in the video should be walking in a straight line perpendicular to the camera. Preferable the person should be walking about 5-7 meters from the camera, and the full body must be visible to the camera. This way MediaPipe can find the specific body parts in the video and get a reasonable amount of steep to make a proper estimation of the walking speed aswell.

Installation
```
git clone https://github.com/maxemilroed-del/gait-analysis-program.git
cd gait-analysis-program
pip install -r requirements.txt
```

Get the model and put it in the program folder
https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

1.	Put your video in the folder and correct the filename in the bottom of ´video_breakdown.py´.
2.	Run ´python video_breakdown.py (make ´landmarks.csv´)
3.	Run ´python Analyse.py´ and enter the persons height in meter.
Constraints:
The program needs the person to walk in a straight line perpendicular to the camera.
The scale presumes that the nose-heel is approximately 90% of the entered height.

There is no video of a person walking in the repository, since it is person sensitive data.

Use of AI
I have made this code with help from Claude. I have modified and corrected parts of the code myself.
