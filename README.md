# football_2d_map
This projects gets a video of a football match. detects players and referees and makes a 2d map of their locations 


In this project I have made a 2d map of a football field with python and OpenCV. 
It detects players and referees with Binary Morphology and Background Subtraction and Classifies them with CNN. 
Then it builds the map with homography

There is another file named tracker.py which uses sparce optical flow techniques to track players within frames.
This was implemented to avoid detection in every frame and enhance performance
