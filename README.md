# RealTime-PeopleCounting-withAlerts
Crowd detection system using MobileNet SSD to check COVID-19 social distancing which counts the number of people at a location taking CCTV feed as input. Alerts are sent when number of people exceed the threshold set for the location.

To run, execute the following terminal command: python run.py

Used Centroid tracker and Dlib’s correlation tracking algorithm for object tracking and assigning each person a unique ID.

Customizations available:
1) AlertLogs.csv to show the timestamp, people count and whether or not the people limit is exceeded.
2) Logs.csv to show the timestamp, people count, people inside the area who have entered, people outside the area who have exited and whether or not the people limit is exceeded.
3) Set the configurations in 'config.py' in 'mylib' folder. Services available are: 
  
    (i) Mailer service to send email alerts to mail IDs.
  
    (ii) Threshold i.e. the limit of the people count beyond which the alert will be activated.
  
    (iii) Scheduler: Auto run/Schedule the software to run at your desired time.
  
    (iv) Timer: Auto stop the software after certain a time/hours.
