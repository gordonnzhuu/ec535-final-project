# Commands to run on BeagleBone Black (BBB) with camera

### Shows camera resolutions and formats
```
v4l2-ctl --list-formats-ext -d /dev/video0
```

### Take a picture 
```
fswebcam -d /dev/video0 -r 640x480 --no-banner test.jpg
```

### Settings
```
v4l2-ctl -d /dev/video0 --list-ctrls
v4l2-ctl -d /dev/video0 --set-ctrl=brightness=128
```

### Take a picture
```
fswebcam -d /dev/video0 -r 1280x960 --no-banner test.jpg
```

### View image on Python server
```
http://168.122.131.226:8000/test.jpg
```

### Host server on BBB
```
python3 -m http.server 8000
```

### SSH into BBB
```
ssh debian@192.168.7.2
```