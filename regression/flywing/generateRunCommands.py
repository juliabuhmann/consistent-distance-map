for i in range(0,11):
    print '%run apply_to_flywing.py /Users/jug/Volumes/FastData/ProjectWithJJL/Flywing/Medium/Predictions/prediction'+str(i)+'.tif /Users/jug/Volumes/FastData/ProjectWithJJL/Flywing/Medium/GroundTruth/t0'+str(60+i)+'.tif /Users/jug/Volumes/FastData/ProjectWithJJL/temp /Volumes/FastData/ProjectWithJJL/Flywing/Medium/Smoothed/t0'+str(60+i)+'.tif --sx 185 --sy 193 -n --ns'
