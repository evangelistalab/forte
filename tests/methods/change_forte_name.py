import re
import os

pattern = re.compile("import forte")
count = 0
for directory in os.listdir(os.getcwd()):
    if(os.path.isdir(directory)):
        os.chdir(directory)
        print directory
        with open( "input.dat", "r+") as input_file:
            fileContents = input_file.read()
            fileContents = pattern.sub("import forte_intel", fileContents)
            input_file.seek(0)
            input_file.write(fileContents)
            input_file.truncate()
            os.chdir('..')



