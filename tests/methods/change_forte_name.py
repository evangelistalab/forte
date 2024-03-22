import re
import os

pattern = re.compile("import forte")
count = 0

string=os.getcwd()
my_forte = string.split('/')[-3]
forte_string = "import " + my_forte
for directory in os.listdir(os.getcwd()):
    if(os.path.isdir(directory)):
        os.chdir(directory)
        try:
            input_file = open( "input.dat", "r+") 
            fileContents = input_file.read()
            fileContents = pattern.sub(forte_string, fileContents)
            input_file.seek(0)
            input_file.write(fileContents)
            input_file.truncate()
            os.chdir('..')
        except IOError:
            print "Skipping %s" % directory
            continue



