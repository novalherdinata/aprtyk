import os
import subprocess

class Train:
    def __init__(self):
        self.setup_environment()

    def check_yolov5_folder(self):
        current_directory = os.getcwd()  # Get the current working directory
        if not current_directory.endswith('yolov5'):
            if not os.path.exists('yolov5'):
                subprocess.run(['git', 'clone', 'https://github.com/ultralytics/yolov5'])

    def change_directory(self):
        if os.path.exists('yolov5'):
            os.chdir('yolov5')

    def install_dependencies(self):
        subprocess.run(['pip', 'install', '-qr', 'requirements.txt'])

    def install_roboflow(self):
        subprocess.run(['pip', 'install', '-q', 'roboflow'])
        
    def setup_environment(self):
        self.check_yolov5_folder()
        self.change_directory()
        self.install_dependencies()
        self.install_roboflow()
