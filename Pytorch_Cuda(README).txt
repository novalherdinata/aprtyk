conda create -n ANPR-Py python=3.10 
conda activate ANPR-Py
conda install -c conda-forge cudatoolkit=11.8 cudnn=8.8.0
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install streamlit
pip3 install streamlit_option_menu
pip3 install easyocr