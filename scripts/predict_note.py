from basic_pitch.inference import predict_and_save
from basic_pitch import ICASSP_2022_MODEL_PATH

predict_and_save(
    audio_path="data\mixkit-guitar-stroke-down-slow-2339.wav",      
    output_directory="output",             
    model_path=ICASSP_2022_MODEL_PATH    
)