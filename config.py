from typing import Mapping
import enum

class generate_label(enum.IntEnum):
    W = 0 # ang
    T = 1 # sad
    F = 2 # hap
    N = 3 # neu
    A = 4 # fear
    others = 5

timesteps = 350
feat_dim = 512
emotion_steps = [{'F':100, 'W':45, 'N':350, 'T':45 }] # F9
em_steps_test = [{'F':timesteps, 'W':timesteps, 'N':timesteps, 'T':timesteps}] # F9
speakers = ['03', '08', '09', '10', '11', '12', '13', '14', '15', '16']

save_dir = './checkpoint'
model_name = 'best_model_emodb.pth'

learning_rate = 1e-2
clip = 0

max_epochs = 50